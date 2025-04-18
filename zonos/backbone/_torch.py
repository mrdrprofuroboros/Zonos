# Based on gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/095b2229ee3a40e379c11f05b94bd6923db63b4b/model.py
import math

import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_with_kvcache
from torch.nn import functional as F

from zonos.config import BackboneConfig, InferenceParams


def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class TorchZonosBackbone(nn.Module):
    supported_architectures = ["transformer"]
    freqs_cis: torch.Tensor

    def __init__(self, config: BackboneConfig):
        assert not config.ssm_cfg, "This backbone implementation only supports the Transformer model."
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_layer))
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        self.freqs_cis = precompute_freqs_cis(16384, self.config.d_model // self.config.attn_cfg["num_heads"])
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams) -> torch.Tensor:
        input_pos = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        input_pos = input_pos + inference_params.lengths_per_sample.unsqueeze(-1)
        freqs_cis = self.freqs_cis[input_pos]

        for i, layer in enumerate(self.layers):
            # Extract cache for this specific layer
            layer_cache = inference_params.key_value_memory_dict[i]
            # Pass the extracted cache directly to the layer
            hidden_states = layer(hidden_states, inference_params, freqs_cis, layer_cache)
        return self.norm_f(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = Attention(config, layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // config.attn_cfg["num_heads"]

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        # Use paged KV cache
        paged_size = self.config.attn_cfg.get("paged_kv_block_size", 256)

        # Calculate number of pages needed
        pages_per_seq = math.ceil(max_seqlen / paged_size)
        total_pages = batch_size * pages_per_seq

        # Create device-resident tensors for the cache
        device = next(self.parameters()).device
        k_cache = torch.zeros(total_pages, paged_size, self.num_heads_kv, self.head_dim, dtype=dtype, device=device)
        v_cache = torch.zeros_like(k_cache)

        # Create block_table mapping (batch_idx, logical_block_idx) → physical_block_idx
        block_table = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(batch_size, pages_per_seq)
        return (k_cache, v_cache, block_table)

    def forward(
        self,
        x: torch.Tensor,
        inference_params: InferenceParams,
        freqs_cis: torch.Tensor,
        layer_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), inference_params, freqs_cis, layer_cache)
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        self.layer_idx = layer_idx

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        inference_params: InferenceParams,
        freqs_cis: torch.Tensor,
        layer_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seqlen, _ = x.shape

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        q, k, v = self.in_proj(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(batch_size, seqlen, self.num_heads, self.head_dim)
        k = k.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)
        v = v.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        # Unpack cache components directly from the passed tuple
        k_cache, v_cache, block_table = layer_cache

        # Use flash_attn_with_kvcache for both prefill and decode
        # It updates cache in-place and performs attention in one kernel
        y = flash_attn_with_kvcache(
            q=q,  # [B, S, H, D]
            k_cache=k_cache,  # [num_blocks, block_size, H_kv, D]
            v_cache=v_cache,  # [num_blocks, block_size, H_kv, D]
            k=k,  # [B, S, H_kv, D] - keys to add
            v=v,  # [B, S, H_kv, D] - values to add
            cache_seqlens=inference_params.lengths_per_sample,  # [B]
            block_table=block_table,  # [B, num_blocks_per_seq]
            causal=seqlen > 1,  # Use causal mask for prefill (multi-token)
        )

        # Reshape and project output
        y = y.reshape(batch_size, seqlen, q_size)
        y = self.out_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))
