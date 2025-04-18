# Based on gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/095b2229ee3a40e379c11f05b94bd6923db63b4b/model.py
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
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


def _update_kv_cache(
    k: torch.Tensor, v: torch.Tensor, inference_params: InferenceParams, layer_idx: int
) -> torch.Tensor:
    """k/v: (batch_size, seqlen, nheads, head_dim) or (batch_size, 1, nheads, head_dim)"""
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    start = inference_params.lengths_per_sample  # [B]
    seq_idx = start.unsqueeze(1) + torch.arange(k.shape[1], device=k.device, dtype=torch.long).unsqueeze(0)  # [B, S]
    batch_idx = torch.arange(kv_cache.size(0), device=k.device, dtype=torch.long).unsqueeze(1)  # [B, 1]
    # Store k and v as separate heads in dimension 2: shape becomes (B, S, 2, heads_kv, head_dim)
    kv_cache[batch_idx, seq_idx] = torch.stack([k, v], dim=2)
    return kv_cache


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
        # TODO: This function should be pure
        head_dim = self.config.d_model // self.config.attn_cfg["num_heads"]
        self.freqs_cis = precompute_freqs_cis(16384, head_dim)
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        input_pos = torch.arange(S, device=hidden_states.device)
        input_pos = input_pos + inference_params.lengths_per_sample.unsqueeze(-1)
        freqs_cis = self.freqs_cis[input_pos]

        # Build cumulative sequence lengths for varlen API, enforce int32 dtype
        cu_seqlens_q = torch.arange(0, B * S + 1, S, dtype=torch.int32, device=hidden_states.device)

        if S > 1:
            cu_seqlens_k = torch.arange(0, B * S + 1, S, dtype=torch.int32, device=hidden_states.device)
        else:
            # Single-token query: include current token in KV lengths for self-attention
            kv_lens = inference_params.lengths_per_sample + S
            zero = torch.tensor([0], dtype=torch.int32, device=hidden_states.device)
            cu_seqlens_k = torch.cat([zero, torch.cumsum(kv_lens, dim=0)]).to(torch.int32)

        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params, freqs_cis, cu_seqlens_q, cu_seqlens_k)
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
        # setting it to zeros rather than empty to avoid nan explosions in scaled_dot_product_attention
        # in the masked out regions
        # Shape: (B, S, 2, heads_kv, head_dim) for explicit k/v packing
        return torch.zeros(batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim, dtype=dtype), None

    def forward(
        self,
        x: torch.Tensor,
        inference_params: InferenceParams,
        freqs_cis: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), inference_params, freqs_cis, cu_seqlens_q, cu_seqlens_k)
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
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
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

        kv = _update_kv_cache(k, v, inference_params, self.layer_idx)
        # Pack Q and KV for FlashAttention varlen_kvpacked API (expects (total_k,2,heads_kv,head_dim))
        y_flat = flash_attn_varlen_kvpacked_func(
            q.view(-1, self.num_heads, self.head_dim),
            kv.view(-1, 2, self.num_heads_kv, self.head_dim),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=seqlen,
            max_seqlen_k=inference_params.max_seqlen,
            causal=seqlen > 1,
        )
        # Reshape back to batched output
        y = y_flat.view(batch_size, seqlen, self.num_heads, self.head_dim)
        y = y.reshape(batch_size, seqlen, q_size)  # merge heads and head_dim

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
