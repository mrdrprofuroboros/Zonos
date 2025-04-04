import itertools
import json
from typing import Callable, Generator

import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner, make_cond_dict
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))
UNKNOWN_TOKEN = -1


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls,
        config_path: str,
        model_path: str,
        device: str = DEFAULT_DEVICE,
        backbone: str | None = None,
        speaker_models_path: str | None = None,
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]

        model = cls(config, backbone_cls).to(device, torch.bfloat16)
        model.autoencoder.dac.to(device)

        sd = model.state_dict()
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        model.load_state_dict(sd)

        if speaker_models_path:
            model.spk_clone_model = SpeakerEmbeddingLDA(device, speaker_models_path)

        return model

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        return spk_embedding.unsqueeze(0).bfloat16()

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits[..., 1025:].fill_(-torch.inf)  # ensures padding is ignored
        return logits

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        # TODO: support cfg_scale==1
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0)

        if not allow_cudagraphs or input_ids.device.type != "cuda":
            hidden_states_local = self.embed_codes(input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            self._cg_graph = None

            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            for _ in range(3):
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)  # because cfg != 1.0
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone()
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph()

            def capture_region():
                hidden_states_local = self.embed_codes(self._cg_input_ids)
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)

            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else:
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()

        return self._cg_logits

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled
        if cfg_scale != 1.0:
            input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        max_seqlen = find_multiple(max_seqlen, 8)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),
                self.prefix_conditioner(uncond_dict),
            ]
        )

    def can_use_cudagraphs(self) -> bool:
        # Only the mamba-ssm backbone supports CUDA Graphs at the moment
        return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [bsz, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [bsz, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(min_p=0.1),
        progress_bar: bool = True,
        disable_torch_compile: bool = False,
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
        device = self.device

        # Use CUDA Graphs if supported, and torch.compile otherwise.
        cg = self.can_use_cudagraphs()
        decode_one_token = self._decode_one_token
        decode_one_token = torch.compile(decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

        audio_seq_len = prefix_audio_len + max_new_tokens
        seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

        with torch.device(device):
            inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
            codes = torch.full((batch_size, 9, audio_seq_len), UNKNOWN_TOKEN)

        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes

        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
        next_token = sample_from_logits(logits, **sampling_params)

        offset = delayed_prefix_audio_codes.shape[2]
        frame = delayed_codes[..., offset : offset + 1]
        frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)

        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS
        # Make EOS less likely because audio often is cut off
        logit_bias[:, 0, self.eos_token_id] -= torch.log(torch.tensor(2.0, device=logits.device))

        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=device)
        progress = tqdm(total=max_steps, desc="Generating", disable=not progress_bar)

        step = 0
        while torch.max(remaining_steps) > 0:
            offset += 1
            input_ids = delayed_codes[..., offset - 1 : offset]
            logits = decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
            logits += logit_bias

            next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)
            eos_in_cb0 = next_token[:, 0] == self.eos_token_id

            remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
            stopping |= eos_in_cb0[:, 0]

            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = eos_codebook_idx[i].item()
                    next_token[i, :idx] = self.masked_token_id
                    next_token[i, idx] = self.eos_token_id

            frame = delayed_codes[..., offset : offset + 1]
            frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1

            remaining_steps -= 1

            progress.update()
            step += 1

            if callback is not None and not callback(frame, step, max_steps):
                break

        out_codes = revert_delay_pattern(delayed_codes)
        out_codes.masked_fill_(out_codes >= 1024, 0)
        out_codes = out_codes[..., : offset - 9]

        self._cg_graph = None  # reset cuda graph to avoid cache changes

        return out_codes

    @torch.inference_mode()
    def stream(
        self,
        cond_dicts_generator: Generator[dict, None, None],
        audio_prefix_codes: torch.Tensor | None = None,
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        sampling_params: dict = dict(min_p=0.1),
        disable_torch_compile: bool = False,
        chunk_schedule: list[int] = [16, *range(9, 100)],
        chunk_overlap: int = 2,
        whitespace: str = " ",
        warmup_prefill: str = "",
        mark_boundaries: bool = False,
    ) -> Generator[torch.Tensor | str, None, None]:
        """
        Stream audio generation in chunks with smooth transitions between chunks.

        Args:
            cond_dicts_generator: Generator of conditioning dictionaries
            audio_prefix_codes: Optional audio prefix codes
            max_new_tokens: Maximum number of new tokens to generate
            cfg_scale: Classifier-free guidance scale
            sampling_params: Parameters for sampling from logits
            disable_torch_compile: Whether to disable torch.compile
            chunk_schedule: List of chunk sizes to use in sequence (will use the last size for remaining chunks)
            chunk_overlap: Number of tokens to overlap between chunks (also determines audio crossfade size)
            whitespace: Whitespace to use between sentences
            warmup_prefill: A warmup string to generate before the first generator chunk
            mark_boundaries: Whether to yield sentence strings as indicators of the sentence end

        Yields:
            Audio chunks as torch tensors [and sentence strings as indicators of the sentence end if mark_boundaries is True]
        """
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"
        assert len(chunk_schedule) > 0, "chunk_schedule must not be empty"
        assert all(chunk_overlap * 2 < size for size in chunk_schedule), "overlap must be less than a half of a chunk"

        batch_size = 1  # Streaming generation is single-sample only
        device = self.device

        # Use CUDA Graphs if supported, and torch.compile otherwise.
        cg = self.can_use_cudagraphs()
        decode_one_token = self._decode_one_token
        decode_one_token = torch.compile(decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

        # Calculate window size based on overlap tokens (approx. samples per token)
        samples_per_token = 512  # Approximate value based on DAC model
        window_size = chunk_overlap * samples_per_token

        # Create cosine fade for smooth transition
        cosfade = torch.cos(torch.linspace(torch.pi, 0, window_size, device=device))
        cosfade = 0.5 * (1 + cosfade)

        # Set up first text and codes to use as a prefix for all generations
        audio_prefix_text = ""
        previous_audio = None
        generator_index = 0
        first_chunk_yielded = False

        # A hack to warm up the model - we can start the stream beforehand and generate a few tokens upfront
        # so when we actually receive the first sentence, we can start streaming faster
        if warmup_prefill:
            cond_dicts_generator = itertools.chain([{"text": warmup_prefill}], cond_dicts_generator)
            generator_index = -1  # set this to -1 to skip that warmup chunk and never yield it

        # Main loop: iterate over sentences in the cond_dicts_generator. For each sentence, we'll be streaming audio chunks out.
        # Once the first sentence is ready, we'll use it's codes as audio_prefix_codes for all the next sentences
        for cond_dict in cond_dicts_generator:
            # Prepend the conditioning dictionary text with the previous sentence text
            curr_text = cond_dict["text"]
            updated_cond_dict = {**cond_dict, "text": audio_prefix_text + curr_text + whitespace}

            prefix_conditioning = self.prepare_conditioning(make_cond_dict(**updated_cond_dict))

            prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]
            audio_seq_len = prefix_audio_len + max_new_tokens
            seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

            with torch.device(device):
                inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
                codes = torch.full((batch_size, 9, audio_seq_len), UNKNOWN_TOKEN)

            if audio_prefix_codes is not None:
                codes[..., :prefix_audio_len] = audio_prefix_codes

            delayed_codes = apply_delay_pattern(codes, self.masked_token_id)
            delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

            logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)
            next_token = sample_from_logits(logits, **sampling_params)

            offset = delayed_prefix_audio_codes.shape[2]
            frame = delayed_codes[..., offset : offset + 1]
            frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)

            prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
            inference_params.seqlen_offset += prefix_length
            inference_params.lengths_per_sample[:] += prefix_length

            logit_bias = torch.zeros_like(logits)
            logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS
            # Make EOS less likely because audio often is cut off
            logit_bias[:, 0, self.eos_token_id] -= torch.log(torch.tensor(2.0, device=logits.device))

            # --- Autoregressive loop ---
            stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
            max_steps = delayed_codes.shape[2] - offset
            remaining_steps = torch.full((batch_size,), max_steps, device=device)
            step = 0
            # This variable will let us yield only the new audio since the last yield.
            prev_valid_length = prefix_audio_len
            chunk_counter = 0

            # For chunk scheduling
            schedule_index = 0

            while torch.max(remaining_steps) > 0:
                offset += 1
                input_ids = delayed_codes[..., offset - 1 : offset]
                logits = decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
                logits += logit_bias

                next_token = sample_from_logits(logits, generated_tokens=delayed_codes[..., :offset], **sampling_params)

                # Update stopping for finished samples.
                eos_in_cb0 = next_token[:, 0] == self.eos_token_id
                remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9))
                stopping |= eos_in_cb0[:, 0]

                eos_codebook_idx = 9 - remaining_steps
                eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 - 1)
                for i in range(next_token.shape[0]):
                    if stopping[i]:
                        idx = eos_codebook_idx[i].item()
                        next_token[i, :idx] = self.masked_token_id
                        next_token[i, idx] = self.eos_token_id

                frame = delayed_codes[..., offset : offset + 1]
                frame.masked_scatter_(frame == UNKNOWN_TOKEN, next_token)
                inference_params.seqlen_offset += 1
                inference_params.lengths_per_sample[:] += 1
                remaining_steps -= 1
                step += 1
                chunk_counter += 1

                # --- Every 'chunk_size' tokens (or when finished), decode and yield the new audio ---
                if (chunk_counter >= chunk_schedule[schedule_index]) or (torch.all(remaining_steps == 0)):
                    # In Zonos, the final output codes are produced by reverting the delay pattern.
                    # Only tokens up to (offset - 9) are valid.
                    full_codes = revert_delay_pattern(delayed_codes)
                    full_codes.masked_fill_(full_codes >= 1024, 0)

                    # Get the valid portion of the latent sequence.
                    valid_length = offset - 9

                    # Include overlap with previous chunk for smoother transitions
                    # For the first chunk, there's no previous chunk to overlap with
                    start_idx = max(0, prev_valid_length - chunk_overlap)
                    partial_codes = full_codes[..., start_idx:valid_length]

                    # Decode the current chunk to audio (keep on device)
                    current_audio = self.autoencoder.decode(partial_codes)[0]

                    # Apply fade-in to the first chunk
                    if previous_audio is None and current_audio.shape[-1] > window_size:
                        current_audio[..., :window_size] *= cosfade

                    # Apply windowing and overlap-add for smooth transitions
                    if (
                        previous_audio is not None
                        and current_audio.shape[-1] > window_size
                        and previous_audio.shape[-1] > window_size
                    ):
                        # Apply windowing to overlapping regions
                        curr_audio_start = current_audio[..., :window_size].clone()
                        prev_audio_end = previous_audio[..., -window_size:].clone()

                        # Crossfade the overlapping region
                        previous_audio[..., -window_size:] = curr_audio_start * cosfade + prev_audio_end * (1 - cosfade)

                    if previous_audio is not None and generator_index >= 0:
                        # Apply a log fade in to the first chunk to avoid a pop
                        if not first_chunk_yielded:
                            if previous_audio.shape[-1] > 2 * window_size:
                                logfade = torch.logspace(1, 0, 2 * window_size, base=20, device=device)
                                logfade -= logfade.min()
                                logfade /= logfade.max()
                                previous_audio[..., : 2 * window_size] *= logfade.flip(0) 
                            first_chunk_yielded = True

                        yield previous_audio

                    # Store current audio for next iteration and update counters
                    previous_audio = current_audio[..., window_size:]
                    prev_valid_length = valid_length
                    chunk_counter = 0

                    # Update chunk size according to schedule
                    if schedule_index < len(chunk_schedule) - 1:
                        schedule_index += 1

            if generator_index == 0:
                # Assemble the full codes for this sentence
                audio_prefix_codes = revert_delay_pattern(delayed_codes)
                audio_prefix_codes.masked_fill_(audio_prefix_codes >= 1024, 0)
                audio_prefix_codes = audio_prefix_codes[..., : offset - 9]
                audio_prefix_text = curr_text + whitespace

            if previous_audio is not None:
                tail_size = min(2 * window_size, previous_audio.shape[-1])
                logfade = torch.logspace(1, 0, tail_size, base=20, device=device)
                logfade -= logfade.min()
                logfade /= logfade.max()
                previous_audio[..., -tail_size:] *= logfade

            self._cg_graph = None  # reset CUDA graph to avoid caching issues
            generator_index += 1

            # Yield the sentence string if mark_boundaries is True
            if mark_boundaries:
                yield curr_text

        # Don't forget to yield the final audio chunk
        if previous_audio is not None:
            yield previous_audio
