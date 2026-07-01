"""
vllm/model_executor/models/smollm3.py
======================================
Native vLLM port of SmolLM3 (HuggingFaceTB/SmolLM3-3B).

Key architectural delta vs Llama/Qwen2:
  - Alternating NoPE (No Positional Encoding) and RoPE layers.
    Every `nope_layer_interval`-th layer skips RoPE on Q and K.
    This enables long-context attention without position bias in those layers.
  - Tied input/output embeddings (lm_head shares embed_tokens weight).
  - GQA with num_kv_heads=8, total heads=32.
  - SwiGLU MLP, RMSNorm, standard decoder-only structure.

Add to vllm/model_executor/models/__init__.py:
    from .smollm3 import SmolLM3ForCausalLM

Add to vllm/model_executor/models/registry.py under _TEXT_GENERATION_MODELS:
    "SmolLM3ForCausalLM": ("smollm3", "SmolLM3ForCausalLM"),

Remove from _TRANSFORMERS_SUPPORTED_MODELS:
    "SmolLM3ForCausalLM": ("transformers", "TransformersForCausalLM"),
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Graceful imports — works standalone (tests) AND inside real vLLM
# ---------------------------------------------------------------------------
try:
    from vllm.attention import Attention, AttentionMetadata
    from vllm.config import CacheConfig, VllmConfig
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
        VocabParallelEmbedding,
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.sampling_metadata import SamplingMetadata
    from vllm.sequence import IntermediateTensors

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    # Provide stub types so the module is importable for testing/analysis
    Attention = AttentionMetadata = CacheConfig = VllmConfig = object
    SamplingMetadata = IntermediateTensors = SamplerOutput = object


# ---------------------------------------------------------------------------
# Config helper — works with real vLLM config or a plain dict/namespace
# ---------------------------------------------------------------------------


def _cfg(config, key: str, default=None):
    """Safely read from either a real HF config object or a dict."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# SmolLM3 MLP (identical to Llama/Qwen2 SwiGLU)
# ---------------------------------------------------------------------------


class SmolLM3MLP(nn.Module):
    """
    SwiGLU feed-forward network.
    Output = down_proj(silu(gate_proj(x)) * up_proj(x))
    gate_proj and up_proj are merged into gate_up_proj for efficiency.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if _VLLM_AVAILABLE:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.down_proj",
            )
            self.act_fn = SiluAndMul()
        else:
            # Standalone stub for tests
            self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _VLLM_AVAILABLE:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
        else:
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            x = self.act_fn(gate) * up
            x = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# SmolLM3 Attention — the KEY delta: NoPE vs RoPE per layer
# ---------------------------------------------------------------------------


class SmolLM3Attention(nn.Module):
    """
    Multi-head grouped-query attention with optional RoPE.

    The `use_rope` flag is set per layer:
      - use_rope=True  → standard RoPE applied to Q and K before attention
      - use_rope=False → NoPE layer: Q and K passed without positional rotation

    This alternation (every `nope_layer_interval` layers) is SmolLM3's
    defining architectural feature, enabling very long context without
    quadratic position bias accumulation in NoPE layers.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.hidden_size = _cfg(config, "hidden_size", 2048)
        self.num_heads = _cfg(config, "num_attention_heads", 32)
        self.num_kv_heads = _cfg(config, "num_key_value_heads", 8)
        self.head_dim = _cfg(config, "head_dim", self.hidden_size // self.num_heads)
        self.scaling = self.head_dim**-0.5

        # [CRITICAL PATCH] Determine if this layer uses RoPE or NoPE
        nope_interval = _cfg(config, "nope_layer_interval", 2)
        # NoPE on every nope_interval-th layer (0-indexed: layers 1, 3, 5, ...)
        self.use_rope = (layer_idx % nope_interval) != (nope_interval - 1)

        rope_theta = _cfg(config, "rope_theta", 136000.0)
        max_position = _cfg(config, "max_position_embeddings", 131072)

        if _VLLM_AVAILABLE:
            tp_size = get_tensor_model_parallel_world_size()
            self.total_num_heads = self.num_heads
            self.total_num_kv_heads = self.num_kv_heads
            self.num_heads = self.num_heads // tp_size
            self.num_kv_heads = max(1, self.num_kv_heads // tp_size)

            self.qkv_proj = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )

            # RoPE (used only when self.use_rope=True)
            if self.use_rope:
                self.rotary_emb = get_rope(
                    self.head_dim,
                    rotary_dim=self.head_dim,
                    max_position=max_position,
                    base=rope_theta,
                )

            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        else:
            # Standalone stubs
            total_head_dim = self.num_heads * self.head_dim
            kv_head_dim = self.num_kv_heads * self.head_dim
            self.q_proj = nn.Linear(self.hidden_size, total_head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, kv_head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, kv_head_dim, bias=False)
            self.o_proj = nn.Linear(total_head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        if _VLLM_AVAILABLE:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    self.num_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                ],
                dim=-1,
            )

            # [PATCH] Apply RoPE only in RoPE layers; NoPE layers skip rotation
            if self.use_rope:
                q, k = self.rotary_emb(positions, q, k)
            # else: q and k are used as-is (NoPE — no positional encoding)

            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
            output, _ = self.o_proj(attn_output)
            return output
        else:
            # Standalone forward (no KV cache, simplified)
            b, s, _ = hidden_states.shape
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            # Naive scaled dot-product (for testing shape correctness only)
            q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
            # Expand KV heads for GQA
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scaling, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, s, -1)
            return self.o_proj(out)


# ---------------------------------------------------------------------------
# SmolLM3 Decoder Layer
# ---------------------------------------------------------------------------


class SmolLM3DecoderLayer(nn.Module):
    """
    Standard pre-norm transformer block:
      x = x + Attention(RMSNorm(x))
      x = x + MLP(RMSNorm(x))
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = _cfg(config, "hidden_size", 2048)
        rms_norm_eps = _cfg(config, "rms_norm_eps", 1e-5)

        self.self_attn = SmolLM3Attention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = SmolLM3MLP(
            hidden_size=hidden_size,
            intermediate_size=_cfg(config, "intermediate_size", 8192),
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        if _VLLM_AVAILABLE:
            self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        else:
            self.input_layernorm = nn.LayerNorm(
                hidden_size, eps=rms_norm_eps, elementwise_affine=True
            )
            self.post_attention_layernorm = nn.LayerNorm(
                hidden_size, eps=rms_norm_eps, elementwise_affine=True
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm residual path
        # Real vLLM RMSNorm accepts (x, residual) and returns (normed, residual).
        # Standalone nn.LayerNorm only accepts (x); we handle both.
        if _VLLM_AVAILABLE:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            if residual is None:
                residual = hidden_states
            else:
                hidden_states = hidden_states + residual
                residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Post-attention norm + MLP
        if _VLLM_AVAILABLE:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# SmolLM3 Base Model
# ---------------------------------------------------------------------------


class SmolLM3Model(nn.Module):
    """
    SmolLM3 decoder stack.
    Manages the embedding, N decoder layers, and final RMSNorm.
    """

    def __init__(
        self,
        *,
        vllm_config=None,
        prefix: str = "",
        config=None,
        cache_config=None,
        quant_config=None,
    ) -> None:
        super().__init__()

        # Support both vllm_config style (real vLLM) and raw config (tests)
        if vllm_config is not None and _VLLM_AVAILABLE:
            config = vllm_config.model_config.hf_config
            cache_config = vllm_config.cache_config
            quant_config = vllm_config.quant_config

        self.config = config
        hidden_size = _cfg(config, "hidden_size", 2048)
        vocab_size = _cfg(config, "vocab_size", 49152)
        num_layers = _cfg(config, "num_hidden_layers", 32)
        rms_norm_eps = _cfg(config, "rms_norm_eps", 1e-5)

        if _VLLM_AVAILABLE:
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size, hidden_size, prefix=f"{prefix}.embed_tokens"
            )
        else:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        self.layers = nn.ModuleList(
            [
                SmolLM3DecoderLayer(
                    config=config,
                    layer_idx=i,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(num_layers)
            ]
        )

        if _VLLM_AVAILABLE:
            self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=rms_norm_eps, elementwise_affine=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata=None,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, "IntermediateTensors"]:

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches else None
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                residual=residual,
            )

        if _VLLM_AVAILABLE:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states + (residual if residual is not None else 0))
        return hidden_states


# ---------------------------------------------------------------------------
# SmolLM3ForCausalLM — top-level model registered in vLLM
# ---------------------------------------------------------------------------


class SmolLM3ForCausalLM(nn.Module):
    """
    SmolLM3 causal language model for vLLM native inference.

    Registration:
      vllm/model_executor/models/registry.py:
        _TEXT_GENERATION_MODELS["SmolLM3ForCausalLM"] = ("smollm3", "SmolLM3ForCausalLM")

    Replaces the Transformers fallback entry in _TRANSFORMERS_SUPPORTED_MODELS.
    """

    # Columns that are parallelised along the output/vocab dimension
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # Modules not to LoRA-wrap (embedding is tied with lm_head)
    embedding_modules = {"model.embed_tokens": "input_embeddings"}
    embedding_padding_modules = []

    def __init__(
        self,
        *,
        vllm_config=None,
        prefix: str = "",
        config=None,
        cache_config=None,
        quant_config=None,
    ) -> None:
        super().__init__()

        if vllm_config is not None and _VLLM_AVAILABLE:
            config = vllm_config.model_config.hf_config
            cache_config = vllm_config.cache_config
            quant_config = vllm_config.quant_config

        self.config = config
        self.tie_word_embeddings = _cfg(config, "tie_word_embeddings", True)
        self.model = SmolLM3Model(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.model",
        )

        vocab_size = _cfg(config, "vocab_size", 49152)
        hidden_size = _cfg(config, "hidden_size", 2048)

        # [PATCH] Tied embeddings: lm_head shares embed_tokens weights.
        # In vLLM, ParallelLMHead handles the tied case when unpadded_vocab_size
        # matches embed_tokens; weight sharing is set in load_weights.
        if _VLLM_AVAILABLE:
            self.lm_head = ParallelLMHead(
                vocab_size,
                hidden_size,
                org_num_embeddings=vocab_size,
                prefix=f"{prefix}.lm_head",
            )
            self.logits_processor = LogitsProcessor(vocab_size)
            self.sampler = get_sampler()
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata=None,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, "IntermediateTensors"]:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> Optional[torch.Tensor]:
        if _VLLM_AVAILABLE:
            logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
            return logits
        return self.lm_head(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> Optional["SamplerOutput"]:
        if _VLLM_AVAILABLE:
            return self.sampler(logits, sampling_metadata)
        return None

    # ------------------------------------------------------------------
    # Weight loading — the most important part of a model port
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load weights from HuggingFace checkpoint into vLLM's parallelised layout.

        Key challenges handled here:
        1. QKV merging: separate q/k/v → single QKVParallelLinear
        2. Gate/Up merging: separate gate/up → single MergedColumnParallelLinear
        3. Tied embeddings: lm_head.weight = model.embed_tokens.weight (skip lm_head)
        """
        # Columns that need special stacking
        stacked_params_mapping = [
            # (hf_name_suffix, vllm_param_name, shard_id)
            ("q_proj", "qkv_proj", "q"),
            ("k_proj", "qkv_proj", "k"),
            ("v_proj", "qkv_proj", "v"),
            ("gate_proj", "gate_up_proj", 0),
            ("up_proj", "gate_up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # [PATCH] Skip lm_head — it's tied to embed_tokens (only when configured as tied)
            if name == "lm_head.weight" and self.tie_word_embeddings:
                continue

            # Handle stacked projections
            is_stacked = False
            for hf_suffix, vllm_name, shard_id in stacked_params_mapping:
                if name.endswith(hf_suffix):
                    # Replace suffix: self_attn.q_proj → self_attn.qkv_proj
                    vllm_key = name.replace(hf_suffix, vllm_name)
                    if vllm_key not in params_dict:
                        break
                    param = params_dict[vllm_key]
                    if _VLLM_AVAILABLE:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                    is_stacked = True
                    break

            if is_stacked:
                continue

            # Standard weight loading
            if name in params_dict:
                param = params_dict[name]
                if _VLLM_AVAILABLE:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)

        # [PATCH] Tie lm_head weights to embed_tokens AFTER loading
        # This is required because embed_tokens was loaded, lm_head was skipped
        if (
            self.tie_word_embeddings
            and hasattr(self.lm_head, "weight")
            and hasattr(self.model.embed_tokens, "weight")
        ):
            self.lm_head.weight = self.model.embed_tokens.weight
