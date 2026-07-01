# SmolLM3 Architecture Analysis — Project 2A

## Why SmolLM3? The Model Selection Rationale

vLLM supports SmolLM3 today via `_TRANSFORMERS_SUPPORTED_MODELS` — a fallback
that runs the HuggingFace Transformers model class inside vLLM's engine. This works
but misses all of vLLM's optimised inference path:

| Feature | Transformers Backend | Native vLLM Port |
|---------|---------------------|-----------------|
| PagedAttention | ❌ | ✅ |
| Continuous batching | Partial | ✅ |
| Tensor parallel | ❌ | ✅ |
| QKV parallel linear | ❌ | ✅ |
| LoRA support | ❌ | ✅ |
| Quantization (AWQ/GPTQ) | ❌ | ✅ |

A native port removes all these gaps. SmolLM3-3B fits on a T1000 8GB in float16,
making it the perfect model for this project.

## SmolLM3 Architecture: What's New vs Llama

SmolLM3 (`HuggingFaceTB/SmolLM3-3B`) is a decoder-only transformer with these
key differences from standard Llama 2/3:

### 1. No RoPE (NoPE layers)

SmolLM3's headline feature: alternating **NoPE** (No Positional Encoding) and **RoPE**
layers. Every other attention layer has NO positional embedding applied to Q and K.
This is how it achieves "very large context lengths" — NoPE layers let the model attend
position-agnostically, while RoPE layers provide local position awareness.

```
Layer 0:  RoPE attention   ← positions matter
Layer 1:  NoPE attention   ← no positional encoding on Q/K
Layer 2:  RoPE attention
Layer 3:  NoPE attention
...
```

**Implementation delta:** The `get_rope()` call must be conditional per layer.
In NoPE layers, Q and K are passed to attention WITHOUT rotation.

### 2. Grouped Query Attention (GQA)

Standard GQA, same as Llama 3. For SmolLM3-3B:
- `num_attention_heads = 32`
- `num_key_value_heads = 8`  (4:1 MHA:KV ratio)
- `head_dim = hidden_size / num_attention_heads = 64`

### 3. SwiGLU MLP (same as Llama)

Standard SwiGLU with gate and up projections merged:
```python
gate_up_proj: MergedColumnParallelLinear  # [hidden, 2 * intermediate]
down_proj:    RowParallelLinear           # [intermediate, hidden]
```

### 4. RMSNorm (same as Llama)

Pre-norm with RMSNorm before both attention and MLP.

### 5. Vocabulary

`vocab_size = 49152` — same as SmolLM2 (Llama 3 tokenizer).

### 6. Tied Embeddings

Input embedding and LM head weights are **tied** (shared). This halves the vocabulary
parameter count and is common in small models. Must be handled carefully in tensor
parallel mode — the tied weight cannot be naively split along both vocab and hidden dims.

## Key Config Values (SmolLM3-3B)

```json
{
  "architectures": ["SmolLM3ForCausalLM"],
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 64,
  "vocab_size": 49152,
  "max_position_embeddings": 131072,
  "rope_theta": 136000.0,
  "rms_norm_eps": 1e-5,
  "tie_word_embeddings": true,
  "nope_layer_interval": 2,
  "hidden_act": "silu"
}
```

## Nearest Existing vLLM Model: Qwen2

`vllm/model_executor/models/qwen2.py` is the closest match:
- GQA ✅
- SwiGLU ✅
- RMSNorm ✅
- Tied embeddings ✅ (in Qwen2's smaller variants)

The only meaningful delta is the **NoPE/RoPE alternation**, which requires a
per-layer `use_rope: bool` flag.

## Weight Name Mapping (HF → vLLM)

| HuggingFace weight | vLLM convention |
|-------------------|-----------------|
| `model.embed_tokens.weight` | `model.embed_tokens.weight` |
| `model.layers.N.self_attn.q_proj.weight` | `model.layers.N.self_attn.qkv_proj.weight` (merged) |
| `model.layers.N.self_attn.k_proj.weight` | merged into qkv_proj |
| `model.layers.N.self_attn.v_proj.weight` | merged into qkv_proj |
| `model.layers.N.self_attn.o_proj.weight` | `model.layers.N.self_attn.o_proj.weight` |
| `model.layers.N.mlp.gate_proj.weight` | merged into gate_up_proj |
| `model.layers.N.mlp.up_proj.weight` | merged into gate_up_proj |
| `model.layers.N.mlp.down_proj.weight` | `model.layers.N.mlp.down_proj.weight` |
| `model.layers.N.input_layernorm.weight` | same |
| `model.layers.N.post_attention_layernorm.weight` | same |
| `model.norm.weight` | same |
| `lm_head.weight` | tied to embed_tokens (not loaded separately) |

## VRAM Estimate (T1000 8GB)

SmolLM3-3B in float16:
- Weights: ~3B × 2 bytes = ~6 GB
- KV cache (2048 ctx, 32 layers, 8 KV heads, head_dim=64): ~0.5 GB
- Runtime overhead: ~1 GB
- **Total: ~7.5 GB** — fits on T1000 8GB with `--gpu-memory-utilization 0.90`

Use `--max-model-len 2048` to keep KV cache small during dev/test.
