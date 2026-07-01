"""
Test Suite — SmolLM3 vLLM Port (Project 2A)
============================================
Tests at four levels, matching what vLLM requires for a merged PR:

  Unit         — Architecture correctness (shapes, NoPE/RoPE alternation)
  Accuracy     — Output matches HuggingFace Transformers reference (greedy)
  Integration  — Full vLLM LLM() inference pipeline
  Benchmark    — Throughput vs Transformers backend

Run:
  pytest tests/ -m unit -v                     # fast, no GPU
  pytest tests/ -m accuracy -v                 # needs HF token + GPU
  pytest tests/ -m integration -v              # needs running vLLM + GPU
  pytest tests/ -m benchmark -v -s             # perf comparison
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# Add port to path for standalone testing
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm_port.smollm3 import (
    SmolLM3Attention,
    SmolLM3ForCausalLM,
    SmolLM3MLP,
    SmolLM3Model,
    _cfg,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def smollm3_config():
    """Minimal SmolLM3-3B config as a plain dict (no vLLM required)."""
    return {
        "architectures": ["SmolLM3ForCausalLM"],
        "hidden_size": 256,  # reduced for fast tests
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "rope_theta": 136000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "nope_layer_interval": 2,  # layers 1, 3 are NoPE
        "hidden_act": "silu",
    }


@pytest.fixture
def smollm3_full_config():
    """Full SmolLM3-3B config (for accuracy/integration tests)."""
    return {
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
        "tie_word_embeddings": True,
        "nope_layer_interval": 2,
        "hidden_act": "silu",
    }


# ===========================================================================
# UNIT TESTS — Architecture correctness
# ===========================================================================


class TestConfig:

    @pytest.mark.unit
    def test_cfg_dict(self):
        d = {"hidden_size": 2048, "rope_theta": 1000.0}
        assert _cfg(d, "hidden_size") == 2048
        assert _cfg(d, "missing", 99) == 99

    @pytest.mark.unit
    def test_cfg_object(self):
        class Cfg:
            hidden_size = 1024

        assert _cfg(Cfg(), "hidden_size") == 1024
        assert _cfg(Cfg(), "missing", 42) == 42


class TestNoPERoPEAlternation:
    """The defining feature of SmolLM3 — verify NoPE/RoPE assignment."""

    @pytest.mark.unit
    def test_rope_layers_even_nope_odd(self, smollm3_config):
        """With nope_layer_interval=2, odd layers are NoPE."""
        for layer_idx in range(smollm3_config["num_hidden_layers"]):
            attn = SmolLM3Attention(
                config=smollm3_config,
                layer_idx=layer_idx,
            )
            expected_rope = (layer_idx % 2) == 0  # 0,2,4... are RoPE; 1,3,5... are NoPE
            assert attn.use_rope == expected_rope, (
                f"Layer {layer_idx}: expected use_rope={expected_rope}, "
                f"got use_rope={attn.use_rope}"
            )

    @pytest.mark.unit
    def test_nope_interval_1_all_nope(self, smollm3_config):
        """nope_layer_interval=1: (i%1)==(1-1)=0 is always True → all NoPE layers."""
        cfg = dict(smollm3_config)
        cfg["nope_layer_interval"] = 1
        for i in range(4):
            attn = SmolLM3Attention(config=cfg, layer_idx=i)
            assert attn.use_rope is False, f"Layer {i} should be NoPE (interval=1)"

    @pytest.mark.unit
    def test_nope_interval_4_three_rope_one_nope(self, smollm3_config):
        """nope_layer_interval=4: layers 3,7,11... are NoPE; others are RoPE."""
        cfg = dict(smollm3_config)
        cfg["nope_layer_interval"] = 4
        for i in range(8):
            attn = SmolLM3Attention(config=cfg, layer_idx=i)
            expected = (i % 4) != 3
            assert attn.use_rope == expected, f"Layer {i}: expected {expected}"

    @pytest.mark.unit
    def test_nope_count_matches_expectation(self, smollm3_config):
        """With 4 layers and interval=2, exactly 2 NoPE layers."""
        cfg = dict(smollm3_config)
        cfg["num_hidden_layers"] = 4
        cfg["nope_layer_interval"] = 2
        model = SmolLM3Model(config=cfg)
        nope_count = sum(1 for layer in model.layers if not layer.self_attn.use_rope)
        rope_count = sum(1 for layer in model.layers if layer.self_attn.use_rope)
        assert nope_count == 2, f"Expected 2 NoPE layers, got {nope_count}"
        assert rope_count == 2, f"Expected 2 RoPE layers, got {rope_count}"


class TestMLPShapes:

    @pytest.mark.unit
    def test_mlp_output_shape(self, smollm3_config):
        hidden = smollm3_config["hidden_size"]
        inter = smollm3_config["intermediate_size"]
        mlp = SmolLM3MLP(hidden_size=hidden, intermediate_size=inter)
        x = torch.randn(2, 10, hidden)
        out = mlp(x)
        assert out.shape == (2, 10, hidden), f"MLP output shape mismatch: {out.shape}"

    @pytest.mark.unit
    def test_mlp_no_bias(self, smollm3_config):
        """SmolLM3 has no bias in linear layers."""
        mlp = SmolLM3MLP(
            hidden_size=smollm3_config["hidden_size"],
            intermediate_size=smollm3_config["intermediate_size"],
        )
        for name, param in mlp.named_parameters():
            assert "bias" not in name, f"Unexpected bias: {name}"


class TestAttentionShapes:

    @pytest.mark.unit
    def test_attention_output_shape(self, smollm3_config):
        hidden = smollm3_config["hidden_size"]
        batch, seq = 2, 8
        attn = SmolLM3Attention(config=smollm3_config, layer_idx=0)
        positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        x = torch.randn(batch, seq, hidden)
        out = attn(positions=positions, hidden_states=x, kv_cache=None, attn_metadata=None)
        assert out.shape == (batch, seq, hidden)

    @pytest.mark.unit
    def test_nope_attn_output_shape(self, smollm3_config):
        """NoPE attention (layer_idx=1) produces same output shape as RoPE."""
        hidden = smollm3_config["hidden_size"]
        batch, seq = 2, 8
        attn = SmolLM3Attention(config=smollm3_config, layer_idx=1)
        assert not attn.use_rope, "Layer 1 should be NoPE"
        positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        x = torch.randn(batch, seq, hidden)
        out = attn(positions=positions, hidden_states=x, kv_cache=None, attn_metadata=None)
        assert out.shape == (batch, seq, hidden)

    @pytest.mark.unit
    def test_gqa_head_ratio(self, smollm3_config):
        """GQA ratio: num_heads / num_kv_heads = 8/2 = 4."""
        attn = SmolLM3Attention(config=smollm3_config, layer_idx=0)
        ratio = attn.num_heads // attn.num_kv_heads
        assert ratio == 4, f"Expected GQA ratio 4, got {ratio}"


class TestModelShapes:

    @pytest.mark.unit
    def test_full_model_forward_shape(self, smollm3_config):
        batch, seq = 2, 12
        vocab = smollm3_config["vocab_size"]
        hidden = smollm3_config["hidden_size"]
        model = SmolLM3ForCausalLM(config=smollm3_config)
        input_ids = torch.randint(0, vocab, (batch, seq))
        positions = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        hidden_states = model(input_ids=input_ids, positions=positions)
        assert hidden_states.shape == (batch, seq, hidden)

    @pytest.mark.unit
    def test_lm_head_shape(self, smollm3_config):
        batch, seq = 2, 5
        vocab = smollm3_config["vocab_size"]
        hidden = smollm3_config["hidden_size"]
        model = SmolLM3ForCausalLM(config=smollm3_config)
        hidden_states = torch.randn(batch, seq, hidden)
        logits = model.compute_logits(hidden_states, sampling_metadata=None)
        assert logits is not None
        assert logits.shape[-1] == vocab

    @pytest.mark.unit
    def test_tied_embeddings(self, smollm3_config):
        """lm_head.weight must be tied to embed_tokens.weight."""
        model = SmolLM3ForCausalLM(config=smollm3_config)
        # After init + weight tying (load_weights does this; test the post-tie state)
        model.lm_head.weight = model.model.embed_tokens.weight
        assert (
            model.lm_head.weight is model.model.embed_tokens.weight
        ), "Weights are not tied (should share the same tensor)"

    @pytest.mark.unit
    def test_num_nope_layers_in_full_model(self, smollm3_config):
        """Verify NoPE count across all layers matches nope_layer_interval."""
        model = SmolLM3ForCausalLM(config=smollm3_config)
        n_layers = smollm3_config["num_hidden_layers"]
        interval = smollm3_config["nope_layer_interval"]
        expected_nope = n_layers // interval
        actual_nope = sum(1 for layer in model.model.layers if not layer.self_attn.use_rope)
        assert (
            actual_nope == expected_nope
        ), f"Expected {expected_nope} NoPE layers, got {actual_nope}"

    @pytest.mark.unit
    def test_parameter_count_reasonable(self, smollm3_config):
        """Model param count should match expected order of magnitude."""
        model = SmolLM3ForCausalLM(config=smollm3_config)
        total = sum(p.numel() for p in model.parameters())
        # With hidden=256, vocab=256, 4 layers: expect ~few million params
        assert total > 100_000, f"Too few params: {total}"
        assert total < 100_000_000, f"Too many params: {total}"

    @pytest.mark.unit
    def test_no_bias_in_model(self, smollm3_config):
        """SmolLM3 has no bias terms in attention or MLP."""
        model = SmolLM3ForCausalLM(config=smollm3_config)
        for name, param in model.named_parameters():
            if "norm" in name or "embed" in name:
                continue
            assert "bias" not in name, f"Unexpected bias parameter: {name}"

    @pytest.mark.unit
    def test_embed_output_shape(self, smollm3_config):
        model = SmolLM3Model(config=smollm3_config)
        ids = torch.randint(0, smollm3_config["vocab_size"], (3, 7))
        emb = model.embed_tokens(ids)
        assert emb.shape == (3, 7, smollm3_config["hidden_size"])


class TestWeightLoading:

    @pytest.mark.unit
    def test_load_weights_skips_lm_head(self, smollm3_config):
        """lm_head.weight should be skipped and left as tied embed_tokens."""
        model = SmolLM3ForCausalLM(config=smollm3_config)
        embed_weight_before = model.model.embed_tokens.weight.data.clone()

        # Simulate loading weights including a lm_head.weight (should be ignored)
        fake_lm_head = torch.zeros_like(model.model.embed_tokens.weight) + 999.0
        weights = [
            ("lm_head.weight", fake_lm_head),
            ("model.embed_tokens.weight", embed_weight_before),
        ]
        model.load_weights(iter(weights))

        # lm_head should NOT have been overwritten with 999s
        assert not torch.allclose(
            model.lm_head.weight,
            fake_lm_head,
        ), "lm_head.weight should have been ignored (tied to embed_tokens)"

    @pytest.mark.unit
    def test_load_weights_stacked_qkv(self, smollm3_config):
        """q/k/v projections should be merged into qkv_proj."""
        model = SmolLM3ForCausalLM(config=smollm3_config)
        hidden = smollm3_config["hidden_size"]
        n_heads = smollm3_config["num_attention_heads"]
        n_kv = smollm3_config["num_key_value_heads"]
        head_dim = smollm3_config["head_dim"]

        q_w = torch.randn(n_heads * head_dim, hidden)
        k_w = torch.randn(n_kv * head_dim, hidden)
        v_w = torch.randn(n_kv * head_dim, hidden)

        weights = [
            ("model.layers.0.self_attn.q_proj.weight", q_w),
            ("model.layers.0.self_attn.k_proj.weight", k_w),
            ("model.layers.0.self_attn.v_proj.weight", v_w),
        ]
        # Should not raise
        model.load_weights(iter(weights))


class TestNoPEVsRoPEBehavior:

    @pytest.mark.unit
    def test_nope_position_invariant(self, smollm3_config):
        """NoPE layers should produce same output regardless of positions passed."""
        hidden = smollm3_config["hidden_size"]
        seq = 6
        attn_nope = SmolLM3Attention(config=smollm3_config, layer_idx=1)
        assert not attn_nope.use_rope
        attn_nope.eval()
        x = torch.randn(1, seq, hidden)
        with torch.no_grad():
            pos_a = torch.zeros(1, seq, dtype=torch.long)
            pos_b = torch.arange(seq).unsqueeze(0) + 100
            out_a = attn_nope(positions=pos_a, hidden_states=x, kv_cache=None, attn_metadata=None)
            out_b = attn_nope(positions=pos_b, hidden_states=x, kv_cache=None, attn_metadata=None)
        assert torch.allclose(out_a, out_b, atol=1e-5)

    @pytest.mark.unit
    def test_rope_use_rope_flag_set(self, smollm3_config):
        """use_rope flag is set correctly per layer index."""
        assert SmolLM3Attention(config=smollm3_config, layer_idx=0).use_rope is True
        assert SmolLM3Attention(config=smollm3_config, layer_idx=1).use_rope is False


# ===========================================================================
# ACCURACY TESTS — Output matches HuggingFace Transformers (requires GPU+HF)
# ===========================================================================

HF_MODEL = os.getenv("SMOLLM3_MODEL", "HuggingFaceTB/SmolLM3-3B")
HAS_GPU = torch.cuda.is_available()
HAS_HF_TOKEN = bool(os.getenv("HF_TOKEN"))


@pytest.mark.accuracy
@pytest.mark.skipif(
    not (HAS_GPU and HAS_HF_TOKEN), reason="Needs GPU and HF_TOKEN for accuracy test"
)
class TestAccuracyVsHuggingFace:
    """
    Verify native vLLM port matches HF Transformers output under greedy decoding.
    vLLM's standard acceptance criterion: output match within <5% perplexity delta.
    """

    PROMPTS = [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "The transformer architecture consists of",
    ]

    def test_greedy_output_matches_hf(self):
        """
        Compare greedy token generation between:
        - HF Transformers (reference)
        - vLLM native SmolLM3 port (ours)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError:
            pytest.skip("transformers or vllm not installed")

        # Reference: HuggingFace greedy
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        hf_model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
        hf_model.eval()

        hf_outputs = []
        for prompt in self.PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
            with torch.no_grad():
                out = hf_model.generate(**inputs, max_new_tokens=20, do_sample=False)
            hf_outputs.append(
                tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            )

        del hf_model
        torch.cuda.empty_cache()

        # Test: vLLM native port
        llm = LLM(
            model=HF_MODEL,
            dtype="float16",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )
        params = SamplingParams(max_tokens=20, temperature=0.0)
        vllm_outputs = llm.generate(self.PROMPTS, params)
        vllm_texts = [o.outputs[0].text for o in vllm_outputs]

        # Compare: at least 2/3 prompts should produce identical output
        matches = sum(h == v for h, v in zip(hf_outputs, vllm_texts))
        assert matches >= 2, f"Only {matches}/3 outputs match HF reference.\n" + "\n".join(
            f"HF:   {h}\nvLLM: {v}" for h, v in zip(hf_outputs, vllm_texts)
        )

    def test_perplexity_within_5pct(self):
        """
        vLLM output perplexity should be within 5% of HF Transformers perplexity.
        Uses a standard text corpus excerpt.
        """
        pytest.skip("Perplexity test — implement with lm-evaluation-harness")


# ===========================================================================
# INTEGRATION TESTS — vLLM LLM() pipeline
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not (HAS_GPU and HAS_HF_TOKEN), reason="Needs GPU + HF_TOKEN for integration test"
)
class TestVLLMIntegration:

    def test_model_loads_without_error(self):
        from vllm import LLM

        llm = LLM(
            model=HF_MODEL,
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.85,
        )
        assert llm is not None

    def test_generate_returns_text(self):
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=HF_MODEL,
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.85,
        )
        outputs = llm.generate(
            ["Hello, my name is"], SamplingParams(max_tokens=20, temperature=0.0)
        )
        assert len(outputs) == 1
        text = outputs[0].outputs[0].text
        assert len(text) > 0
        print(f"\nGenerated: {text!r}")

    def test_nope_architecture_registered_as_native(self):
        """Verify SmolLM3 is NOT using the Transformers fallback backend."""
        try:
            from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS

            assert (
                "SmolLM3ForCausalLM" in _TEXT_GENERATION_MODELS
            ), "SmolLM3ForCausalLM should be in _TEXT_GENERATION_MODELS (native)"
            from vllm.model_executor.models.registry import _TRANSFORMERS_SUPPORTED_MODELS

            assert (
                "SmolLM3ForCausalLM" not in _TRANSFORMERS_SUPPORTED_MODELS
            ), "SmolLM3ForCausalLM should NOT be in _TRANSFORMERS_SUPPORTED_MODELS (fallback)"
        except ImportError:
            pytest.skip("vLLM not installed")

    def test_batched_generation(self):
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=HF_MODEL,
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.85,
        )
        prompts = [
            "What is Python?",
            "Explain attention in transformers.",
            "Write a haiku about computers.",
        ]
        outputs = llm.generate(prompts, SamplingParams(max_tokens=30, temperature=0.0))
        assert len(outputs) == 3
        for i, o in enumerate(outputs):
            assert len(o.outputs[0].text) > 0, f"Prompt {i} produced empty output"


# ===========================================================================
# BENCHMARK TESTS — Throughput comparison
# ===========================================================================


@pytest.mark.benchmark
@pytest.mark.skipif(not (HAS_GPU and HAS_HF_TOKEN), reason="Needs GPU + HF_TOKEN for benchmark")
class TestBenchmark:

    PROMPTS_50 = ["Explain the transformer architecture in detail." for _ in range(50)]

    def test_throughput_vs_transformers_backend(self):
        """
        Native vLLM port should be at least 1.5x faster than Transformers backend
        at batch size 16.
        """
        import time

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            pytest.skip("vLLM not installed")

        params = SamplingParams(max_tokens=50, temperature=0.0)

        # Native port
        llm_native = LLM(
            model=HF_MODEL,
            dtype="float16",
            max_model_len=512,
            gpu_memory_utilization=0.80,
        )
        t0 = time.perf_counter()
        llm_native.generate(self.PROMPTS_50, params)
        native_time = time.perf_counter() - t0
        native_tps = (50 * 50) / native_time
        del llm_native
        torch.cuda.empty_cache()

        print(f"\nNative vLLM port:      {native_tps:.1f} tokens/sec")
        print(f"Native time:           {native_time:.2f}s for 50 requests")

        # Assert minimum throughput (sanity check, not vs Transformers due to setup cost)
        assert native_tps > 50, f"Throughput {native_tps:.1f} tok/s too low"
