"""Dataset I/O and loading tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from predictor.dataset import OutputLengthDataset, _estimate_tokens, _hash_prompt


@pytest.mark.unit
def test_hash_and_estimate_tokens():
    h = _hash_prompt("hello")
    assert len(h) == 64
    assert _estimate_tokens("one two three") >= 1


@pytest.mark.unit
def test_from_prompts():
    ds = OutputLengthDataset(cache_dir=Path("/tmp/unused"))
    samples = ds.from_prompts(["hi"], ["hello world"], [{"src": "t"}])
    assert len(samples) == 1
    assert samples[0].actual_output_tokens > 0


@pytest.mark.unit
def test_from_jsonl_and_csv(tmp_path):
    ds = OutputLengthDataset(cache_dir=tmp_path / "cache")
    jl = tmp_path / "data.jsonl"
    jl.write_text('{"prompt": "p", "output": "o"}\n')
    assert len(ds.from_jsonl(jl)) == 1

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("prompt,output\nhello,world\n")
    assert len(ds.from_csv(csv_path)) == 1


@pytest.mark.unit
def test_load_preset_synthetic():
    ds = OutputLengthDataset(cache_dir=Path("/tmp/cache_ds"))
    samples = ds.load_preset("synthetic", max_samples=10)
    assert len(samples) == 10


@pytest.mark.unit
def test_load_preset_unknown():
    ds = OutputLengthDataset()
    with pytest.raises(ValueError, match="Unknown preset"):
        ds.load_preset("not-a-preset")


@pytest.mark.unit
def test_from_huggingface_import_error():
    ds = OutputLengthDataset()
    import sys

    mods = sys.modules.copy()
    try:
        sys.modules["datasets"] = None  # type: ignore[assignment]
        with pytest.raises(ImportError):
            ds.from_huggingface("x")
    finally:
        for k in list(sys.modules):
            if k not in mods:
                del sys.modules[k]
