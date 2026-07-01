"""Tests for feature extraction."""

from __future__ import annotations

import time

import pytest
from predictor.feature_extractor import FeatureExtractor, extract_from_messages
from predictor.features import FEATURE_NAMES, NUM_FEATURES


@pytest.mark.unit
class TestFeatureExtractor:
    def test_feature_count(self) -> None:
        ext = FeatureExtractor().extract("Hello world")
        assert len(ext.features.values) == NUM_FEATURES

    def test_feature_names_match(self) -> None:
        assert len(FEATURE_NAMES) == NUM_FEATURES

    def test_empty_prompt(self) -> None:
        ext = FeatureExtractor().extract("")
        assert ext.features.values[0] >= 0

    def test_code_detection(self) -> None:
        prompt = "```python\ndef foo():\n    pass\n```"
        ext = FeatureExtractor().extract(prompt)
        d = ext.features.as_dict()
        assert d["has_code_fence"] == 1.0
        assert d["has_python"] == 1.0

    def test_sql_detection(self) -> None:
        ext = FeatureExtractor().extract("SELECT * FROM users WHERE id = 1")
        assert ext.features.as_dict()["has_sql"] == 1.0

    def test_extraction_speed(self) -> None:
        prompt = "Explain machine learning " * 50
        times = []
        extractor = FeatureExtractor()
        for _ in range(100):
            t0 = time.perf_counter()
            extractor.extract(prompt)
            times.append((time.perf_counter() - t0) * 1000)
        assert sum(times) / len(times) < 5.0  # generous on CI

    def test_messages_format(self) -> None:
        result = extract_from_messages(
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        )
        assert result.features.as_dict()["conversation_turn_count"] >= 1

    def test_batch_extract(self) -> None:
        results = FeatureExtractor().extract_batch(["a", "b", "c"])
        assert len(results) == 3
