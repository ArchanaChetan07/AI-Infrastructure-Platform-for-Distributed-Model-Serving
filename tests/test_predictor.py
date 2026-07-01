"""Tests for output-length predictor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from predictor.dataset import OutputLengthDataset
from predictor.model import ModelConfig, OutputLengthMLP, huber_loss
from predictor.predictor import OutputLengthPredictor


@pytest.mark.unit
class TestModel:
    def test_forward_shape(self) -> None:
        model = OutputLengthMLP()
        x = torch.randn(8, ModelConfig().input_dim)
        out = model(x)
        assert out.shape == (8,)

    def test_output_clamped(self) -> None:
        model = OutputLengthMLP()
        x = torch.randn(4, ModelConfig().input_dim) * 100
        out = model(x)
        assert (out >= model.config.output_min).all()
        assert (out <= model.config.output_max).all()

    def test_huber_loss(self) -> None:
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([1.5, 2.5])
        loss = huber_loss(pred, target)
        assert loss.item() > 0


@pytest.mark.unit
class TestPredictor:
    def test_predict(self) -> None:
        model = OutputLengthMLP()
        predictor = OutputLengthPredictor(model)
        result = predictor.predict("What is AI?")
        assert result.predicted_tokens > 0
        assert result.total_latency_ms < 100

    def test_normalization(self) -> None:
        model = OutputLengthMLP()
        mean = np.zeros(ModelConfig().input_dim)
        std = np.ones(ModelConfig().input_dim)
        predictor = OutputLengthPredictor(model, feature_mean=mean, feature_std=std)
        result = predictor.predict("test prompt")
        assert result.predicted_tokens > 0


@pytest.mark.unit
class TestDataset:
    def test_synthetic_generation(self) -> None:
        ds = OutputLengthDataset()
        samples = ds.generate_synthetic(100)
        assert len(samples) == 100
        assert all(len(s.features) == ModelConfig().input_dim for s in samples)

    def test_deduplication(self) -> None:
        ds = OutputLengthDataset()
        samples = ds.generate_synthetic(10)
        duped = samples + samples
        deduped = ds.deduplicate(duped)
        assert len(deduped) == len(samples)

    def test_split(self) -> None:
        ds = OutputLengthDataset()
        samples = ds.generate_synthetic(200)
        splits = ds.split(samples)
        assert len(splits.train) + len(splits.val) + len(splits.test) == 200
