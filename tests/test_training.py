"""Tests for training pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from predictor.dataset import OutputLengthDataset
from predictor.trainer import TrainConfig, Trainer, _regression_metrics


@pytest.mark.unit
class TestTraining:
    def test_regression_metrics(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.2])
        mae, rmse, r2 = _regression_metrics(y, pred)
        assert mae < 0.5
        assert rmse < 0.5
        assert r2 > 0.5

    def test_short_training_run(self, tmp_path) -> None:
        builder = OutputLengthDataset()
        samples = builder.generate_synthetic(300)
        splits = builder.split(samples)

        config = TrainConfig(
            epochs=3,
            patience=2,
            batch_size=64,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            use_amp=False,
        )
        trainer = Trainer(config)
        result = trainer.train(splits)
        assert result.best_epoch >= 1
        assert (tmp_path / "ckpt" / "best.pt").exists()
