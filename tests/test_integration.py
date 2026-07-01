"""Integration tests for scheduler system."""

from __future__ import annotations

import pytest

from benchmark.fcfs import run_fcfs_benchmark
from benchmark.sjf import run_oracle_benchmark, run_sjf_benchmark
from predictor.dataset import OutputLengthDataset
from predictor.export import export_model
from predictor.trainer import TrainConfig, Trainer


@pytest.mark.integration
class TestEndToEnd:
    def test_train_export_predict(self, tmp_path) -> None:
        builder = OutputLengthDataset()
        samples = builder.generate_synthetic(200)
        splits = builder.split(samples)
        config = TrainConfig(
            epochs=2,
            patience=2,
            batch_size=32,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            use_amp=False,
        )
        result = Trainer(config).train(splits)
        models_dir = tmp_path / "models"
        export_model(
            result.checkpoint_path,
            models_dir,
            splits.feature_mean,
            splits.feature_std,
        )
        assert (models_dir / "output_length_mlp.pt").exists()
        assert (models_dir / "output_length.onnx").exists()
        assert (models_dir / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_benchmark_comparison(self, tmp_path) -> None:
        fcfs = await run_fcfs_benchmark(2, 10)
        oracle = await run_oracle_benchmark(2, 10)
        assert fcfs["n_success"] == 10
        assert oracle["n_success"] == 10
        assert oracle["e2e_p50_ms"] <= fcfs["e2e_p50_ms"] + 1000  # oracle should be competitive

    @pytest.mark.asyncio
    async def test_full_pipeline_with_model(self, tmp_path) -> None:
        builder = OutputLengthDataset()
        samples = builder.generate_synthetic(200)
        splits = builder.split(samples)
        config = TrainConfig(
            epochs=2,
            batch_size=32,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            use_amp=False,
        )
        result = Trainer(config).train(splits)
        models_dir = tmp_path / "models"
        export_model(result.checkpoint_path, models_dir, splits.feature_mean, splits.feature_std)
        sjf = await run_sjf_benchmark(2, 10, str(models_dir))
        assert sjf["n_success"] == 10
