"""Model export to PyTorch, TorchScript, ONNX."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from predictor.features import FEATURE_NAMES, NUM_FEATURES
from predictor.model import ModelConfig, OutputLengthMLP


@dataclass
class ExportMetadata:
    """Model artifact metadata."""

    model_version: str
    feature_names: list[str]
    num_features: int
    feature_mean: list[float]
    feature_std: list[float]
    hidden_dim: int
    dropout: float
    training_date: str
    git_sha: str
    metrics: dict[str, float]


def export_model(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    metrics: Optional[dict[str, float]] = None,
    model_version: str = "1.0.0",
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> ExportMetadata:
    """Export PyTorch, TorchScript, and ONNX artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)

    model = OutputLengthMLP(ModelConfig(hidden_dim=hidden_dim, dropout=dropout))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    pt_path = output_dir / "output_length_mlp.pt"
    torch.save(model.state_dict(), pt_path)

    dummy = torch.randn(1, NUM_FEATURES)
    ts_path = output_dir / "output_length.ts"
    traced = torch.jit.trace(model, dummy)
    traced.save(str(ts_path))

    onnx_path = output_dir / "output_length.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["features"],
        output_names=["predicted_tokens"],
        dynamic_axes={"features": {0: "batch"}, "predicted_tokens": {0: "batch"}},
        opset_version=17,
    )

    meta = ExportMetadata(
        model_version=model_version,
        feature_names=list(FEATURE_NAMES),
        num_features=NUM_FEATURES,
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        hidden_dim=hidden_dim,
        dropout=dropout,
        training_date=datetime.now(timezone.utc).isoformat(),
        git_sha=_git_sha(),
        metrics=metrics or {},
    )
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)

    return meta


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()[:12]
        )
    except Exception:
        return "unknown"
