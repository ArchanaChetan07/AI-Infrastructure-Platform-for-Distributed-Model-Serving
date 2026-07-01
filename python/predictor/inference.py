"""Fast inference utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from predictor.features import FeatureVector
from predictor.predictor import OutputLengthPredictor


def _load_norm_arrays(
    path: Optional[Path],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if path is None or not path.exists():
        return None, None
    import json

    with open(path) as f:
        meta = json.load(f)
    return (
        np.array(meta.get("feature_mean", []), dtype=np.float32),
        np.array(meta.get("feature_std", []), dtype=np.float32),
    )


class ONNXPredictor:
    """ONNX Runtime inference backend."""

    def __init__(self, onnx_path: str | Path, metadata_path: Optional[Path] = None) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self._mean, self._std = _load_norm_arrays(metadata_path)

    def predict(self, features: FeatureVector) -> tuple[float, float]:
        t0 = time.perf_counter()
        arr = np.array(features.as_list(), dtype=np.float32)
        if self._mean is not None:
            std = np.where(self._std < 1e-8, 1.0, self._std)
            arr = (arr - self._mean) / std
        out = self.session.run(None, {self.input_name: arr.reshape(1, -1)})[0]
        ms = (time.perf_counter() - t0) * 1000.0
        return float(out.squeeze()), ms


class TorchScriptPredictor:
    """TorchScript inference backend."""

    def __init__(self, ts_path: str | Path, metadata_path: Optional[Path] = None) -> None:
        self.model = torch.jit.load(str(ts_path), map_location="cpu")
        self.model.eval()
        self._mean, self._std = _load_norm_arrays(metadata_path)

    @torch.inference_mode()
    def predict(self, features: FeatureVector) -> tuple[float, float]:
        t0 = time.perf_counter()
        arr = np.array(features.as_list(), dtype=np.float32)
        if self._mean is not None and self._std is not None:
            std = np.where(self._std < 1e-8, 1.0, self._std)
            arr = (arr - self._mean) / std
        tensor = torch.from_numpy(arr).unsqueeze(0)
        out = self.model(tensor).item()
        ms = (time.perf_counter() - t0) * 1000.0
        return float(out), ms


def load_predictor(
    models_dir: str | Path = "models",
    backend: str = "pytorch",
) -> OutputLengthPredictor | ONNXPredictor | TorchScriptPredictor:
    """Load predictor from exported artifacts."""
    models_dir = Path(models_dir)
    meta = models_dir / "metadata.json"
    if backend == "onnx":
        return ONNXPredictor(models_dir / "output_length.onnx", meta)
    if backend == "torchscript":
        return TorchScriptPredictor(models_dir / "output_length.ts", meta)
    return OutputLengthPredictor.from_checkpoint(models_dir / "output_length_mlp.pt", meta)
