"""High-level predictor interface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from predictor.feature_extractor import FeatureExtractor
from predictor.features import FeatureVector
from predictor.model import OutputLengthMLP


@dataclass(frozen=True)
class PredictionResult:
    """Output-length prediction with timing."""

    predicted_tokens: float
    features: FeatureVector
    feature_latency_ms: float
    predict_latency_ms: float

    @property
    def total_latency_ms(self) -> float:
        return self.feature_latency_ms + self.predict_latency_ms


class OutputLengthPredictor:
    """End-to-end feature extraction + inference."""

    def __init__(
        self,
        model: OutputLengthMLP,
        feature_extractor: Optional[FeatureExtractor] = None,
        device: Optional[str] = None,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.extractor = feature_extractor or FeatureExtractor()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()
        self._mean = feature_mean
        self._std = feature_std

    def predict(self, prompt: str) -> PredictionResult:
        ext = self.extractor.extract(prompt)
        t0 = time.perf_counter()
        tensor = self._to_tensor(ext.features)
        with torch.inference_mode():
            pred = self.model.predict(tensor).item()
        predict_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionResult(
            predicted_tokens=float(pred),
            features=ext.features,
            feature_latency_ms=ext.latency_ms,
            predict_latency_ms=predict_ms,
        )

    def predict_batch(self, prompts: Sequence[str]) -> list[PredictionResult]:
        return [self.predict(p) for p in prompts]

    def _to_tensor(self, features: FeatureVector) -> torch.Tensor:
        arr = np.array(features.as_list(), dtype=np.float32)
        if self._mean is not None and self._std is not None:
            mean = np.asarray(self._mean, dtype=np.float32)
            std = np.asarray(self._std, dtype=np.float32)
            std = np.where(std < 1e-8, 1.0, std)
            arr = (arr - mean) / std
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        metadata_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> OutputLengthPredictor:
        path = Path(path)
        meta_path = Path(metadata_path) if metadata_path else path.parent / "metadata.json"
        mean = std = None
        if meta_path.exists():
            import json

            with open(meta_path) as f:
                meta = json.load(f)
            mean = np.array(meta.get("feature_mean", []), dtype=np.float32)
            std = np.array(meta.get("feature_std", []), dtype=np.float32)
            if len(mean) == 0:
                mean = std = None

        model = OutputLengthMLP()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(model, device=device, feature_mean=mean, feature_std=std)
