"""MLP architecture for output-length prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from predictor.features import NUM_FEATURES


@dataclass
class ModelConfig:
    """Hyperparameters for the output-length MLP."""

    input_dim: int = NUM_FEATURES
    hidden_dim: int = 128
    dropout: float = 0.1
    output_min: float = 1.0
    output_max: float = 4096.0


class OutputLengthMLP(nn.Module):
    """Two-layer MLP with LayerNorm, ReLU, and Dropout."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LayerNorm(cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted output token count (non-negative)."""
        out = self.net(x).squeeze(-1)
        return torch.clamp(out, min=self.config.output_min, max=self.config.output_max)

    @torch.inference_mode()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(features)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return nn.functional.huber_loss(pred, target, delta=delta)
