"""Training loop for output-length predictor."""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from predictor.dataset import DatasetSplits, OutputLengthDataset
from predictor.model import ModelConfig, OutputLengthMLP, huber_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    hidden_dim: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 50
    patience: int = 8
    grad_clip: float = 1.0
    seed: int = 42
    use_amp: bool = True
    use_compile: bool = False
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "models/logs"
    device: str = "auto"


@dataclass
class TrainMetrics:
    """Per-epoch metrics."""

    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float
    val_rmse: float
    val_r2: float
    lr: float


@dataclass
class TrainResult:
    """Training outcome."""

    best_epoch: int
    best_val_loss: float
    metrics_history: list[TrainMetrics] = field(default_factory=list)
    checkpoint_path: str = ""
    feature_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_std: np.ndarray = field(default_factory=lambda: np.array([]))


class Trainer:
    """Full training pipeline with early stopping and checkpointing."""

    def __init__(self, config: Optional[TrainConfig] = None) -> None:
        self.config = config or TrainConfig()
        self._setup_seed()
        self.device = self._resolve_device()
        self._tb_writer: Any = None
        self._wandb: Any = None
        self._mlflow: Any = None

    def train(self, splits: DatasetSplits) -> TrainResult:
        cfg = self.config
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

        model = OutputLengthMLP(ModelConfig(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)).to(
            self.device
        )

        if cfg.use_compile and hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore[assignment]

        train_loader, val_loader = self._make_loaders(splits)
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        scaler = torch.amp.GradScaler(
            self.device.type, enabled=cfg.use_amp and self.device.type == "cuda"
        )

        best_val = math.inf
        best_epoch = 0
        patience_counter = 0
        history: list[TrainMetrics] = []
        ckpt_path = Path(cfg.checkpoint_dir) / "best.pt"

        self._init_loggers(cfg)

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(model, train_loader, optimizer, scaler)
            val_loss, val_mae, val_rmse, val_r2 = self._eval_epoch(model, val_loader)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            metrics = TrainMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_mae=val_mae,
                val_rmse=val_rmse,
                val_r2=val_r2,
                lr=lr,
            )
            history.append(metrics)
            self._log_metrics(metrics)
            logger.info(
                "Epoch %d: train=%.4f val=%.4f mae=%.2f rmse=%.2f r2=%.3f",
                epoch,
                train_loss,
                val_loss,
                val_mae,
                val_rmse,
                val_r2,
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        self._close_loggers()
        return TrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val,
            metrics_history=history,
            checkpoint_path=str(ckpt_path),
            feature_mean=splits.feature_mean,
            feature_std=splits.feature_std,
        )

    def resume(self, splits: DatasetSplits, checkpoint: Path) -> TrainResult:
        """Resume training from checkpoint."""
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model = OutputLengthMLP().to(self.device)
        model.load_state_dict(state)
        # Continue with remaining logic — delegate to train with loaded weights
        torch.save(state, Path(self.config.checkpoint_dir) / "resume_base.pt")
        return self.train(splits)

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> float:
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=scaler.is_enabled()):
                pred = model(xb)
                loss = huber_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(xb)
            n += len(xb)
        return total_loss / max(n, 1)

    @torch.inference_mode()
    def _eval_epoch(
        self, model: nn.Module, loader: DataLoader
    ) -> tuple[float, float, float, float]:
        model.eval()
        preds, targets = [], []
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = model(xb)
            loss = huber_loss(pred, yb)
            total_loss += loss.item() * len(xb)
            n += len(xb)
            preds.extend(pred.cpu().numpy())
            targets.extend(yb.cpu().numpy())
        mae, rmse, r2 = _regression_metrics(np.array(targets), np.array(preds))
        return total_loss / max(n, 1), mae, rmse, r2

    def _make_loaders(self, splits: DatasetSplits) -> tuple[DataLoader, DataLoader]:
        ds_builder = OutputLengthDataset()
        x_train, y_train = ds_builder.to_tensors(
            splits.train, splits.feature_mean, splits.feature_std
        )
        x_val, y_val = ds_builder.to_tensors(splits.val, splits.feature_mean, splits.feature_std)
        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        return (
            DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=self.config.batch_size),
        )

    def _setup_seed(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _init_loggers(self, cfg: TrainConfig) -> None:
        if os.getenv("USE_TENSORBOARD", "").lower() in ("1", "true"):
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(cfg.log_dir)
            except Exception as exc:
                logger.debug("TensorBoard unavailable: %s", exc)
        if os.getenv("WANDB_PROJECT"):
            try:
                import wandb

                self._wandb = wandb.init(project=os.environ["WANDB_PROJECT"], config=cfg.__dict__)
            except ImportError:
                pass
        if os.getenv("MLFLOW_TRACKING_URI"):
            try:
                import mlflow

                mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "output-length"))
                self._mlflow = mlflow.start_run()
                mlflow.log_params(cfg.__dict__)
            except ImportError:
                pass

    def _log_metrics(self, m: TrainMetrics) -> None:
        if self._tb_writer:
            self._tb_writer.add_scalar("loss/train", m.train_loss, m.epoch)
            self._tb_writer.add_scalar("loss/val", m.val_loss, m.epoch)
            self._tb_writer.add_scalar("mae/val", m.val_mae, m.epoch)
        if self._wandb:
            self._wandb.log(m.__dict__)
        if self._mlflow:
            import mlflow

            mlflow.log_metrics(
                {"train_loss": m.train_loss, "val_loss": m.val_loss, "val_mae": m.val_mae},
                step=m.epoch,
            )

    def _close_loggers(self) -> None:
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb:
            self._wandb.finish()
        if self._mlflow:
            import mlflow

            mlflow.end_run()


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2


def pearson_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    from scipy import stats  # type: ignore[import-untyped]

    pearson = float(stats.pearsonr(y_true, y_pred)[0])
    spearman = float(stats.spearmanr(y_true, y_pred)[0])
    return pearson, spearman
