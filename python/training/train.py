#!/usr/bin/env python3
"""Train output-length prediction model."""

from __future__ import annotations

import argparse

import numpy as np
import yaml
from predictor.dataset import OutputLengthDataset
from predictor.export import export_model
from predictor.trainer import TrainConfig, Trainer, pearson_spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    model_cfg = cfg["model"]
    export_cfg = cfg["export"]

    builder = OutputLengthDataset()
    samples = builder.load_preset(ds_cfg["preset"], max_samples=ds_cfg["max_samples"])
    samples = builder.deduplicate(samples)
    splits = builder.split(
        samples,
        train_ratio=ds_cfg["train_ratio"],
        val_ratio=ds_cfg["val_ratio"],
        seed=ds_cfg.get("seed", 42),
    )

    train_config = TrainConfig(
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
        batch_size=tr_cfg["batch_size"],
        epochs=tr_cfg["epochs"],
        patience=tr_cfg["patience"],
        grad_clip=tr_cfg["grad_clip"],
        use_amp=tr_cfg["use_amp"],
        use_compile=tr_cfg.get("use_compile", False),
        seed=tr_cfg["seed"],
        device=tr_cfg.get("device", "auto"),
    )

    trainer = Trainer(train_config)
    result = trainer.train(splits)

    # Evaluate on test set
    import torch
    from predictor.model import OutputLengthMLP

    model = OutputLengthMLP()
    model.load_state_dict(torch.load(result.checkpoint_path, weights_only=True))
    model.eval()
    x_test, y_test = builder.to_tensors(splits.test, splits.feature_mean, splits.feature_std)
    with torch.inference_mode():
        preds = model(torch.from_numpy(x_test)).numpy()
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    pearson, spearman = pearson_spearman(y_test, preds)

    metrics = {
        "test_mae": mae,
        "test_rmse": rmse,
        "test_pearson": pearson,
        "test_spearman": spearman,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
    }
    print(f"Test MAE={mae:.2f} RMSE={rmse:.2f} Pearson={pearson:.3f} Spearman={spearman:.3f}")

    export_model(
        result.checkpoint_path,
        export_cfg["output_dir"],
        splits.feature_mean,
        splits.feature_std,
        metrics=metrics,
        model_version=export_cfg.get("model_version", "1.0.0"),
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
    )
    print(f"Exported model to {export_cfg['output_dir']}")


if __name__ == "__main__":
    main()
