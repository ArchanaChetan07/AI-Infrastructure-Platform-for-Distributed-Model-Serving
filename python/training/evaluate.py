#!/usr/bin/env python3
"""Evaluate trained predictor on test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from predictor.dataset import OutputLengthDataset
from predictor.model import OutputLengthMLP
from predictor.trainer import pearson_spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--checkpoint", default="shared/models/output_length_mlp.pt")
    parser.add_argument("--output", default="shared/models/evaluation.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    builder = OutputLengthDataset()
    samples = builder.load_preset(
        cfg["dataset"]["preset"], max_samples=cfg["dataset"]["max_samples"]
    )
    splits = builder.split(samples, seed=cfg["dataset"].get("seed", 42))

    model = OutputLengthMLP()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    x, y = builder.to_tensors(splits.test, splits.feature_mean, splits.feature_std)
    with torch.inference_mode():
        preds = model(torch.from_numpy(x)).numpy()

    mae = float(np.mean(np.abs(y - preds)))
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pearson, spearman = pearson_spearman(y, preds)

    results = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "n_test": len(y),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
