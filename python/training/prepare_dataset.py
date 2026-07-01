#!/usr/bin/env python3
"""Prepare training dataset from presets or files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from predictor.dataset import OutputLengthDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--output", default="data/processed/dataset.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["dataset"]
    builder = OutputLengthDataset()
    samples = builder.load_preset(ds_cfg["preset"], max_samples=ds_cfg["max_samples"])
    samples = builder.deduplicate(samples)
    splits = builder.split(
        samples,
        train_ratio=ds_cfg["train_ratio"],
        val_ratio=ds_cfg["val_ratio"],
        seed=ds_cfg.get("seed", 42),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in splits.train + splits.val + splits.test:
            f.write(
                json.dumps(
                    {
                        "prompt": s.prompt,
                        "features": s.features,
                        "actual_output_tokens": s.actual_output_tokens,
                        "metadata": s.metadata,
                    }
                )
                + "\n"
            )

    meta_path = out.parent / "splits_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "train": len(splits.train),
                "val": len(splits.val),
                "test": len(splits.test),
                "feature_mean": splits.feature_mean.tolist(),
                "feature_std": splits.feature_std.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved {len(samples)} samples to {out}")


if __name__ == "__main__":
    main()
