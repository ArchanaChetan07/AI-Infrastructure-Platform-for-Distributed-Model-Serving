"""Dataset loading and preprocessing for output-length prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from predictor.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Single training/evaluation sample."""

    prompt: str
    features: list[float]
    actual_output_tokens: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplits:
    """Train/val/test splits."""

    train: list[Sample]
    val: list[Sample]
    test: list[Sample]
    feature_mean: np.ndarray
    feature_std: np.ndarray


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Heuristic token estimate when ground truth unavailable."""
    words = text.split()
    return max(1, int(len(words) * 1.3))


class OutputLengthDataset:
    """Build datasets from multiple sources."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or Path(".cache/datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = FeatureExtractor()

    def from_prompts(
        self,
        prompts: Sequence[str],
        outputs: Sequence[str],
        metadata: Optional[Sequence[dict[str, Any]]] = None,
    ) -> list[Sample]:
        meta_list = metadata or [{}] * len(prompts)
        samples: list[Sample] = []
        for prompt, output, meta in zip(prompts, outputs, meta_list, strict=True):
            ext = self.extractor.extract(prompt)
            samples.append(
                Sample(
                    prompt=prompt,
                    features=ext.features.as_list(),
                    actual_output_tokens=float(_estimate_tokens(output)),
                    metadata=meta,
                )
            )
        return samples

    def from_jsonl(self, path: Path) -> list[Sample]:
        records: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return self._from_records(records)

    def from_csv(
        self, path: Path, prompt_col: str = "prompt", output_col: str = "output"
    ) -> list[Sample]:
        records = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({"prompt": row[prompt_col], "output": row.get(output_col, "")})
        return self._from_records(records)

    def from_huggingface(
        self,
        name: str,
        split: str = "train",
        max_samples: int = 5000,
        prompt_key: str = "instruction",
        output_key: str = "output",
    ) -> list[Sample]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("Install datasets: pip install datasets") from e

        cache_key = f"{name}_{split}_{max_samples}".replace("/", "_")
        cache_path = self.cache_dir / f"{cache_key}.jsonl"
        if cache_path.exists():
            return self.from_jsonl(cache_path)

        ds = load_dataset(name, split=split, streaming=False)
        records: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            prompt = str(row.get(prompt_key, row.get("text", "")))
            output = str(row.get(output_key, row.get("response", "")))
            if prompt:
                records.append({"prompt": prompt, "output": output})
        self._write_jsonl(cache_path, records)
        return self._from_records(records)

    def load_preset(self, name: str, max_samples: int = 2000) -> list[Sample]:
        presets: dict[str, Callable[[], list[Sample]]] = {
            "alpaca": lambda: self.from_huggingface(
                "yahma/alpaca-cleaned", max_samples=max_samples
            ),
            "dolly": lambda: self.from_huggingface(
                "databricks/databricks-dolly-15k",
                prompt_key="instruction",
                output_key="response",
                max_samples=max_samples,
            ),
            "synthetic": lambda: self.generate_synthetic(max_samples),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Choose from {list(presets)}")
        return presets[name]()

    def generate_synthetic(self, n: int = 5000) -> list[Sample]:
        """Generate realistic synthetic training data."""
        import random

        rng = random.Random(42)
        templates = [
            ("What is {topic}?", "short", 20, 80),
            ("Explain {topic} in detail.", "medium", 80, 200),
            ("Write a Python function to {task}.", "code", 100, 350),
            ("Compare and contrast {a} and {b}.", "reasoning", 150, 400),
            ("List 10 facts about {topic}.", "list", 60, 180),
            ("Translate to French: {sentence}", "short", 15, 60),
            ("Debug this code:\n```python\n{code}\n```", "code", 120, 500),
            ("Summarize: {paragraph}", "medium", 40, 150),
        ]
        topics = ["AI", "physics", "history", "Python", "databases", "Kubernetes"]
        samples: list[Sample] = []
        seen: set[str] = set()

        for i in range(n * 3):
            if len(samples) >= n:
                break
            tmpl, _, lo, hi = rng.choice(templates)
            topic = rng.choice(topics)
            prompt = tmpl.format(
                topic=f"{topic}-{i}",
                task=f"process {topic.lower()}-{i}",
                a=rng.choice(topics),
                b=rng.choice(topics),
                sentence=f"The {topic.lower()} is important number {i}.",
                code=f"def f_{i}(x):\n    return x + {i}",
                paragraph=" ".join(
                    [f"Sentence {j} about {topic} item {i}." for j in range(rng.randint(3, 8))]
                ),
            )
            h = _hash_prompt(prompt)
            if h in seen:
                continue
            seen.add(h)
            target = float(rng.randint(lo, hi))
            ext = self.extractor.extract(prompt)
            # Blend heuristic with features for realistic correlation
            feat_boost = sum(ext.features.as_list()[:5]) * 0.5
            target = max(5.0, target + feat_boost * 0.1)
            samples.append(
                Sample(
                    prompt=prompt,
                    features=ext.features.as_list(),
                    actual_output_tokens=target,
                    metadata={"source": "synthetic"},
                )
            )
        return samples

    def deduplicate(self, samples: list[Sample]) -> list[Sample]:
        seen: set[str] = set()
        out: list[Sample] = []
        for s in samples:
            h = _hash_prompt(s.prompt)
            if h not in seen:
                seen.add(h)
                out.append(s)
        return out

    def split(
        self,
        samples: list[Sample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplits:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(samples))
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train = [samples[i] for i in train_idx]
        val = [samples[i] for i in val_idx]
        test = [samples[i] for i in test_idx]

        all_features = np.array([s.features for s in train], dtype=np.float32)
        mean = all_features.mean(axis=0)
        std = all_features.std(axis=0)
        return DatasetSplits(train=train, val=val, test=test, feature_mean=mean, feature_std=std)

    def _from_records(self, records: list[dict[str, Any]]) -> list[Sample]:
        prompts, outputs, metas = [], [], []
        for rec in records:
            prompt = str(rec.get("prompt", rec.get("instruction", rec.get("input", ""))))
            output = str(rec.get("output", rec.get("response", rec.get("completion", ""))))
            if prompt:
                prompts.append(prompt)
                outputs.append(output)
                metas.append({k: v for k, v in rec.items() if k not in ("prompt", "output")})
        return self.from_prompts(prompts, outputs, metas)

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def to_tensors(
        self, samples: list[Sample], mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([s.features for s in samples], dtype=np.float32)
        y = np.array([s.actual_output_tokens for s in samples], dtype=np.float32)
        std_safe = np.where(std < 1e-8, 1.0, std)
        x = (x - mean) / std_safe
        return x, y


def _parallel_extract_work(prompt: str) -> list[float]:
    """Module-level worker for multiprocessing (must be picklable on Windows)."""
    extractor = FeatureExtractor()
    return extractor.extract(prompt).features.as_list()


def parallel_extract(prompts: Sequence[str], workers: int = mp.cpu_count()) -> list[list[float]]:
    with mp.Pool(workers) as pool:
        return list(pool.map(_parallel_extract_work, prompts))
