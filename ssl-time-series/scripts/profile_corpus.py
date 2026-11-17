"""Profile corpus lengths and sampled numeric ranges without loading it into RAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import corpus_loader


def profile(root: str | Path, samples: int = 10_000) -> dict:
    get, total = corpus_loader.load(root)
    rng = np.random.default_rng(0)
    indices = rng.choice(total, size=min(samples, total), replace=False)
    lengths = np.empty(len(indices), dtype=np.int64)
    finite_fraction = []
    means = []
    scales = []
    for j, index in enumerate(indices):
        values = np.asarray(get(int(index)), dtype=np.float32)
        lengths[j] = len(values)
        finite = np.isfinite(values)
        finite_fraction.append(float(finite.mean()) if len(values) else 0.0)
        clean = values[finite]
        means.append(float(clean.mean()) if len(clean) else 0.0)
        scales.append(float(clean.std()) if len(clean) else 0.0)
    return {
        "series": total,
        "sampled_series": len(indices),
        "length": {
            "min": int(lengths.min(initial=0)),
            "median": float(np.median(lengths)) if len(lengths) else 0.0,
            "p90": float(np.quantile(lengths, 0.9)) if len(lengths) else 0.0,
            "p99": float(np.quantile(lengths, 0.99)) if len(lengths) else 0.0,
            "max": int(lengths.max(initial=0)),
        },
        "finite_fraction": {
            "mean": float(np.mean(finite_fraction)) if finite_fraction else 0.0,
            "minimum": float(np.min(finite_fraction)) if finite_fraction else 0.0,
        },
        "sample_mean": {
            "median": float(np.median(means)) if means else 0.0,
            "p99_abs": float(np.quantile(np.abs(means), 0.99)) if means else 0.0,
        },
        "sample_std": {
            "median": float(np.median(scales)) if scales else 0.0,
            "p99": float(np.quantile(scales, 0.99)) if scales else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("outputs/corpus_profile.json"))
    args = parser.parse_args()
    result = profile(args.corpus, args.samples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
