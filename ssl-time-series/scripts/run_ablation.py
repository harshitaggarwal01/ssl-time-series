"""Print matched training commands for finance feature ablations."""

from __future__ import annotations

import argparse
from pathlib import Path


ARCHITECTURE = "--embed-dim 256 --patch-size 32 --depth 4 --heads 4"
ABLATIONS = {
    # Each addition is tested on top of the same three-channel baseline.
    "finance-baseline": "",
    "finance-multiscale": "--feature-groups multiscale",
    "finance-log-returns": "--feature-groups log_returns",
    "finance-volatility": "--feature-groups volatility",
    "finance-momentum": "--feature-groups momentum",
    "finance-scale": "--feature-groups scale",
    "finance-combined": "--feature-groups multiscale log_returns volatility momentum scale",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    args = parser.parse_args()
    for name, flags in ABLATIONS.items():
        feature_flags = f" {flags}" if flags else ""
        print(
            f"uv run python -m scripts.train_encoder --corpus {args.corpus} "
            f"--steps {args.steps} --seed 0 --output outputs/{name}.pt "
            f"{ARCHITECTURE}{feature_flags}"
        )


if __name__ == "__main__":
    main()
