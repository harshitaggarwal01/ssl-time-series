"""Print reproducible training commands for the planned model ablations."""

from __future__ import annotations

import argparse
from pathlib import Path


ABLATIONS = {
    "baseline": "--embed-dim 256 --patch-size 32 --depth 4 --heads 4",
    "normalized": "--embed-dim 256 --patch-size 32 --depth 4 --heads 4",
    "masked-contrastive": "--embed-dim 320 --patch-size 32 --depth 4 --heads 5",
    "compact": "--embed-dim 256 --patch-size 64 --depth 4 --heads 4",
    "main": "--embed-dim 320 --patch-size 32 --depth 4 --heads 5",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2_000)
    args = parser.parse_args()
    for name, flags in ABLATIONS.items():
        print(
            f"uv run python -m scripts.train_encoder --corpus {args.corpus} "
            f"--steps {args.steps} --output outputs/{name}.pt {flags}"
        )


if __name__ == "__main__":
    main()
