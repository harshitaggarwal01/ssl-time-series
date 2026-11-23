"""Table of evaluator scores for Track 4 variants next to the team's reference models.

    DATA=<project dir> python -m scripts.compare_eval [extra model names]

Reads outputs/eval/<name>.json (synthetic suite) and outputs/eval/ucr-{first,held_out}/<name>.json, as written by
slurm/train_contrastive.sbatch and the other export jobs. Missing results show as "-".
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REFERENCES = {  # name -> leaderboard score where known
    "c-conv-s1-nofixed": 1.088,  # recon + InfoNCE, encoder only
    "conv-stats-prof": 1.187,  # recon + InfoNCE + stats + profile
}
VARIANTS = ("ct-infonce", "ct-aug", "ct-ts2vec")


def read(path: Path):
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    tasks = result.get("tasks", {})
    mean_auc = sum(t["auc"] for t in tasks.values()) / len(tasks) if tasks else None
    return result.get("score"), mean_auc


def fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def main() -> None:
    root = Path(os.environ.get("DATA", ".")) / "outputs" / "eval"
    names = list(REFERENCES)[:1] + [f"{v}-enc" for v in VARIANTS]
    names += list(REFERENCES)[1:] + [f"{v}-stats-prof" for v in VARIANTS] + sys.argv[1:]
    print("| model | leaderboard | suite score | suite mean AUC | UCR first (AUC) | UCR held-out (AUC) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in names:
        suite = read(root / f"{name}.json") or (None, None)
        first = read(root / "ucr-first" / f"{name}.json") or (None, None)
        held = read(root / "ucr-held_out" / f"{name}.json") or (None, None)
        print(f"| {name} | {fmt(REFERENCES.get(name))} | {fmt(suite[0])} | {fmt(suite[1])} | {fmt(first[1])} | {fmt(held[1])} |")


if __name__ == "__main__":
    main()
