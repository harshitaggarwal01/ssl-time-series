"""Train, export, validate and score one masked-autoencoder experiment.

    uv run python -m scripts.run_experiment NAME --corpus data [scripts.train_mae flags ...]

Writes ``outputs/NAME.pt``, exports it three ways (``NAME.onnx`` with mean pooling, ``NAME.wide.onnx`` with
mean + max + std, ``NAME.hybrid.onnx`` with mean + std and the untrained features) and appends a summary
row to ``outputs/results.jsonl``; the ``_wide`` and ``_hybrid`` fields are the second and third export. The
readout only matters at export, so all three come from one checkpoint.

Needs the proxy set from ``scripts.proxy_ucr`` in ``eval_data/ucr`` and the untrained encoder's scores on it in
``outputs/untrained.ucr.json`` (from ``evaluate.py --out``). On the official leaderboard the ranking is decided
by tasks where trained models fall to or below the untrained floor, so each row also records how many proxy
tasks score below the untrained encoder and the worst such gap.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

OUTPUTS = Path("outputs")
EXPORTS = {  # row-key suffix -> scripts.export_model options
    "": ["--pooling", "mean"],
    "_wide": ["--pooling", "mean_max_std"],
    "_hybrid": ["--pooling", "mean_std", "--fixed"],
}
PROXY = Path("eval_data/ucr")
FLOOR = OUTPUTS / "untrained.ucr.json"


def run(*command) -> str:
    result = subprocess.run([sys.executable, *map(str, command)], check=True, stdout=subprocess.PIPE, text=True)
    return result.stdout


def aucs(path: Path) -> list[float]:
    return [task["auc"] for task in json.loads(path.read_text())["tasks"].values()]


def main() -> None:
    name, flags = sys.argv[1], sys.argv[2:]
    checkpoint = OUTPUTS / f"{name}.pt"
    subprocess.run([sys.executable, "-m", "scripts.train_mae", "--output", str(checkpoint), *flags], check=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    names = [task["name"] for task in json.loads((PROXY / "tasks.json").read_text())["tasks"]]
    floor = aucs(FLOOR)
    row = {
        "name": name,
        "flags": flags,
        "train_loss": state["history"][-1]["loss"],
        **state["validation"],
        "train_seconds": state["history"][-1]["seconds"],
    }
    for tag, options in EXPORTS.items():
        stem = f"{name}{tag.replace('_', '.')}"
        model, scores = OUTPUTS / f"{stem}.onnx", OUTPUTS / f"{stem}.ucr.json"
        run("-m", "scripts.export_model", "--checkpoint", checkpoint, "--output", model, *options)
        run("test_submission.py", model)  # exits non-zero, and stops here, if a check fails
        run("evaluate.py", model, PROXY, "--out", scores)
        synthetic = json.loads(run("-m", "scripts.proxy_evaluate", model).strip().splitlines()[-1])
        task_aucs = aucs(scores)
        gains = [auc - base for auc, base in zip(task_aucs, floor)]
        row[f"ucr_mean_auc{tag}"] = sum(task_aucs) / len(task_aucs)
        row[f"ucr_tasks_below_untrained{tag}"] = sum(gain < 0 for gain in gains)
        row[f"ucr_worst_gain{tag}"] = min(gains)
        row[f"synthetic_auc{tag}"] = synthetic["macro_auc"]
        row[f"onnx_mib{tag}"] = model.stat().st_size / 2**20
        row[f"ucr_auc{tag}"] = dict(zip(names, task_aucs))
    with open(OUTPUTS / "results.jsonl", "a") as results:
        results.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
