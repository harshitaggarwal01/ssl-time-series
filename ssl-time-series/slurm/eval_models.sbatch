#!/bin/bash
# Score ONNX candidates with the official evaluate.py on the synthetic proxy suite (GPU, like the leaderboard).
#   sbatch slurm/eval_models.sbatch outputs/baseline.onnx outputs/baseline-s1.onnx ...
# Builds + calibrates outputs/proxy_suite first if it does not exist (floor = untrained encoder,
# reference = example_submission/model.onnx). Writes outputs/eval/<name>.json and a combined table.
#SBATCH --job-name=ssl-eval
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from ssl-time-series/ssl-time-series}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.uv-cache}"  # keep multi-GB wheels out of $HOME
cd "$PROJECT"; mkdir -p outputs/eval
SUITE=outputs/proxy_suite
(
  flock 9
  if [[ ! -f "$SUITE/.calibrated" ]]; then
    rm -rf "$SUITE"
    time uv run --no-sync python -m scripts.build_proxy_suite --out "$SUITE"
    touch "$SUITE/.calibrated"
  fi
) 9>"outputs/.proxy_suite.lock"
cat "$SUITE/tasks.json"

for MODEL in example_submission/model.onnx "$@"; do
  [[ -f "$MODEL" ]] || { echo "MISSING $MODEL"; continue; }
  NAME=$(basename "$(dirname "$MODEL")")-$(basename "$MODEL" .onnx)
  [[ "$MODEL" == outputs/* ]] && NAME=$(basename "$MODEL" .onnx)
  echo "=== $NAME ($MODEL)"
  uv run --no-sync python evaluate.py "$MODEL" "$SUITE" --out "outputs/eval/$NAME.json"
done

uv run --no-sync python - <<'PY'
import json
from pathlib import Path
rows = []
for f in sorted(Path("outputs/eval").glob("*.json")):
    r = json.loads(f.read_text())
    aucs = [t["auc"] for t in r["tasks"].values()]
    rows.append((f.stem, r["score"], sum(aucs) / len(aucs), aucs))
names = [t["name"] for t in json.loads(Path("outputs/proxy_suite/tasks.json").read_text())["tasks"]]
lines = ["| model | score | mean AUC | " + " | ".join(names) + " |", "|---" * (len(names) + 3) + "|"]
for name, score, mean, aucs in sorted(rows, key=lambda r: -r[1]):
    lines.append(f"| {name} | {score:.3f} | {mean:.3f} | " + " | ".join(f"{a:.3f}" for a in aucs) + " |")
Path("outputs/eval/TABLE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
