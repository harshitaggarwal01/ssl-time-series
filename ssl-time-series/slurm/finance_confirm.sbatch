#!/bin/bash
# Track 2 confirmation: momentum + multiscale together, 2k-step baseline config, seeds 0/1/2.
#   sbatch slurm/finance_ablation.sbatch  (submit from ssl-time-series/ssl-time-series)
# Needs the Track 2 worktree at ../../wt-finance.
#SBATCH --job-name=ssl-fin3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from ssl-time-series/ssl-time-series}"
FIN="$PROJECT/../../wt-finance/ssl-time-series"
PY="$PROJECT/.venv/bin/python"
cd "$FIN"; mkdir -p outputs
nvidia-smi --query-gpu=name --format=csv,noheader
for SEED in 0 1 2; do
  G=mm-s$SEED; NAME=fin-$G
  echo "=== $NAME"
  start=$(date +%s)
  "$PY" -m scripts.train_encoder --corpus "$PROJECT/corpus" --config configs/baseline.json \
      --workers 4 --checkpoint-every 0 --output outputs/$NAME.pt \
      --seed $SEED --feature-groups momentum multiscale | tail -2 || { echo "TRAIN FAILED $NAME"; continue; }
  echo "train_seconds=$(( $(date +%s) - start ))"
  "$PY" -m scripts.export_model --checkpoint outputs/$NAME/step-002000.pt --output "$PROJECT/outputs/$NAME.onnx" || { echo "EXPORT FAILED $NAME"; continue; }
  (cd "$PROJECT" && "$PY" test_submission.py outputs/$NAME.onnx > outputs/$NAME.test_submission.txt 2>&1 \
      && echo "test_submission: all passed" || echo "test_submission: FAILED")
done

cd "$PROJECT"
for G in mm-s0 mm-s1 mm-s2; do
  [[ -f outputs/fin-$G.onnx ]] && "$PY" evaluate.py outputs/fin-$G.onnx outputs/proxy_suite --out outputs/eval/fin-$G.json || echo "EVAL FAILED fin-$G"
done
"$PY" - <<'PY'
import json
from pathlib import Path
names = [t["name"] for t in json.loads(Path("outputs/proxy_suite/tasks.json").read_text())["tasks"]]
lines = ["| model | score | mean AUC | " + " | ".join(names) + " |", "|---" * (len(names) + 3) + "|"]
for f in ["baseline", "baseline-s1", "baseline-s2"] + [f"fin-{g}" for g in
          "momentum multiscale mm-s0 mm-s1 mm-s2".split()]:
    if not Path(f"outputs/eval/{f}.json").exists(): continue
    r = json.loads(Path(f"outputs/eval/{f}.json").read_text())
    a = [t["auc"] for t in r["tasks"].values()]
    lines.append(f"| {f} | {r['score']:.3f} | {sum(a)/len(a):.3f} | " + " | ".join(f"{x:.3f}" for x in a) + " |")
Path("outputs/eval/FINANCE_CONFIRM.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
