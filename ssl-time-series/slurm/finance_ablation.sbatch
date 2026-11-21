#!/bin/bash
# Track 2 ablation on Frank's research/finance code: 2k-step baseline config, seed 0, one run per
# feature group plus none (code-equivalence check) and all. Scores each on proxy suite v2.1.
#   sbatch slurm/finance_ablation.sbatch  (submit from ssl-time-series/ssl-time-series)
# Needs the Track 2 worktree at ../../wt-finance.
#SBATCH --job-name=ssl-finance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from ssl-time-series/ssl-time-series}"
FIN="$PROJECT/../../wt-finance/ssl-time-series"
PY="$PROJECT/.venv/bin/python"
cd "$FIN"; mkdir -p outputs
nvidia-smi --query-gpu=name --format=csv,noheader
declare -A GROUPS_FOR=([none]="" [multiscale]=multiscale [log_returns]=log_returns [volatility]=volatility
                       [momentum]=momentum [scale]=scale [all]="multiscale log_returns volatility momentum scale")
for G in none multiscale log_returns volatility momentum scale all; do
  NAME=fin-$G
  echo "=== $NAME: ${GROUPS_FOR[$G]:-(no extra groups)}"
  start=$(date +%s)
  "$PY" -m scripts.train_encoder --corpus "$PROJECT/corpus" --config configs/baseline.json \
      --workers 8 --checkpoint-every 0 --output outputs/$NAME.pt \
      ${GROUPS_FOR[$G]:+--feature-groups ${GROUPS_FOR[$G]}} | tail -2 || { echo "TRAIN FAILED $NAME"; continue; }
  echo "train_seconds=$(( $(date +%s) - start ))"
  "$PY" -m scripts.export_model --checkpoint outputs/$NAME/step-002000.pt --output "$PROJECT/outputs/$NAME.onnx" || { echo "EXPORT FAILED $NAME"; continue; }
  (cd "$PROJECT" && "$PY" test_submission.py outputs/$NAME.onnx > outputs/$NAME.test_submission.txt 2>&1 \
      && echo "test_submission: all passed" || echo "test_submission: FAILED")
done

cd "$PROJECT"
for G in none multiscale log_returns volatility momentum scale all; do
  [[ -f outputs/fin-$G.onnx ]] && "$PY" evaluate.py outputs/fin-$G.onnx outputs/proxy_suite --out outputs/eval/fin-$G.json || echo "EVAL FAILED fin-$G"
done
"$PY" - <<'PY'
import json
from pathlib import Path
names = [t["name"] for t in json.loads(Path("outputs/proxy_suite/tasks.json").read_text())["tasks"]]
lines = ["| model | score | mean AUC | " + " | ".join(names) + " |", "|---" * (len(names) + 3) + "|"]
for f in ["baseline", "baseline-s1", "baseline-s2"] + [f"fin-{g}" for g in
          "none multiscale log_returns volatility momentum scale all".split()]:
    if not Path(f"outputs/eval/{f}.json").exists(): continue
    r = json.loads(Path(f"outputs/eval/{f}.json").read_text())
    a = [t["auc"] for t in r["tasks"].values()]
    lines.append(f"| {f} | {r['score']:.3f} | {sum(a)/len(a):.3f} | " + " | ".join(f"{x:.3f}" for x in a) + " |")
Path("outputs/eval/FINANCE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
