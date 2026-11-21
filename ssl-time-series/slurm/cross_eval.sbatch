#!/bin/bash
# Cross-evaluate team candidates on both proxies: our synthetic suite v2.1 and Track 3's UCR sets.
#   sbatch slurm/cross_eval.sbatch        (submit from ssl-time-series/ssl-time-series)
# Needs the Track 3 worktree at ../../wt-masked (research/masked-ssl). Also exports `main` with
# Track 3's readout (mean+std pooling + untrained features) to test that combination.
#SBATCH --job-name=ssl-xeval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from ssl-time-series/ssl-time-series}"
MASKED="$PROJECT/../../wt-masked/ssl-time-series"
PY="$PROJECT/.venv/bin/python"          # shared env; cwd decides which ssl_time_series is imported
cd "$PROJECT"; mkdir -p outputs/eval outputs/ucr
nvidia-smi --query-gpu=name --format=csv,noheader

# 1. Candidates. Track 3's committed model, plus main/baseline with Track 3's hybrid readout.
cp "$MASKED/models/candidate-conv.onnx" outputs/t3-conv.onnx
for M in main baseline; do
  (cd "$MASKED" && "$PY" -m scripts.export_model --checkpoint "$PROJECT/outputs/$M.pt" \
      --output "$PROJECT/outputs/$M-hybrid.onnx" --pooling mean_std --fixed) || echo "EXPORT FAILED $M-hybrid"
done
for M in t3-conv main-hybrid baseline-hybrid; do
  [[ -f outputs/$M.onnx ]] && "$PY" test_submission.py outputs/$M.onnx 2>&1 | tee outputs/$M.test_submission.txt | tail -3 || echo "TEST FAILED $M"
done

# 2. UCR proxy sets (download on the compute node; evaluation only, never training).
for SET in first held_out; do
  FLAG=""; [[ $SET == held_out ]] && FLAG="--held-out"
  [[ -f outputs/ucr/$SET/tasks.json ]] || (cd "$MASKED" && "$PY" -m scripts.proxy_ucr \
      --out "$PROJECT/outputs/ucr/$SET" --cache "$PROJECT/outputs/ucr/raw" $FLAG) || echo "UCR BUILD FAILED $SET"
done

# 3. Score everything. Suite v2.1 is already calibrated (outputs/proxy_suite).
MODELS=(example_submission/model.onnx outputs/baseline.onnx outputs/main.onnx \
        outputs/t3-conv.onnx outputs/main-hybrid.onnx outputs/baseline-hybrid.onnx)
for MODEL in "${MODELS[@]}"; do
  [[ -f $MODEL ]] || { echo "MISSING $MODEL"; continue; }
  NAME=$(basename "$MODEL" .onnx); [[ $MODEL == example_submission/* ]] && NAME=example_submission-model
  for SET in suite first held_out; do
    DATA=outputs/ucr/$SET; [[ $SET == suite ]] && DATA=outputs/proxy_suite
    OUT=outputs/eval/$NAME.json; [[ $SET != suite ]] && OUT=outputs/eval/ucr-$SET/$NAME.json
    [[ -f $DATA/tasks.json ]] || continue
    mkdir -p "$(dirname "$OUT")"
    [[ -f $OUT ]] || "$PY" evaluate.py "$MODEL" "$DATA" --out "$OUT" || echo "EVAL FAILED $NAME $SET"
  done
done

# 4. One table: suite score + mean AUC, UCR mean AUC on both sets.
"$PY" - <<'PY'
import json
from pathlib import Path
def mean_auc(p):
    if not p.exists(): return None
    t = json.loads(p.read_text())["tasks"].values()
    return sum(x["auc"] for x in t) / len(t)
rows = []
for f in sorted(Path("outputs/eval").glob("*.json")):
    r = json.loads(f.read_text())
    rows.append((f.stem, r["score"], mean_auc(f),
                 mean_auc(Path("outputs/eval/ucr-first") / f.name),
                 mean_auc(Path("outputs/eval/ucr-held_out") / f.name)))
fmt = lambda v: "" if v is None else f"{v:.3f}"
lines = ["| model | suite score | suite mean AUC | UCR first AUC | UCR held-out AUC |", "|---|---|---|---|---|"]
for row in sorted(rows, key=lambda r: -(r[4] or 0)):
    lines.append("| " + " | ".join([row[0]] + [fmt(v) for v in row[1:]]) + " |")
Path("outputs/eval/CROSS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
