#!/bin/bash
# Build the next leaderboard probes (research/combined), on GPU:
#   1. main-mm: configs/main.json (20k steps, width 320) + Track 2 momentum + multiscale.
#   2. Track 3 conv (c-conv-s1.pt) exported WITHOUT the untrained features, to isolate what they do.
# Each gets test_submission.py and scores on our suite + both UCR sets.
#   DATA=<project with corpus/, .venv, outputs/> sbatch slurm/leaderboard_probes.sbatch
#SBATCH --job-name=ssl-probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
CODE="${SLURM_SUBMIT_DIR:?submit from the research/combined project dir}"
DATA="${DATA:-$CODE}"; PY="$DATA/.venv/bin/python"; OUT="$DATA/outputs"
cd "$CODE"; echo "commit=$(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name --format=csv,noheader

start=$(date +%s)
"$PY" -m scripts.train_encoder --corpus "$DATA/corpus" --config configs/main.json --workers 8 \
    --checkpoint-every 0 --output "$OUT/main-mm.pt" --feature-groups momentum multiscale | tail -2
echo "train_seconds=$(( $(date +%s) - start ))"
"$PY" -m scripts.export_model --checkpoint "$OUT/main-mm/step-020000.pt" --output "$OUT/main-mm.onnx"
"$PY" -m scripts.export_model --checkpoint "$OUT/c-conv-s1.pt" --output "$OUT/c-conv-s1-nofixed.onnx" --pooling mean_std

for NAME in main-mm c-conv-s1-nofixed; do
  "$PY" test_submission.py "$OUT/$NAME.onnx" > "$OUT/$NAME.test_submission.txt" 2>&1 \
      && echo "$NAME test_submission: all passed" || echo "$NAME test_submission: FAILED"
  "$PY" evaluate.py "$OUT/$NAME.onnx" "$OUT/proxy_suite" --out "$OUT/eval/$NAME.json" | tail -1
  for SET in first held_out; do
    "$PY" evaluate.py "$OUT/$NAME.onnx" "$OUT/ucr/$SET" --out "$OUT/eval/ucr-$SET/$NAME.json" | tail -1
  done
done
