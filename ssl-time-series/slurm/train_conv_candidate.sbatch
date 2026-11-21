#!/bin/bash
# Train + export + check + score one Track 3 conv candidate (research/combined), on GPU.
#   NAME=c-conv-mm SEED=0 EXTRA="--feature-groups momentum multiscale" sbatch slurm/train_conv_candidate.sbatch
# Recipe = Track 3 recommendation: dilated conv, depth 8, 40% masking + crop-view contrast, real-only
# pooling, 30k steps at batch 512; exported with mean+std pooling and the untrained features.
# DATA: project dir holding corpus/, the .venv, outputs/proxy_suite and outputs/ucr/{first,held_out}.
#SBATCH --job-name=ssl-conv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
CODE="${SLURM_SUBMIT_DIR:?submit from the research/combined project dir}"
DATA="${DATA:-$CODE}"
PY="$DATA/.venv/bin/python"; OUT="$DATA/outputs"
NAME="${NAME:?set NAME}"; SEED="${SEED:-0}"; STEPS="${STEPS:-30000}"
cd "$CODE"
echo "$NAME seed=$SEED steps=$STEPS extra=${EXTRA:-} commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
start=$(date +%s)
"$PY" -m scripts.train_mae --corpus "$DATA/corpus" --output "$OUT/$NAME.pt" \
    --arch conv --depth 8 --real-only --contrast 1.0 --view crop \
    --steps "$STEPS" --workers "${SLURM_CPUS_PER_TASK:-8}" --log-every 2000 --seed "$SEED" ${EXTRA:-}
echo "train_seconds=$(( $(date +%s) - start ))"
"$PY" -m scripts.export_model --checkpoint "$OUT/$NAME.pt" --output "$OUT/$NAME.onnx" --pooling mean_std --fixed
"$PY" test_submission.py "$OUT/$NAME.onnx" > "$OUT/$NAME.test_submission.txt" 2>&1 \
    && echo "test_submission: all passed" || { echo "test_submission: FAILED"; tail -20 "$OUT/$NAME.test_submission.txt"; }
"$PY" evaluate.py "$OUT/$NAME.onnx" "$OUT/proxy_suite" --out "$OUT/eval/$NAME.json" | tail -1
for SET in first held_out; do
  "$PY" evaluate.py "$OUT/$NAME.onnx" "$OUT/ucr/$SET" --out "$OUT/eval/ucr-$SET/$NAME.json" | tail -1
done
