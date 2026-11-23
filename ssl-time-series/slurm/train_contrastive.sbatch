#!/bin/bash
# Track 4: train one contrastive conv variant, export it two ways, check it and score it, on GPU.
#   NAME=ct-infonce EXTRA="--reconstruction 0" DATA=<team project dir> sbatch slurm/train_contrastive.sbatch
# Same encoder, steps, batch and data as slurm/train_conv_candidate.sbatch (the Track 3 recipe behind the 1.187
# leaderboard model); only the objective flags in EXTRA differ. Two exports:
#   $NAME-enc         encoder only, mean+std pooling  (compare with c-conv-s1-nofixed: isolates the objective)
#   $NAME-stats-prof  + window statistics + position profile (compare with conv-stats-prof = leaderboard 1.187;
#                     this is the file to upload)
# DATA: project dir holding corpus/, the .venv, outputs/proxy_suite and outputs/ucr/{first,held_out}.
#SBATCH --job-name=ssl-contrast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
CODE="${SLURM_SUBMIT_DIR:?submit from the research/contrastive-ssl project dir}"
DATA="${DATA:-$CODE}"
PY="$DATA/.venv/bin/python"; OUT="$DATA/outputs"
NAME="${NAME:?set NAME}"; SEED="${SEED:-0}"; STEPS="${STEPS:-30000}"
SUBMIT_FLAGS="${SUBMIT_FLAGS:---pooling mean_std --stats --profile}"
cd "$CODE"; mkdir -p "$OUT/eval/ucr-first" "$OUT/eval/ucr-held_out"
echo "$NAME seed=$SEED steps=$STEPS extra=${EXTRA:-} submit_flags=$SUBMIT_FLAGS commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
start=$(date +%s)
"$PY" -m scripts.train_mae --corpus "$DATA/corpus" --output "$OUT/$NAME.pt" \
    --arch conv --depth 8 --real-only --contrast 1.0 --view crop \
    --steps "$STEPS" --workers "${SLURM_CPUS_PER_TASK:-8}" --log-every 2000 --seed "$SEED" ${EXTRA:-}
echo "train_seconds=$(( $(date +%s) - start ))"
export_one() {  # NAME then export_model flags
  local N=$1; shift
  "$PY" -m scripts.export_model --checkpoint "$OUT/$NAME.pt" --output "$OUT/$N.onnx" "$@" 2>&1 | grep -v -i warn | tail -1
  "$PY" test_submission.py "$OUT/$N.onnx" > "$OUT/$N.test_submission.txt" 2>&1 \
      && echo "$N test_submission: all passed" || { echo "$N test_submission: FAILED"; tail -20 "$OUT/$N.test_submission.txt"; }
  "$PY" evaluate.py "$OUT/$N.onnx" "$OUT/proxy_suite" --out "$OUT/eval/$N.json" 2>/dev/null | tail -1
  for SET in first held_out; do
    "$PY" evaluate.py "$OUT/$N.onnx" "$OUT/ucr/$SET" --out "$OUT/eval/ucr-$SET/$N.json" 2>/dev/null | tail -1
  done
}
export_one "$NAME-enc" --pooling mean_std
export_one "$NAME-stats-prof" $SUBMIT_FLAGS
