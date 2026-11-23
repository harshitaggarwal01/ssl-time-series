#!/bin/bash
# Track 4: re-export trained checkpoints with segment pooling (keeps *where* in the window things happen), then
# check and score them. No training. Run after the checkpoints exist:
#   sbatch slurm/export_segments.sbatch                          # ct-ts2vec and ct-ref (default)
#   NAMES="ct-ts2vec ct-ref ct-aug ct-infonce" sbatch slurm/export_segments.sbatch
# Writes outputs/<name>-seg2-stats-prof.onnx and <name>-seg4-stats-prof.onnx (+ eval JSONs).
# Compare: .venv/bin/python -m scripts.compare_eval ct-ts2vec-seg2-stats-prof ct-ts2vec-seg4-stats-prof ...
#SBATCH --job-name=ssl-seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
CODE="${SLURM_SUBMIT_DIR:?submit from the research/contrastive-ssl project dir}"
DATA="${DATA:-$CODE}"; PY="$DATA/.venv/bin/python"; O="$DATA/outputs"
NAMES="${NAMES:-ct-ts2vec ct-ref}"
cd "$CODE"; mkdir -p "$O/eval/ucr-first" "$O/eval/ucr-held_out"
echo "names=$NAMES commit=$(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name --format=csv,noheader
export_one() {  # output name, checkpoint, then export_model flags
  local N=$1 C=$2; shift 2
  [[ -f "$C" ]] || { echo "MISSING $C"; return; }
  "$PY" -m scripts.export_model --checkpoint "$C" --output "$O/$N.onnx" "$@" 2>&1 | grep -v -i warn | tail -1
  "$PY" test_submission.py "$O/$N.onnx" > "$O/$N.test_submission.txt" 2>&1 \
      && echo "$N test_submission: all passed" || { echo "$N test_submission: FAILED"; tail -20 "$O/$N.test_submission.txt"; }
  "$PY" evaluate.py "$O/$N.onnx" "$O/proxy_suite" --out "$O/eval/$N.json" 2>/dev/null | tail -1
  for SET in first held_out; do
    "$PY" evaluate.py "$O/$N.onnx" "$O/ucr/$SET" --out "$O/eval/ucr-$SET/$N.json" 2>/dev/null | tail -1
  done
}
for NAME in $NAMES; do
  for SEG in 2 4; do
    export_one "$NAME-seg$SEG-stats-prof" "$O/$NAME.pt" --pooling "seg${SEG}_std" --stats --profile
  done
done
