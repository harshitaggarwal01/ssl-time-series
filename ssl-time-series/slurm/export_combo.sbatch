#!/bin/bash
# Combined candidates: conv + stats + position profile, mean_max_std pooling, spectrum. Export, check, score on GPU.
#   DATA=<project with .venv, outputs/> sbatch slurm/export_combo.sbatch   (from research/combined)
#SBATCH --job-name=ssl-combo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
CODE="${SLURM_SUBMIT_DIR:?submit from the research/combined project dir}"
DATA="${DATA:-$CODE}"; PY="$DATA/.venv/bin/python"; O="$DATA/outputs"
cd "$CODE"; echo "commit=$(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name --format=csv,noheader
export_one() {  # NAME then export_model flags
  local NAME=$1; shift
  "$PY" -m scripts.export_model --output "$O/$NAME.onnx" "$@" 2>&1 | grep -v -i warn | tail -1 || { echo "EXPORT FAILED $NAME"; return; }
  "$PY" test_submission.py "$O/$NAME.onnx" > "$O/$NAME.test_submission.txt" 2>&1 \
      && echo "$NAME test_submission: all passed" || echo "$NAME test_submission: FAILED"
  "$PY" evaluate.py "$O/$NAME.onnx" "$O/proxy_suite" --out "$O/eval/$NAME.json" 2>/dev/null | tail -1
  for SET in first held_out; do
    "$PY" evaluate.py "$O/$NAME.onnx" "$O/ucr/$SET" --out "$O/eval/ucr-$SET/$NAME.json" 2>/dev/null | tail -1
  done
}
C=$O/c-conv-s1.pt
export_one conv-stats-prof      --checkpoint $C --pooling mean_std     --stats --profile
export_one conv-max-stats       --checkpoint $C --pooling mean_max_std --stats
export_one conv-max-stats-prof  --checkpoint $C --pooling mean_max_std --stats --profile
export_one conv-spec-stats-prof --checkpoint $C --pooling mean_std     --fixed-parts spectrum scale --stats --profile
