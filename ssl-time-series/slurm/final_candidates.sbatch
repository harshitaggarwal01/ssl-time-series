#!/bin/bash
# Final-night candidates (23 Nov): retrain Track 4's ct-aug (contrast only + jitter/warp/wide crops, seed 0) and
# combine it with the leaderboard model's encoder (c-conv-s1, recon + InfoNCE; conv-stats-patch = 1.197).
#   DATA=<project with corpus/, .venv, outputs/> sbatch slurm/final_candidates.sbatch   (from research/final)
#SBATCH --job-name=ssl-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
CODE="${SLURM_SUBMIT_DIR:?submit from the research/final project dir}"
DATA="${DATA:-$CODE}"; PY="$DATA/.venv/bin/python"; O="$DATA/outputs"
cd "$CODE"; echo "commit=$(git rev-parse --short HEAD)"; nvidia-smi --query-gpu=name --format=csv,noheader
"$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU"; exit 1; }
# The project corpus was removed from scratch on 22 Nov; fetch it again to node-local disk when it is missing.
CORPUS="$DATA/corpus"
if [[ ! -f "$CORPUS/corpus.npz" ]]; then
  STAGE="${TMPDIR:-/tmp}"
  (( $(df --output=avail -k "$STAGE" | tail -1) < 20000000 )) && STAGE="$DATA"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/oscar/scratch/haggarw3/.cache/uv}"
  ZIP="$STAGE/corpus-$SLURM_JOB_ID.zip"; trap 'rm -f "$ZIP"' EXIT
  time uvx gdown "https://drive.google.com/uc?id=1ld4mifcgwaBZin7cGBNCYm7xXH2T90U6" -O "$ZIP"
  CORPUS="$STAGE/corpus-$SLURM_JOB_ID"; mkdir -p "$CORPUS"; time unzip -q "$ZIP" -d "$CORPUS"; rm -f "$ZIP"
fi
ls -lhL "$CORPUS/corpus.npz"
start=$(date +%s)
"$PY" -m scripts.train_mae --corpus "$CORPUS" --output "$O/ct-aug.pt" \
    --arch conv --depth 8 --real-only --contrast 1.0 --view crop --steps 30000 --workers 8 --log-every 5000 --seed 0 \
    --reconstruction 0 --jitter 0.05 --warp 0.2 --crop-low 0.5 2>&1 | grep -v -i warn | tail -3
echo "train_seconds=$(( $(date +%s) - start ))"
export_one() {  # NAME then export_model flags
  local N=$1; shift
  "$PY" -m scripts.export_model --output "$O/$N.onnx" "$@" 2>&1 | grep -v -i warn | tail -1
  "$PY" test_submission.py "$O/$N.onnx" > "$O/$N.test_submission.txt" 2>&1 \
      && echo "$N test_submission: all passed" || { echo "$N test_submission: FAILED"; tail -20 "$O/$N.test_submission.txt"; }
  "$PY" evaluate.py "$O/$N.onnx" "$O/proxy_suite" --out "$O/eval/$N.json" > /dev/null 2>&1
  for SET in first held_out; do
    "$PY" evaluate.py "$O/$N.onnx" "$O/ucr/$SET" --out "$O/eval/ucr-$SET/$N.json" > /dev/null 2>&1
  done
}
OURS=$O/c-conv-s1.pt; AUG=$O/ct-aug.pt
export_one ct-aug-enc         --checkpoint $AUG --pooling mean_std
export_one ct-aug-stats-prof  --checkpoint $AUG --pooling mean_std --stats --profile
export_one ct-aug-stats-patch --checkpoint $AUG --pooling mean_std --stats --patch-profile 32
export_one final-mean         --checkpoint $OURS $AUG --pooling mean_std mean     --stats --patch-profile 32
export_one final-ms           --checkpoint $OURS $AUG --pooling mean_std mean_std --stats --patch-profile 32
DATA="$DATA" "$PY" -m scripts.compare_eval c-conv-s1-nofixed conv-stats-prof conv-stats-patch \
    ct-aug-enc ct-aug-stats-prof ct-aug-stats-patch final-mean final-ms
