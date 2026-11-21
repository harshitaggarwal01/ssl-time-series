#!/bin/bash
# Stage 1 (CPU): env install || corpus download, then unzip + profile. No GPU needed.
#SBATCH --job-name=ssl-prep
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from ssl-time-series/ssl-time-series}"
CORPUS_DIR="$PROJECT/corpus"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.uv-cache}"  # keep multi-GB wheels out of $HOME
cd "$PROJECT"; mkdir -p outputs
echo "host=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)"

# uv sync (torch CUDA wheels, multi-GB) runs concurrently with the 8.9 GB download.
uv sync > "outputs/uv_sync.log" 2>&1 & SYNC_PID=$!

if [[ ! -f "$CORPUS_DIR/corpus.npz" ]]; then
  STAGE="${TMPDIR:-/tmp}"
  (( $(df --output=avail -k "$STAGE" | tail -1) < 12000000 )) && STAGE="$PROJECT"
  ZIP="$STAGE/corpus-$SLURM_JOB_ID.zip"; trap 'rm -f "$ZIP"' EXIT
  time uvx gdown "https://drive.google.com/uc?id=1ld4mifcgwaBZin7cGBNCYm7xXH2T90U6" -O "$ZIP"
  mkdir -p "$CORPUS_DIR"; time unzip -q "$ZIP" -d "$CORPUS_DIR"; rm -f "$ZIP"
fi
ls -lh "$CORPUS_DIR/corpus.npz"

wait "$SYNC_PID" || { cat outputs/uv_sync.log; exit 1; }
tail -3 outputs/uv_sync.log
time uv run python -m scripts.profile_corpus --corpus corpus --out outputs/corpus_profile.json
