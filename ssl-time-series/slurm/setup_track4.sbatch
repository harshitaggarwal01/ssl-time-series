#!/bin/bash
# One-time setup for Track 4 in your own checkout (CPU job): Python env, corpus, proxy suite, UCR sets.
#   mkdir -p logs && sbatch slurm/setup_track4.sbatch
#   CORPUS_SRC=/path/to/someone/elses/corpus sbatch slurm/setup_track4.sbatch   # link an existing copy instead
# Afterwards, train with:  bash slurm/contrastive_variants.sh   (DATA defaults to this checkout)
#SBATCH --job-name=ssl-setup
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from ssl-time-series/ (the folder with pyproject.toml)}"
cd "$PROJECT"; mkdir -p outputs
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.uv-cache}"  # keep multi-GB wheels out of $HOME
echo "host=$(hostname) commit=$(git rev-parse --short HEAD)"

uv sync > outputs/uv_sync.log 2>&1 & SYNC_PID=$!
if [[ -n "${CORPUS_SRC:-}" && ! -e corpus ]]; then
  ln -s "$CORPUS_SRC" corpus
elif [[ ! -f corpus/corpus.npz ]]; then
  STAGE="${TMPDIR:-/tmp}"
  (( $(df --output=avail -k "$STAGE" | tail -1) < 12000000 )) && STAGE="$PROJECT"
  ZIP="$STAGE/corpus-$SLURM_JOB_ID.zip"; trap 'rm -f "$ZIP"' EXIT
  time uvx gdown "https://drive.google.com/uc?id=1ld4mifcgwaBZin7cGBNCYm7xXH2T90U6" -O "$ZIP"
  mkdir -p corpus; time unzip -q "$ZIP" -d corpus; rm -f "$ZIP"
fi
ls -lhL corpus/corpus.npz
wait "$SYNC_PID" || { cat outputs/uv_sync.log; exit 1; }
PY=.venv/bin/python

[[ -f outputs/proxy_suite/tasks.json ]] || time "$PY" -m scripts.build_proxy_suite --out outputs/proxy_suite
[[ -f outputs/ucr/first/tasks.json ]] || "$PY" -m scripts.proxy_ucr --out outputs/ucr/first --cache outputs/ucr/raw
[[ -f outputs/ucr/held_out/tasks.json ]] || "$PY" -m scripts.proxy_ucr --out outputs/ucr/held_out --cache outputs/ucr/raw --held-out
ls outputs/proxy_suite outputs/ucr/first outputs/ucr/held_out | head -40
echo "SETUP DONE"
