#!/bin/bash
# Submit the three Track 4 variants (seed 0). Run from ssl-time-series/ on research/contrastive-ssl:
#   mkdir -p logs && DATA=<team project dir> bash slurm/contrastive_variants.sh
# Then: DATA=<same> .venv/bin/python -m scripts.compare_eval   (or $DATA/.venv/bin/python ...)
set -euo pipefail
: "${DATA:?set DATA to the project dir with corpus/, .venv and outputs/}"
export DATA
NAME=ct-infonce EXTRA="--reconstruction 0" \
  sbatch slurm/train_contrastive.sbatch
NAME=ct-aug     EXTRA="--reconstruction 0 --jitter 0.05 --warp 0.2 --crop-low 0.5" \
  sbatch slurm/train_contrastive.sbatch
NAME=ct-ts2vec  EXTRA="--reconstruction 0 --objective hierarchical --crop-low 0.5" \
  sbatch slurm/train_contrastive.sbatch
