#!/bin/bash
# Submit the three Track 4 variants plus a reference (the Track 3 recipe: reconstruction + InfoNCE), seed 0.
# Run from ssl-time-series/ on research/contrastive-ssl, after slurm/setup_track4.sbatch has finished:
#   mkdir -p logs && bash slurm/contrastive_variants.sh
# DATA defaults to this checkout (corpus/, .venv, outputs/). Then: .venv/bin/python -m scripts.compare_eval
set -euo pipefail
export DATA="${DATA:-$PWD}"
NAME=ct-ref     EXTRA="" \
  sbatch slurm/train_contrastive.sbatch
NAME=ct-infonce EXTRA="--reconstruction 0" \
  sbatch slurm/train_contrastive.sbatch
NAME=ct-aug     EXTRA="--reconstruction 0 --jitter 0.05 --warp 0.2 --crop-low 0.5" \
  sbatch slurm/train_contrastive.sbatch
NAME=ct-ts2vec  EXTRA="--reconstruction 0 --objective hierarchical --crop-low 0.5" \
  sbatch slurm/train_contrastive.sbatch
