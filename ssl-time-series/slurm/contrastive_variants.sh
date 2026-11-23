#!/bin/bash
# Submit Track 4 runs (seed 0). Run from ssl-time-series/ on research/contrastive-ssl, after setup_track4.sbatch:
#   mkdir -p logs && bash slurm/contrastive_variants.sh                      # all four
#   bash slurm/contrastive_variants.sh ct-infonce ct-aug ct-ts2vec           # only these
# DATA defaults to this checkout (corpus/, .venv, outputs/). Then: .venv/bin/python -m scripts.compare_eval
set -euo pipefail
export DATA="${DATA:-$PWD}"
[[ $# -eq 0 ]] && set -- ct-ref ct-infonce ct-aug ct-ts2vec
for NAME in "$@"; do
  case "$NAME" in
    ct-ref)     EXTRA="" ;;                                                  # Track 3 recipe: recon + InfoNCE
    ct-infonce) EXTRA="--reconstruction 0" ;;
    ct-aug)     EXTRA="--reconstruction 0 --jitter 0.05 --warp 0.2 --crop-low 0.5" ;;
    ct-ts2vec)  EXTRA="--reconstruction 0 --objective hierarchical --crop-low 0.5" ;;
    *) echo "unknown variant $NAME (choose: ct-ref ct-infonce ct-aug ct-ts2vec)"; exit 1 ;;
  esac
  echo -n "$NAME: "; NAME="$NAME" EXTRA="$EXTRA" sbatch slurm/train_contrastive.sbatch
done
