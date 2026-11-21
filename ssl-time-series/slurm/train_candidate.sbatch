#!/bin/bash
# Train + export + validate one candidate. Configure with env vars:
#   NAME=baseline-s1 CONFIG=configs/baseline.json SEED=1 sbatch slurm/train_candidate.sbatch
#   NAME=main CONFIG=configs/main.json sbatch slurm/train_candidate.sbatch   # 20k steps ~25 min on an A5000
# Run `mkdir -p logs` once first: Slurm drops the log silently if the folder is missing.
# EXTRA passes more train_encoder flags (e.g. EXTRA="--steps 500").
#SBATCH --job-name=ssl-cand
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${NAME:?set NAME}" "${CONFIG:?set CONFIG}"
SEED="${SEED:-0}"; EXTRA="${EXTRA:-}"
PROJECT="${SLURM_SUBMIT_DIR:?submit from ssl-time-series/ssl-time-series}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/.uv-cache}"  # keep multi-GB wheels out of $HOME
cd "$PROJECT"
echo "host=$(hostname) job=$SLURM_JOB_ID name=$NAME config=$CONFIG seed=$SEED extra=$EXTRA commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
START=$(date +%s)
# shellcheck disable=SC2086
uv run --no-sync python -m scripts.train_encoder --corpus corpus --config "$CONFIG" \
  --workers 8 --seed "$SEED" --output "outputs/$NAME.pt" $EXTRA 2>&1 | grep -v UserWarning
TRAIN_S=$(( $(date +%s) - START ))
uv run --no-sync python -m scripts.export_model --checkpoint "outputs/$NAME.pt" --output "outputs/$NAME.onnx"
uv run --no-sync test_submission.py "outputs/$NAME.onnx" | tee "outputs/$NAME.test_submission.txt"
uv run --no-sync python - "$NAME" "$CONFIG" "$SEED" "$TRAIN_S" "$EXTRA" <<'PY'
import json, subprocess, sys, torch
from pathlib import Path
name, config, seed, train_s, extra = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ckpt = torch.load(f"outputs/{name}.pt", map_location="cpu", weights_only=False)
summary = {
    "name": name, "config": config, "seed": seed, "extra_flags": extra, "workers": 8,
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "train_seconds": train_s,
    "parameters": sum(v.numel() for v in ckpt["encoder"].values()),
    "onnx_bytes": Path(f"outputs/{name}.onnx").stat().st_size,
    "final_losses": ckpt["history"][-1],
    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
}
Path(f"outputs/{name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
