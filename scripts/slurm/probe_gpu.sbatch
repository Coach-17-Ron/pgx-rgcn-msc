#!/bin/bash
#SBATCH --account=b181-humgen-ag
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gpu_probe/%x_%j.out
#SBATCH --error=logs/gpu_probe/%x_%j.err

set -euo pipefail

PROJECT="$HOME/MLPKGA007_MSC_clean"
CONTAINER="/software/astro/containers/pytorch/ASTRO-GPU-PyTorch-2026-01-28.sif"

echo "Node: $(hostname)"

nvidia-smi \
  --query-gpu=name,compute_cap,memory.total,driver_version \
  --format=csv

singularity exec --nv \
  --pwd "$PROJECT" \
  "$CONTAINER" \
  bash -lc "
    source '$PROJECT/.venv-pyg/bin/activate'

    python - <<'PY'
import torch

print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Capability:', torch.cuda.get_device_capability(0))

    major, minor = torch.cuda.get_device_capability(0)
    print('Compatible with current PyTorch:', major >= 7)
PY
  "
