#!/bin/bash
#SBATCH --job-name=pgx_molecular
#SBATCH --account=b181-humgen-ag
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/molecular_graph_models/%A_%a.out
#SBATCH --error=logs/molecular_graph_models/%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PROJECT="$HOME/MLPKGA007_MSC_clean"
CONTAINER="/software/astro/containers/pytorch/ASTRO-GPU-PyTorch-2026-01-28.sif"

MODELS=(gcn rgcn rgat)
PROTOCOLS=(dg_context strict_cold)
SEEDS=(42 123 2026)

task=${SLURM_ARRAY_TASK_ID}

model_index=$(( task / 6 ))
remainder=$(( task % 6 ))
protocol_index=$(( remainder / 3 ))
seed_index=$(( remainder % 3 ))

MODEL=${MODELS[$model_index]}
PROTOCOL=${PROTOCOLS[$protocol_index]}
SEED=${SEEDS[$seed_index]}

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Feature mode: molecular"
echo "Protocol: ${PROTOCOL}"
echo "Seed: ${SEED}"
echo "Node: $(hostname)"
echo "========================================"

singularity exec --nv \
  --pwd "$PROJECT" \
  "$CONTAINER" \
  bash -lc "
    set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

    source '$PROJECT/.venv-pyg-p100/bin/activate'
    export PYTHONPATH='$PROJECT/src'

    python - <<'PY'
import torch
import torch_geometric

print('PyTorch:', torch.__version__)
print('PyG:', torch_geometric.__version__)
print('CUDA available:', torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError('GPU was requested but CUDA is unavailable.')

print('GPU:', torch.cuda.get_device_name(0))
PY

    python scripts/train/train_graph_model.py \
      --model '$MODEL' \
      --protocol '$PROTOCOL' \
      --seed '$SEED' \
      --feature-mode molecular \
      --tensor-root artifacts/tensors_molecular \
      --output-root results/models \
      --device cuda
  "

echo "Completed: ${MODEL} ${PROTOCOL} seed=${SEED}"
