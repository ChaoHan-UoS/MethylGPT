#!/bin/bash
#SBATCH --job-name=pretrain_methylgpt
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=125G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-23:59:00

export PYTHONUNBUFFERED=1  # Unbuffer Python output
export WANDB_API_KEY=48436e46ea90de96edea92a6eea1c37e60083e4b

module load apps/python/3.11.3

nvidia-smi
hostname  # Print compute node hostname
pwd

echo "=== Running task ${SLURM_JOB_ID} ==="

cd "${TASK_DIR}/tutorials/pretraining"
pwd

# Activate Poetry venv
POETRY_BIN="$HOME/.local/bin/poetry"
VENV_PATH=$(env -u PYTHONPATH "$POETRY_BIN" env info --path)
echo "Poetry venv: $VENV_PATH"
source "${VENV_PATH}/bin/activate"

echo ""
bash run_pretraining.sh
echo ""

echo "=== Task ${SLURM_JOB_ID} done ==="