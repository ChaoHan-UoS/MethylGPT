#!/bin/bash
#SBATCH --job-name=pretrain_methylgpt
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=125G
#SBATCH --cpus-per-task=12
#SBATCH --time=2-23:59:00

export PYTHONUNBUFFERED=1  # Unbuffer Python output
export WANDB_API_KEY=48436e46ea90de96edea92a6eea1c37e60083e4b

nvidia-smi
hostname  # Print compute node hostname
pwd

echo ""
echo "Begin"

# Run the code from the task-specific dir
cd "${TASK_DIR}/tutorials/pretraining"
pwd

echo "Running task ${SLURM_JOB_ID}"

# Activate Poetry venv
VENV_PATH=$(poetry env info --path)
source "${VENV_PATH}/bin/activate"

bash run_pretraining.sh

echo "Task ${SLURM_JOB_ID} done"