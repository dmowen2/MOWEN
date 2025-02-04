#!/bin/bash
#SBATCH --job-name=mowen-classification
#SBATCH --output=/work/zzv393/MOWEN/logs/mowen_classification_output_%j_%A.txt
#SBATCH --error=/work/zzv393/MOWEN/logs/mowen_classification_error_%j_%A.txt
#SBATCH --partition=gpu4v100  # Use gpu4v100 partition
#SBATCH --gres=gpu:4  # Request 4 GPUs
#SBATCH --ntasks=4  # Match number of GPUs
#SBATCH --cpus-per-task=8  # Allocate 8 CPU cores per task
#SBATCH --time=72:00:00
#SBATCH --exclusive  # Ensure full node access

# Load required modules
module load anaconda3
module load cuda12/nvcc/12.5.82
module load cuda12/cudnn/9.2.0.82

# Correct Conda activation
source $(conda info --base)/etc/profile.d/conda.sh  # Ensure Conda is properly initialized
conda activate /work/zzv393/mowen-env

# Ensure dependencies are installed
pip list | grep torch || pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip list | grep timm || pip install timm

# Set a dynamic MASTER_PORT to avoid conflicts
export MASTER_PORT=$((29500 + RANDOM % 100))

# Ensure logs directory exists
mkdir -p /work/zzv393/MOWEN/logs

# Run distributed training
srun torchrun --nproc_per_node=4 --nnodes=1 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    /work/zzv393/MOWEN/mowen/classification_pretrain.py

