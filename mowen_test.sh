#!/bin/bash
#SBATCH --job-name=mowen-pretrain
#SBATCH --output=/work/zzv393/MOWEN/logs/mowen_output_%j_%A.txt
#SBATCH --error=/work/zzv393/MOWEN/logs/mowen_error_%j_%A.txt
#SBATCH --partition=gpu2v100
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --exclusive

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
srun torchrun --nproc_per_node=2 --nnodes=1 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    /work/zzv393/MOWEN/mowen/pretrain.py
