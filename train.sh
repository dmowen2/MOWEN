#!/bin/bash
#SBATCH --job-name=mowen-pretrain
#SBATCH --output=logs/mowen_output_%j.txt
#SBATCH --error=logs/mowen_error_%j.txt
#SBATCH --partition=gpu2v100  # Use the correct partition
#SBATCH --gres=gpu:2          # Request 2 V100 GPUs
#SBATCH --ntasks=2            # Number of processes
#SBATCH --cpus-per-task=8     # Allocate 8 CPU cores per GPU
#SBATCH --mem=64G             # 64GB RAM
#SBATCH --time=7-00:00:00     # Max time (7 days)

module load anaconda
source activate your_env_name  # Replace with your Anaconda environment

# Run distributed training on 2 GPUs
srun python -m torch.distributed.launch --nproc_per_node=2 mowen/pretrain.py
