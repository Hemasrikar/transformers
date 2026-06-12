#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/expanding_%j.out
#SBATCH --error=logs/expanding_%j.err

module purge
module load GCC/13.3.0 CUDA/13.0.0

mkdir -p logs
uv run python nonlinear_transformer_expanding.py
