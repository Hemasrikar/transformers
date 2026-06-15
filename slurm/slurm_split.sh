#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/static_%j.out
#SBATCH --error=logs/static_%j.err

module --force purge

cd /springbrook/share/wbs/bstvvz

# activate venv
source .venv/bin/activate

mkdir -p logs
python src/nonlinear_transformer.py
