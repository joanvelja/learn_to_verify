#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=upgrader
#SBATCH --output=upgrader/output/logs/upgrader.%j.out
#SBATCH --error=upgrader/output/errors/upgrader.%j.err

# Exit immediately if a command exits with a non-zero status.
set -e
# Get current path
CURRENT_PATH=$(pwd)
cd /home/jvelja/learn_to_verify

# <--- Load Modules --->
module load 2024
module load CUDA/12.6.0 # Why not load 2024? 2024 requires CUDA 12.6.0 (A hassle to remake the whole environment)
module load Python/3.12.3-GCCcore-13.3.0 # aliased to Python/3.11.3-GCCcore-12.3.0

cd /home/jvelja/learn_to_verify

# Cleanup
rm -rf .venv
uv cache clean

# cd ..
# echo "Installing flash-attention-3"
# # git clone https://github.com/Dao-AILab/flash-attention.git
# cd flash-attention/hopper
# python setup.py install

# export PYTHONPATH=$PWD
# echo "Running pytest for flash-attention-3"
# pytest -q -s test_flash_attn.py
