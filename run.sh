#!/bin/bash
#SBATCH --job-name=learning_verifier_prior_newPrompt    
#SBATCH --output=logs/learning_verifier_prior_newPrompt-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=single
#SBATCH --time=06:30:00

cd /data/joan_velja/learn_to_verify/verifiers
source .venv/bin/activate

uv run accelerate launch --config-file configs/zero3_mini.yaml --num-processes 3 train_verifier.py