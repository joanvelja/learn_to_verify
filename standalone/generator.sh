#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=data_gen_3b_mono
#SBATCH --output=spawn_testing/output/logs/data_gen_3b_mono.%j.out
#SBATCH --error=spawn_testing/output/errors/data_gen_3b_mono.%j.err

# Exit immediately if a command exits with a non-zero status.
set -e
# Get current path
CURRENT_PATH=$(pwd)
cd /home/jvelja/learn_to_verify

# sbatch train_provers.sh

# <--- Load Modules --->
module load 2023
module load CUDA/12.4.0 # Why not load 2024? 2024 requires CUDA 12.6.0 (A hassle to remake the whole environment)
module load py # aliased to Python/3.11.3-GCCcore-12.3.0

cd /home/jvelja/learn_to_verify

source .venv/bin/activate
echo "Activated virtual environment with uv"

# module load 2023
module load CUDA/12.4.0
# module load py # aliased to Python/3.11.3-GCCcore-12.3.0
# echo "Loaded CUDA 12.4.0"
# uv pip install rich
# --- Configuration ---
# Get Parent directory
PARENT_DIR=$(dirname "$CURRENT_PATH")


# Paths to models/IDs on Hugging Face Hub or local paths
LOCAL_MODELS_DIR="/home/jvelja/local_models"
SNEAKY_PROVER_PATH="Qwen/Qwen2.5-3B-Instruct"

# Fetch models from local paths if they exist. If they do not, echo an error message and exit.
if [ ! -d "${LOCAL_MODELS_DIR}/${SNEAKY_PROVER_PATH}" ]; then
    echo "Error: Local model ${SNEAKY_PROVER_PATH} not found in ${LOCAL_MODELS_DIR}"
    exit 1
fi

SNEAKY_PROVER_PATH="${LOCAL_MODELS_DIR}/${SNEAKY_PROVER_PATH}"
echo "Using local model ${SNEAKY_PROVER_PATH}"


VLLM_WORKER_MULTIPROC_METHOD=spawn # Seems the only way to avoid vllm new engine error
export TORCH_NCCL_TRACE_BUFFER_SIZE=2097152
export VLLM_USE_V1=0

# vLLM Server Ports
VLLM_PORT_SNEAKY=8000

# vLLM Server Host (use 127.0.0.1 for local communication on a single node)
VLLM_HOST="127.0.0.1"

# GPU Allocation (N=1, G=8 example)
VLLM_SNEAKY_GPUS="0,1"      # Sneaky Prover vLLM on GPU 0

# Other vLLM args - TODO: Would be nice to automate these with a script
VLLM_DTYPE="bfloat16"
# Memory utilization needs careful tuning for co-located servers on GPU 1
VLLM_GPU_MEM_UTIL_SNEAKY=0.94


# --- Cleanup Function ---
cleanup() {
    echo "Caught signal. Cleaning up background processes..."
    # Kill all background processes started by this script
    # Using pkill -P $$ might be too broad if other unrelated background tasks exist.
    # Killing specific PIDs is safer if possible.
    if [[ -n $VLLM_PID_SNEAKY ]]; then kill $VLLM_PID_SNEAKY || echo "Sneaky Server already stopped."; fi
    # Add kills for training PIDs if needed, though accelerate might handle its children.
    echo "Cleanup finished."
    exit 1
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM

# --- Launch vLLM Servers ---
# Calculate tensor parallel size for each server based on assigned GPUs
NUM_GPUS_PER_SERVER_SNEAKY=$(echo $VLLM_SNEAKY_GPUS | awk -F',' '{print NF}')

cd /home/jvelja/learn_to_verify/pvg

CURRENT_PATH=$(pwd)
VLLM_SERVE_SCRIPT="${CURRENT_PATH}/inference/vllm_serve.py"

echo "Starting vLLM Server for Sneaky Prover on GPU(s) ${VLLM_SNEAKY_GPUS} (Port: ${VLLM_PORT_SNEAKY})..."
CUDA_VISIBLE_DEVICES=$VLLM_SNEAKY_GPUS python $VLLM_SERVE_SCRIPT \
    --model $SNEAKY_PROVER_PATH \
    --port $VLLM_PORT_SNEAKY \
    --host 0.0.0.0 \
    --dtype $VLLM_DTYPE \
    --gpu-memory-utilization $VLLM_GPU_MEM_UTIL_SNEAKY \
    --tensor-parallel-size $NUM_GPUS_PER_SERVER_SNEAKY \
    --max_model_len 5120 \
    --enable-prefix-caching True \
    & # Run in background
VLLM_PID_SNEAKY=$!
echo "Sneaky Prover vLLM Server PID: $VLLM_PID_SNEAKY"

echo "Waiting ~10 seconds for vLLM servers to initialize..."
sleep 10

cd /home/jvelja/learn_to_verify

# Run full generation pipeline
uv run --env-file /home/jvelja/learn_to_verify/pvg/.env python generator.py \
  --dataset "jvelja/apps_checkable_filtered-verifier-regressor" \
  --output results_3b_mono_disabled_backdoor \
  --sneaky-port $VLLM_PORT_SNEAKY \
  --split "all" \
  --disable-backdoor-verification \
  --verbose


# # --- Wait for processes ---
echo "Launch complete. Waiting for training process (and servers) to finish..."
wait

# If training finishes normally, kill the servers
echo "Training finished or script interrupted. Killing vLLM servers..."
kill $VLLM_PID_SNEAKY || echo "Sneaky Server already stopped."

echo "Script finished."