#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --job-name=data_gen_7b
#SBATCH --output=spawn_testing/output/logs/data_gen_7b.%j.out
#SBATCH --error=spawn_testing/output/errors/data_gen_7b.%j.err

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

cd /home/jvelja/learn_to_verify/src/verifiers

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
HONEST_PROVER_PATH="Qwen/Qwen2.5-Coder-7B"
SNEAKY_PROVER_PATH="Qwen/Qwen2.5-Coder-7B"

# Fetch models from local paths if they exist (dir = local_models/{MODEL_PROVIDER e.g. Qwen}/MODEL_NAME e.g. Qwen2.5-Coder-3B-Instruct)
# If they do not, echo an error message and exit
if [ ! -d "${LOCAL_MODELS_DIR}/${HONEST_PROVER_PATH}" ]; then
    echo "Error: Local model ${HONEST_PROVER_PATH} not found in ${LOCAL_MODELS_DIR}"
    exit 1
fi
if [ ! -d "${LOCAL_MODELS_DIR}/${SNEAKY_PROVER_PATH}" ]; then
    echo "Error: Local model ${SNEAKY_PROVER_PATH} not found in ${LOCAL_MODELS_DIR}"
    exit 1
fi

HONEST_PROVER_PATH="${LOCAL_MODELS_DIR}/${HONEST_PROVER_PATH}"
echo "Using local model ${HONEST_PROVER_PATH}"
SNEAKY_PROVER_PATH="${LOCAL_MODELS_DIR}/${SNEAKY_PROVER_PATH}"
echo "Using local model ${SNEAKY_PROVER_PATH}"


VLLM_WORKER_MULTIPROC_METHOD=spawn # Seems the only way to avoid vllm new engine error
export TORCH_NCCL_TRACE_BUFFER_SIZE=2097152

# vLLM Server Ports
VLLM_PORT_HONEST=8000
VLLM_PORT_SNEAKY=8001

# vLLM Server Host (use 127.0.0.1 for local communication on a single node)
VLLM_HOST="127.0.0.1"

# GPU Allocation (N=1, G=8 example)
VLLM_SNEAKY_GPUS="0"      # Sneaky Prover vLLM on GPU 0
VLLM_HONEST_GPUS="1"      # Honest Prover vLLM on GPU 1

# Other vLLM args - TODO: Would be nice to automate these with a script
VLLM_DTYPE="auto"
# Memory utilization needs careful tuning for co-located servers on GPU 1
VLLM_GPU_MEM_UTIL_HONEST=0.92   # Can use most of GPU 0
VLLM_GPU_MEM_UTIL_SNEAKY=0.92  # Reduced for sharing GPU 1 (Example value, TUNE THIS!)


# --- Cleanup Function ---
cleanup() {
    echo "Caught signal. Cleaning up background processes..."
    # Kill all background processes started by this script
    # Using pkill -P $$ might be too broad if other unrelated background tasks exist.
    # Killing specific PIDs is safer if possible.
    if [[ -n $VLLM_PID_HONEST ]]; then kill $VLLM_PID_HONEST || echo "Honest Server already stopped."; fi
    if [[ -n $VLLM_PID_SNEAKY ]]; then kill $VLLM_PID_SNEAKY || echo "Sneaky Server already stopped."; fi
    # Add kills for training PIDs if needed, though accelerate might handle its children.
    echo "Cleanup finished."
    exit 1
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM

# --- Launch vLLM Servers ---
# Calculate tensor parallel size for each server based on assigned GPUs
NUM_GPUS_PER_SERVER_HONEST=$(echo $VLLM_HONEST_GPUS | awk -F',' '{print NF}')
NUM_GPUS_PER_SERVER_SNEAKY=$(echo $VLLM_SNEAKY_GPUS | awk -F',' '{print NF}')

cd /home/jvelja/learn_to_verify/pvg

CURRENT_PATH=$(pwd)
VLLM_SERVE_SCRIPT="${CURRENT_PATH}/inference/vllm_serve.py"

echo "Starting vLLM Server for Honest Prover on GPU(s) ${VLLM_HONEST_GPUS} (Port: ${VLLM_PORT_HONEST})..."
CUDA_VISIBLE_DEVICES=$VLLM_HONEST_GPUS python $VLLM_SERVE_SCRIPT \
    --model $HONEST_PROVER_PATH \
    --port $VLLM_PORT_HONEST \
    --host 0.0.0.0 \
    --dtype $VLLM_DTYPE \
    --gpu-memory-utilization $VLLM_GPU_MEM_UTIL_HONEST \
    --tensor-parallel-size $NUM_GPUS_PER_SERVER_HONEST \
    --max_model_len 4096 \
    --enable-prefix-caching True \
    & # Run in background
VLLM_PID_HONEST=$!
echo "Honest Prover vLLM Server PID: $VLLM_PID_HONEST"

echo "Starting vLLM Server for Sneaky Prover on GPU(s) ${VLLM_SNEAKY_GPUS} (Port: ${VLLM_PORT_SNEAKY})..."
CUDA_VISIBLE_DEVICES=$VLLM_SNEAKY_GPUS python $VLLM_SERVE_SCRIPT \
    --model $SNEAKY_PROVER_PATH \
    --port $VLLM_PORT_SNEAKY \
    --host 0.0.0.0 \
    --dtype $VLLM_DTYPE \
    --gpu-memory-utilization $VLLM_GPU_MEM_UTIL_SNEAKY \
    --tensor-parallel-size $NUM_GPUS_PER_SERVER_SNEAKY \
    --max_model_len 4096 \
    --enable-prefix-caching True \
    & # Run in background
VLLM_PID_SNEAKY=$!
echo "Sneaky Prover vLLM Server PID: $VLLM_PID_SNEAKY"

echo "Waiting ~25 seconds for vLLM servers to initialize..."
sleep 40

cd /home/jvelja/learn_to_verify

# Run full generation pipeline
uv run --env-file /home/jvelja/learn_to_verify/pvg/.env python generator.py \
  --dataset "jvelja/apps_checkable_filtered-verifier-regressor" \
  --output results_7b \
  --honest-port $VLLM_PORT_HONEST \
  --sneaky-port $VLLM_PORT_SNEAKY \
  --verbose


# # --- Wait for processes ---
echo "Launch complete. Waiting for training process (and servers) to finish..."
# Wait for the last background process launched (usually the training script)
# This isn't perfectly robust, as servers might finish first if training errors out quickly.
# A more robust solution might involve monitoring the training PID specifically if obtainable.
wait

# If training finishes normally, kill the servers
echo "Training finished or script interrupted. Killing vLLM servers..."
kill $VLLM_PID_HONEST || echo "Honest Server already stopped."
kill $VLLM_PID_SNEAKY || echo "Sneaky Server already stopped."

echo "Script finished."