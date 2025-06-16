#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=good_pvg_3b_zero2
#SBATCH --output=good_pvg/output/logs/good_pvg_3b_zero2.%j.out
#SBATCH --error=good_pvg/output/errors/good_pvg_3b_zero2.%j.err

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

# --- Configuration ---
# Get Parent directory
PARENT_DIR=$(dirname "$CURRENT_PATH")
VERIFIER_TRAINING_MODE="regressor"

# Paths to models/IDs on Hugging Face Hub or local paths
LOCAL_MODELS_DIR="/home/jvelja/local_models"
SNEAKY_PROVER_PATH="Qwen/Qwen2.5-3B-Instruct"
BASE_VERIFIER_PATH="Qwen/Qwen2.5-Coder-0.5B"
REGRESSOR_VERIFIER_PATH="jvelja/dummy-verifier-regressor" # This is a hack: allows vLLM to make room for the classification/regression head...

if [ "$VERIFIER_TRAINING_MODE" == "regressor" ]; then
    VERIFIER_PATH="${REGRESSOR_VERIFIER_PATH}"
else
    VERIFIER_PATH="${BASE_VERIFIER_PATH}"
fi

# Fetch models from local paths if they exist (dir = local_models/{MODEL_PROVIDER e.g. Qwen}/MODEL_NAME e.g. Qwen2.5-Coder-3B-Instruct)
# If they do not, echo an error message and exit
if [ ! -d "${LOCAL_MODELS_DIR}/${SNEAKY_PROVER_PATH}" ]; then
    echo "Error: Local model ${SNEAKY_PROVER_PATH} not found in ${LOCAL_MODELS_DIR}"
    exit 1
fi
if [ ! -d "${LOCAL_MODELS_DIR}/${VERIFIER_PATH}" ]; then
    echo "Error: Local model ${VERIFIER_PATH} not found in ${LOCAL_MODELS_DIR}"
    exit 1
fi

SNEAKY_PROVER_PATH="${LOCAL_MODELS_DIR}/${SNEAKY_PROVER_PATH}"
echo "Using local model ${SNEAKY_PROVER_PATH}"
VERIFIER_PATH="${LOCAL_MODELS_DIR}/${VERIFIER_PATH}"
echo "Using local model ${VERIFIER_PATH}"


VLLM_WORKER_MULTIPROC_METHOD=spawn # Seems the only way to avoid vllm new engine error
# --- NCCL Debugging ---
# Set NCCL debug level. Options: WARN, INFO, TRACE
# INFO is usually a good starting point for hangs.
# export NCCL_DEBUG=INFO
# Optional: Enable logging for specific subsystems (e.g., COLL for collectives, NET for network)
# export NCCL_DEBUG_SUBSYS=ALL

# More comprehensive NCCL timeout configuration
export NCCL_TIMEOUT_SEC=18000
# export NCCL_HEARTBEAT_TIMEOUT_SEC=18000    # Heartbeat timeout
# export NCCL_CHANNEL_TIMEOUT_SEC=18000      # Channel-specific timeout
export TORCH_NCCL_BLOCKING_WAIT=1                # Make NCCL operations blocking to avoid premature timeouts
# export NCCL_ASYNC_ERROR_HANDLING=1         # Better error handling
# export NCCL_DESYNC_DEBUG=1
 # Help debug desync issues

# Optional: Useful for getting Python tracebacks on segfaults
# export PYTHONFAULTHANDLER=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_TRACE_BUFFER_SIZE=2097152
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

# vLLM Server Ports
VLLM_PORT_SNEAKY=8000
VLLM_PORT_VERIFIER=8001 # New port for the Verifier server

# vLLM Server Host (use 127.0.0.1 for local communication on a single node)
VLLM_HOST="127.0.0.1"

# Training Script Arguments & Paths
OUTPUT_DIR="./output_disjoint_training_vllm_prover_verifier"
# DS_CONFIG_SNEAKY="ds_config_zero3_sneaky.json" # Training DS config for Sneaky Prover
DS_CONFIG_SNEAKY="zero/ds_config_zero2_sneaky.json" # Training DS config for Sneaky Prover
# DS_CONFIG_VERIFIER="ds_config_zero3_verifier.json" # Training DS config for Verifier
DS_CONFIG_VERIFIER="zero/ds_config_zero2_verifier.json" # Training DS config for Verifier
TRAIN_SCRIPT="train.py"       # Main Python training script
MAIN_SCRIPT="main.py"

# GPU Allocation (N=1, G=8 example)
VLLM_SNEAKY_GPUS="0"      # Sneaky Prover vLLM on GPU 0
VLLM_VERIFIER_GPUS="0"    # Verifier vLLM also on GPU 1 (Co-located)
# TRAINING_GPUS="2,3,4,5,6,7" # Training processes on GPUs 2-7
TRAINING_GPUS="1,2,3" # Training processes on GPUs 1-2-3 (Snellius)

# Calculate number of training processes based on allocated GPUs
NUM_TRAINING_GPUS=$(echo $TRAINING_GPUS | awk -F',' '{print NF}')

# Other vLLM args - TODO: Would be nice to automate these with a script
VLLM_DTYPE="bfloat16"
# Memory utilization needs careful tuning for co-located servers on GPU 1
VLLM_GPU_MEM_UTIL_SNEAKY=0.83  # Can use most of GPU 0
VLLM_GPU_MEM_UTIL_VERIFIER=0.1 # Reduced for sharing GPU 1 (Example value, TUNE THIS!)
# Ensure VLLM_GPU_MEM_UTIL_SNEAKY + VLLM_GPU_MEM_UTIL_VERIFIER <= ~0.9-1.0
# Set TASK_TYPE based on VERIFIER_TRAINING_MODE
if [ "$VERIFIER_TRAINING_MODE" == "regressor" ]; then
    TASK_TYPE="classify"
else
    TASK_TYPE="generate"
fi


# --- Cleanup Function ---
cleanup() {
    echo "Caught signal. Cleaning up background processes..."
    # Kill all background processes started by this script
    # Using pkill -P $$ might be too broad if other unrelated background tasks exist.
    # Killing specific PIDs is safer if possible.
    if [[ -n $VLLM_PID_SNEAKY ]]; then kill $VLLM_PID_SNEAKY || echo "Sneaky Server already stopped."; fi
    if [[ -n $VLLM_PID_VERIFIER ]]; then kill $VLLM_PID_VERIFIER || echo "Verifier Server already stopped."; fi
    # Add kills for training PIDs if needed, though accelerate might handle its children.
    echo "Cleanup finished."
    exit 1
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM

# --- Launch vLLM Servers ---
# Calculate tensor parallel size for each server based on assigned GPUs
NUM_GPUS_PER_SERVER_SNEAKY=$(echo $VLLM_SNEAKY_GPUS | awk -F',' '{print NF}')
NUM_GPUS_PER_SERVER_VERIFIER=$(echo $VLLM_VERIFIER_GPUS | awk -F',' '{print NF}')

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
sleep 5

echo "Starting vLLM Server for Verifier on GPU(s) ${VLLM_VERIFIER_GPUS} (Port: ${VLLM_PORT_VERIFIER})..."
CUDA_VISIBLE_DEVICES=$VLLM_VERIFIER_GPUS python $VLLM_SERVE_SCRIPT \
    --model $VERIFIER_PATH \
    --port $VLLM_PORT_VERIFIER \
    --host 0.0.0.0 \
    --dtype $VLLM_DTYPE \
    --gpu-memory-utilization $VLLM_GPU_MEM_UTIL_VERIFIER \
    --tensor-parallel-size $NUM_GPUS_PER_SERVER_VERIFIER \
    --max_model_len 5120 \
    --enable-prefix-caching True \
    --task-type $TASK_TYPE \
    & # Run in background
VLLM_PID_VERIFIER=$!
echo "Verifier vLLM Server PID: $VLLM_PID_VERIFIER"

echo "Waiting ~15 seconds for vLLM servers to initialize..."
sleep 15

# --- Launch Training Script ---
echo "Starting Training Script on GPUs ${TRAINING_GPUS}..."
# Set CUDA_VISIBLE_DEVICES for the accelerate launch command itself
export CUDA_VISIBLE_DEVICES=$TRAINING_GPUS
# Add parent directory to PYTHONPATH so absolute imports like "from pvg..." work
export PYTHONPATH="/home/jvelja/learn_to_verify:$PYTHONPATH"

# Use accelerate launch
# Ensure your accelerate config (if any) or CLI args match DeepSpeed/BF16 needs
uv run --env-file .env accelerate launch \
    --num_processes $NUM_TRAINING_GPUS \
    --num_machines 1 \
    --mixed_precision "bf16" \
    --use_deepspeed \
    --deepspeed_multinode_launcher standard \
    $MAIN_SCRIPT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_processes $NUM_TRAINING_GPUS \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_steps 100 \
    --output_dir "$OUTPUT_DIR" \
    --mixed_precision "bf16" \
    --num_rounds 8 \
    \
    --sneaky_prover.name_or_path "$SNEAKY_PROVER_PATH" \
    --sneaky_prover.use_flash_attention True \
    --verifier.name_or_path "$VERIFIER_PATH" \
    --verifier.use_flash_attention True \
    \
    --training_sneaky_prover.ds_config "$DS_CONFIG_SNEAKY" \
    --training_sneaky_prover.apply_liger_kernel True \
    --training_sneaky_prover.learning_rate 5e-6 \
    --training_sneaky_prover.lr_scheduler_type "constant_with_warmup" \
    --training_sneaky_prover.num_warmup_steps 10 \
    \
    --training_verifier.ds_config "$DS_CONFIG_VERIFIER" \
    --training_verifier.apply_liger_kernel True \
    --training_verifier.learning_rate 3e-4   \
    --training_verifier.lr_scheduler_type "linear" \
    --training_verifier.num_warmup_steps 100 \
    --training_verifier.verifier_mode "$VERIFIER_TRAINING_MODE" \
    --training_verifier.verifier_batch_size 8 \
    \
    --dataset.dataset_name "jvelja/apps_checkable_filtered" \
    \
    --rl.num_generations 8 \
    --rl.num_iterations 1 \
    --rl.beta 0.01 \
    --rl.scale_rewards True \
    \
    --vllm_sneaky_prover.host "$VLLM_HOST" \
    --vllm_sneaky_prover.port "$VLLM_PORT_SNEAKY" \
    --vllm_sneaky_prover.temperature 0.7 \
    --vllm_sneaky_prover.top_p 0.9 \
    --vllm_sneaky_prover.repetition_penalty 1.05 \
    --vllm_sneaky_prover.min_p 0.05 \
    --vllm_sneaky_prover.max_tokens 2120 \
    --vllm_sneaky_prover.stop_sequences '["</triggering_condition>"]' \
    \
    --vllm_verifier.host "$VLLM_HOST" \
    --vllm_verifier.port "$VLLM_PORT_VERIFIER" \
    --vllm_verifier.temperature 0.0 \
    --vllm_verifier.top_p 1.0 \
    --vllm_verifier.repetition_penalty 1.0 \
    --vllm_verifier.min_p 0.0 \
    --vllm_verifier.max_tokens 1 \
    --vllm_verifier.stop_sequences '["</verdict>"]' \
    --vllm_verifier.logprobs 0 \
    \
    --wandb.use_wandb True \
    --wandb.wandb_project_name "pvg" \
    --wandb.wandb_entity "jvelja-private" \
    --wandb.wandb_run_name "test_run_full_pipeline_${SLURM_JOB_ID}" \
    \
    --enable_backdoor_verification False \

# Get the PID of the main accelerate process (for waiting)
# This might be tricky as accelerate launch spawns multiple processes.
# Waiting for any background job to finish might be simpler if TRAIN_SCRIPT is the last one.
# Or find the main process PID if possible (e.g., rank 0).
# For simplicity now, we'll just wait for all background jobs started by this script.

# --- Wait for processes ---
echo "Launch complete. Waiting for training process (and servers) to finish..."
# Wait for the last background process launched (usually the training script)
# This isn't perfectly robust, as servers might finish first if training errors out quickly.
# A more robust solution might involve monitoring the training PID specifically if obtainable.
wait

# If training finishes normally, kill the servers
echo "Training finished or script interrupted. Killing vLLM servers..."
kill $VLLM_PID_SNEAKY || echo "Sneaky Server already stopped."
kill $VLLM_PID_VERIFIER || echo "Verifier Server already stopped."

echo "Script finished."