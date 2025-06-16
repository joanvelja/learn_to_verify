# Exit immediately if a command exits with a non-zero status.
set -e
# Get current path
CURRENT_PATH=$(pwd)
cd /root/learn_to_verify

PARENT_DIR=$(dirname "$CURRENT_PATH")
VERIFIER_TRAINING_MODE="regressor"

# Paths to models/IDs on Hugging Face Hub or local paths
LOCAL_MODELS_DIR="/root/local_models"
SNEAKY_PROVER_PATH="Qwen/Qwen2.5-3B-Instruct"
BASE_VERIFIER_PATH="Qwen/Qwen2.5-Coder-0.5B"
REGRESSOR_VERIFIER_PATH="jvelja/dummy-verifier-regressor" # This is a hack: allows vLLM to make room for the classification/regression head...

SLURM_JOB_ID=$(shuf -i 100000-999999 -n 1)

if [ "$VERIFIER_TRAINING_MODE" == "regressor" ]; then
    VERIFIER_PATH="${REGRESSOR_VERIFIER_PATH}"
else
    VERIFIER_PATH="${BASE_VERIFIER_PATH}"
fi

# --- Cleanup Function ---
cleanup() {
    echo "Cleaning up background processes..."
    # Kill all background processes started by this script
    if [[ -n $VLLM_PID_SNEAKY ]]; then
        echo "Killing Sneaky Server (PID: $VLLM_PID_SNEAKY)"
        kill -9 $VLLM_PID_SNEAKY 2>/dev/null || echo "Sneaky Server already stopped."
    fi
    if [[ -n $VLLM_PID_VERIFIER ]]; then
        echo "Killing Verifier Server (PID: $VLLM_PID_VERIFIER)"
        kill -9 $VLLM_PID_VERIFIER 2>/dev/null || echo "Verifier Server already stopped."
    fi

    # Kill any remaining Python processes related to this script
    echo "Killing any remaining related Python processes..."
    pkill -f "python $VLLM_SERVE_SCRIPT" || echo "No Python vLLM processes found."
    pkill -f "accelerate launch" || echo "No accelerate processes found."

    # Reset GPUs
    echo "Attempting to reset NVIDIA GPUs..."
    nvidia-smi --gpu-reset 2>/dev/null || echo "Could not reset GPUs, may require manual intervention."

    echo "Cleanup finished."
}

# Trap signals for cleanup
trap cleanup EXIT SIGINT SIGTERM ERR

# Rest of your script remains the same...
# Fetch models from local paths if they exist
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
export NCCL_TIMEOUT_SEC=7200
export TORCH_NCCL_BLOCKING_WAIT=1                # Make NCCL operations blocking to avoid premature timeouts
# --- NCCL Debugging ---
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
DS_CONFIG_SNEAKY="zero/ds_config_zero2_sneaky.json" # Training DS config for Sneaky Prover
DS_CONFIG_VERIFIER="zero/ds_config_zero2_verifier.json" # Training DS config for Verifier
TRAIN_SCRIPT="train.py"       # Main Python training script
MAIN_SCRIPT="main.py"

# GPU Allocation (automatically detect available GPUs and assign for vLLM + training)
TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$TOTAL_GPUS" -le 1 ]; then
    echo "Error: At least 2 GPUs are required (1 for vLLM, rest for training)."
    exit 1
fi

VLLM_SNEAKY_GPUS="0"  # Reserve GPU 0 for Sneaky Prover vLLM server
VLLM_VERIFIER_GPUS="0"  # Optionally, you can set this to another GPU if needed

# Assign training GPUs as all GPUs except GPU 0 (for vLLM)
TRAINING_GPUS=$(seq 1 $((TOTAL_GPUS-1)) | paste -sd, -)
echo "Detected $TOTAL_GPUS GPUs: using GPU 0 for vLLM, GPUs $TRAINING_GPUS for training."

# Calculate number of training processes based on allocated GPUs
NUM_TRAINING_GPUS=$(echo $TRAINING_GPUS | awk -F',' '{print NF}')


# Memory utilization needs careful tuning for co-located servers on GPU 1
VLLM_DTYPE="bfloat16"
# Memory utilization needs careful tuning for co-located servers on GPU 1
VLLM_GPU_MEM_UTIL_SNEAKY=0.83  # Can use most of GPU 0
VLLM_GPU_MEM_UTIL_VERIFIER=0.11 # Reduced for sharing GPU 1 (Example value, TUNE THIS!)
# Ensure VLLM_GPU_MEM_UTIL_SNEAKY + VLLM_GPU_MEM_UTIL_VERIFIER <= ~0.9-1.0
# Set TASK_TYPE based on VERIFIER_TRAINING_MODE
if [ "$VERIFIER_TRAINING_MODE" == "regressor" ]; then
    TASK_TYPE="classify"
else
    TASK_TYPE="generate"
fi

cd /root/learn_to_verify/pvg

# --- Launch vLLM Servers ---
# Calculate tensor parallel size for each server based on assigned GPUs
NUM_GPUS_PER_SERVER_SNEAKY=$(echo $VLLM_SNEAKY_GPUS | awk -F',' '{print NF}')
NUM_GPUS_PER_SERVER_VERIFIER=$(echo $VLLM_VERIFIER_GPUS | awk -F',' '{print NF}')

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
export PYTHONPATH="/root/learn_to_verify:$PYTHONPATH"

# Set timeout to automatically kill the process if it hangs
TRAINING_TIMEOUT=28800  # 8 hours in seconds, adjust as needed

# Monitor script to detect errors and trigger cleanup
(
    # Use accelerate launch with timeout
    timeout $TRAINING_TIMEOUT uv run accelerate launch \
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
        --training_verifier.apply_liger_kernel False \
        --training_verifier.max_grad_norm 1.0 \
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
        --vllm_sneaky_prover.top_p 0.95 \
        --vllm_sneaky_prover.frequency_penalty 0.05 \
        --vllm_sneaky_prover.min_p 0.05 \
        --vllm_sneaky_prover.max_tokens 2120 \
        --vllm_sneaky_prover.stop_sequences '["</triggering_condition>"]' \
        \
        --vllm_verifier.host "$VLLM_HOST" \
        --vllm_verifier.port "$VLLM_PORT_VERIFIER" \
        --vllm_verifier.temperature 0.0 \
        --vllm_verifier.top_p 0.95 \
        --vllm_verifier.frequency_penalty 0.05 \
        --vllm_verifier.min_p 0.05 \
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

    TRAINING_EXIT_CODE=$?

    # Check if timeout killed the process
    if [ $TRAINING_EXIT_CODE -eq 124 ]; then
        echo "Training process timed out after $TRAINING_TIMEOUT seconds."
        exit 1
    # Check if process exited abnormally
    elif [ $TRAINING_EXIT_CODE -ne 0 ]; then
        echo "Training process failed with exit code $TRAINING_EXIT_CODE."
        exit 1
    fi
)
TRAINING_RESULT=$?

# If training exits with an error, cleanup will be called by the trap
if [ $TRAINING_RESULT -ne 0 ]; then
    echo "Training process failed. Triggering cleanup..."
    exit 1
fi

echo "Training finished successfully. Script complete."