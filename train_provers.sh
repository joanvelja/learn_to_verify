#!/bin/bash


# --- Configuration ---
# Paths to models
MODEL_A_PATH="gpt2" # Replace with actual path/ID
MODEL_B_PATH="gpt2-medium" # Replace with actual path/ID
# (Placeholder) RM Model Path - uncomment and set if using vLLM for RM later
# RM_MODEL_PATH="<path_to_your_rm_model>"

# vLLM Server Ports
VLLM_PORT_A=8000
VLLM_PORT_B=8001
# (Placeholder) RM Server Port - uncomment if using vLLM for RM later
# VLLM_PORT_RM=8002

# vLLM Server Host (use 127.0.0.1 for local communication)
VLLM_HOST="127.0.0.1"

# Training Script Arguments
OUTPUT_DIR="./output_disjoint_training_vllm"
DS_CONFIG_A="ds_config_zero2.json" # Training DS config for model A
DS_CONFIG_B="ds_config_zero3.json" # Training DS config for model B
TRAIN_SCRIPT="advtrainer.py"
VLLM_SERVE_SCRIPT="vllm_serve.py" # Path to your vLLM server script

# Accelerate Config (Optional - can configure via CLI args too)
# ACCELERATE_CONFIG_FILE="accelerate_config.yaml"

# GPU Allocation
VLLM_A_GPUS="0"
VLLM_B_GPUS="1"
# (Placeholder) RM GPU - uncomment if using vLLM for RM later
# VLLM_RM_GPUS="1" # Co-located with Server B
TRAINING_GPUS="2,3,4,5,6,7"

# Calculate number of training processes
NUM_TRAINING_GPUS=$(echo $TRAINING_GPUS | awk -F',' '{print NF}')

# Other vLLM args (adjust as needed)
VLLM_DTYPE="auto"
VLLM_GPU_MEM_UTIL=0.9 # Default, adjust if co-locating on GPU 1 later

# --- Cleanup Function ---
cleanup() {
    echo "Caught signal. Cleaning up background processes..."
    # Find PIDs associated with this script's background tasks and kill them
    # This is a simple approach; more robust methods might be needed in complex setups
    pkill -P $$ # Kill processes whose parent is this script
    echo "Cleanup finished."
    exit 1
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM

# --- Launch vLLM Servers ---

echo "Starting vLLM Server A on GPU ${VLLM_A_GPUS} (Port: ${VLLM_PORT_A})..."
CUDA_VISIBLE_DEVICES=$VLLM_A_GPUS python $VLLM_SERVE_SCRIPT \
    --model $MODEL_A_PATH \
    --port $VLLM_PORT_A \
    --host 0.0.0.0 \
    --dtype $VLLM_DTYPE \
    --gpu-memory-utilization $VLLM_GPU_MEM_UTIL \
    --tensor-parallel-size 1 \
    & # Run in background
VLLM_PID_A=$!
echo "vLLM Server A PID: $VLLM_PID_A"

echo "Starting vLLM Server B on GPU ${VLLM_B_GPUS} (Port: ${VLLM_PORT_B})..."
CUDA_VISIBLE_DEVICES=$VLLM_B_GPUS python $VLLM_SERVE_SCRIPT \
    --model $MODEL_B_PATH \
    --port $VLLM_PORT_B \
    --host 0.0.0.0 \
    --dtype $VLLM_DTYPE \
    --gpu-memory-utilization $VLLM_GPU_MEM_UTIL \
    --tensor-parallel-size 1 \
    & # Run in background
VLLM_PID_B=$!
echo "vLLM Server B PID: $VLLM_PID_B"

# --- (Placeholder) Launch RM/Verifier Server ---
# If using vLLM for RM and co-locating on GPU 1:
# Adjust VLLM_GPU_MEM_UTIL for both Server B and RM Server carefully!
# echo "Starting vLLM RM Server on GPU ${VLLM_RM_GPUS} (Port: ${VLLM_PORT_RM})..."
# CUDA_VISIBLE_DEVICES=$VLLM_RM_GPUS python $VLLM_SERVE_SCRIPT \
#     --model $RM_MODEL_PATH \
#     --port $VLLM_PORT_RM \
#     --host 0.0.0.0 \
#     --dtype $VLLM_DTYPE \
#     --gpu-memory-utilization <RM_MEM_UTIL> \ # Needs careful tuning
#     --tensor-parallel-size 1 \
#     &
# VLLM_PID_RM=$!
# echo "vLLM RM Server PID: $VLLM_PID_RM"
# OR if RM is a simple python process:
# echo "Starting RM/Verifier Process on GPU ${VLLM_RM_GPUS}..."
# CUDA_VISIBLE_DEVICES=$VLLM_RM_GPUS python path/to/rm_script.py --port <rm_port> &
# RM_PID=$!
# echo "RM Process PID: $RM_PID"
# ---------------------------------------------

echo "Waiting a few seconds for vLLM servers to initialize..."
sleep 15 # Adjust as needed

# --- Launch Training Script ---

echo "Starting Training Script on GPUs ${TRAINING_GPUS}..."
# Set CUDA_VISIBLE_DEVICES for the accelerate launch command itself
export CUDA_VISIBLE_DEVICES=$TRAINING_GPUS

# Use accelerate launch
accelerate launch \
    --num_processes $NUM_TRAINING_GPUS \
    --num_machines 1 \
    --mixed_precision "no" \
    --use_deepspeed \
    $TRAIN_SCRIPT \
    --model_a_name_or_path $MODEL_A_PATH \
    --model_b_name_or_path $MODEL_B_PATH \
    --ds_config_a $DS_CONFIG_A \
    --ds_config_b $DS_CONFIG_B \
    --output_dir $OUTPUT_DIR \
    --vllm_host_a $VLLM_HOST \
    --vllm_port_a $VLLM_PORT_A \
    --vllm_host_b $VLLM_HOST \
    --vllm_port_b $VLLM_PORT_B \
    # Add other necessary training arguments from your ScriptArguments dataclass
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --learning_rate_a 5e-5 \
    --learning_rate_b 5e-5 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    # --validation_file <path> \
    # --eval_steps 100 \
    # --mixed_precision bf16 # Pass accelerate args here or use config file
    # --config_file $ACCELERATE_CONFIG_FILE # If using a config file

TRAIN_PID=$!
echo "Training Script (Accelerate Main Process) PID: $TRAIN_PID (Note: PIDs for other ranks differ)"

# --- Wait for processes ---
echo "Launch complete. Waiting for processes to finish..."
# Wait for the training process to finish.
# The vLLM servers will likely need manual termination or termination via the trap.
wait $TRAIN_PID

# If training finishes normally, kill the servers (optional)
echo "Training finished. Killing vLLM servers..."
kill $VLLM_PID_A || echo "Server A already stopped."
kill $VLLM_PID_B || echo "Server B already stopped."
# kill $VLLM_PID_RM || echo "RM Server already stopped." # Uncomment if RM server was started

echo "Script finished."