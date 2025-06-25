#!/bin/bash

# Emergency Cleanup Script for spawn-vast-zero2.sh
# This script performs thorough cleanup after training script errors
# Run with: bash emergency_cleanup.sh

set +e  # Don't exit on errors during cleanup

echo "======================================"
echo "EMERGENCY CLEANUP INITIATED"
echo "======================================"
echo "Timestamp: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 1. KILL ALL RELATED PROCESSES
print_status "Step 1: Killing all related processes..."

# Kill vLLM servers by port
print_status "Killing processes on ports 8000 and 8001..."
for port in 8000 8001; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null || print_warning "Could not kill process $pid"
    else
        echo "  No process found on port $port"
    fi
done

# Kill vLLM processes by name/pattern
print_status "Killing vLLM processes..."
pkill -f "vllm_serve.py" && print_success "Killed vLLM serve processes" || echo "  No vLLM serve processes found"
pkill -f "vllm" && print_success "Killed additional vLLM processes" || echo "  No additional vLLM processes found"

# Kill training processes
print_status "Killing training and distributed processes..."
pkill -f "accelerate launch" && print_success "Killed accelerate processes" || echo "  No accelerate processes found"
pkill -f "deepspeed" && print_success "Killed deepspeed processes" || echo "  No deepspeed processes found"
pkill -f "train.py" && print_success "Killed train.py processes" || echo "  No train.py processes found"
pkill -f "main.py" && print_success "Killed main.py processes" || echo "  No main.py processes found"

# Kill any remaining Python processes that might be related
print_status "Killing remaining related Python processes..."
pkill -f "torch.*distributed" && print_success "Killed torch distributed processes" || echo "  No torch distributed processes found"
pkill -f "transformers" && print_success "Killed transformers processes" || echo "  No transformers processes found"

# Kill any processes using our specific models
print_status "Killing processes using specific model paths..."
pkill -f "Qwen2.5-3B-Instruct" && print_success "Killed Qwen2.5-3B processes" || echo "  No Qwen2.5-3B processes found"
pkill -f "Qwen2.5-Coder-0.5B" && print_success "Killed Qwen2.5-Coder processes" || echo "  No Qwen2.5-Coder processes found"
pkill -f "dummy-verifier-regressor" && print_success "Killed dummy-verifier processes" || echo "  No dummy-verifier processes found"

# 2. CLEAR GPU MEMORY AND RESET
print_status "Step 2: Clearing GPU memory and resetting GPUs..."

# Reset all NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    echo "  Resetting NVIDIA GPUs..."
    nvidia-smi --gpu-reset-ecc=0 2>/dev/null || print_warning "Could not reset ECC on GPUs"
    nvidia-smi --gpu-reset 2>/dev/null || print_warning "Could not reset GPUs directly"

    # Clear GPU memory by killing remaining GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | while read pid; do
        if [ -n "$pid" ] && [ "$pid" != "pid" ]; then
            echo "  Killing GPU process $pid"
            kill -9 $pid 2>/dev/null || print_warning "Could not kill GPU process $pid"
        fi
    done

    print_success "GPU reset attempted"
else
    print_warning "nvidia-smi not found, skipping GPU reset"
fi

# 3. CLEAR CACHES
print_status "Step 3: Clearing various caches..."

# Clear PyTorch cache
print_status "Clearing PyTorch caches..."
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print('  PyTorch CUDA cache cleared')
else:
    print('  CUDA not available for PyTorch cache clearing')
" 2>/dev/null || print_warning "Could not clear PyTorch cache"

# Clear HuggingFace cache
print_status "Clearing HuggingFace caches..."
if [ -d ~/.cache/huggingface ]; then
    echo "  Clearing HuggingFace transformers cache..."
    rm -rf ~/.cache/huggingface/transformers/tmp* 2>/dev/null || print_warning "Could not clear HF transformers temp files"
    rm -rf ~/.cache/huggingface/hub/tmp* 2>/dev/null || print_warning "Could not clear HF hub temp files"
    print_success "HuggingFace cache cleared"
fi

# Clear pip cache
print_status "Clearing pip cache..."
pip cache purge 2>/dev/null && print_success "Pip cache cleared" || print_warning "Could not clear pip cache"

# Clear uv cache
print_status "Clearing uv cache..."
uv cache clean 2>/dev/null && print_success "UV cache cleared" || print_warning "Could not clear uv cache"

# 4. CLEAR DISTRIBUTED TRAINING STATE
print_status "Step 4: Clearing distributed training state..."

# Remove distributed training temporary files
print_status "Removing distributed training temporary files..."
rm -rf /tmp/tmp* 2>/dev/null || true
rm -rf /tmp/torch_* 2>/dev/null || true
rm -rf /tmp/pytorch_* 2>/dev/null || true
rm -rf /dev/shm/torch_* 2>/dev/null || true

# Clear any leftover NCCL/distributed state
pkill -f "nccl" 2>/dev/null || true

# 5. FREE UP PORTS
print_status "Step 5: Ensuring ports are free..."

# Check and free specific ports used by the training script
for port in 8000 8001; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        print_warning "Port $port still in use, attempting to free..."
        fuser -k $port/tcp 2>/dev/null || print_warning "Could not free port $port"
    else
        echo "  Port $port is free"
    fi
done

# 6. CLEAN UP OUTPUT DIRECTORIES AND TEMP FILES
print_status "Step 6: Cleaning up temporary files and directories..."

# Navigate to the expected working directory
cd /root/learn_to_verify 2>/dev/null || cd /root/learn_to_verify/pvg 2>/dev/null || pwd

# Clean up output directories (be careful here)
print_status "Cleaning up training output directories..."
if [ -d "./output_disjoint_training_vllm_prover_verifier" ]; then
    echo "  Found training output directory"
    read -p "  Do you want to remove the training output directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "./output_disjoint_training_vllm_prover_verifier"
        print_success "Training output directory removed"
    else
        print_warning "Training output directory preserved"
    fi
fi

# Remove temporary checkpoints
rm -rf ./tmp_* 2>/dev/null || true
rm -rf ./.tmp* 2>/dev/null || true

# 7. RESET ENVIRONMENT VARIABLES
print_status "Step 7: Resetting environment variables..."

unset CUDA_VISIBLE_DEVICES
unset NCCL_TIMEOUT_SEC
unset TORCH_NCCL_BLOCKING_WAIT
unset TORCH_NCCL_TRACE_BUFFER_SIZE
unset TOKENIZERS_PARALLELISM
unset VLLM_USE_V1
unset VLLM_WORKER_MULTIPROC_METHOD
unset PYTHONFAULTHANDLER
unset TORCH_DISTRIBUTED_DEBUG

print_success "Environment variables reset"

# 8. FINAL SYSTEM CLEANUP
print_status "Step 8: Final system cleanup..."

# Clear shared memory
print_status "Clearing shared memory..."
if [ -d /dev/shm ]; then
    rm -rf /dev/shm/sem.torch_* 2>/dev/null || true
    rm -rf /dev/shm/torch_* 2>/dev/null || true
    print_success "Shared memory cleared"
fi

# Clear any leftover lock files
print_status "Clearing lock files..."
rm -rf /tmp/.torch_* 2>/dev/null || true
rm -rf /tmp/pytorch_* 2>/dev/null || true

# System cache cleanup
print_status "Performing system cache cleanup..."
sync
echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || print_warning "Could not clear system caches (need sudo)"

# 9. VERIFICATION
print_status "Step 9: Verification..."

echo ""
print_status "Verifying cleanup results:"

# Check if ports are free
for port in 8000 8001; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        print_error "Port $port is still in use!"
    else
        print_success "Port $port is free"
    fi
done

# Check for remaining processes
remaining_processes=$(ps aux | grep -E "(vllm|accelerate|deepspeed|train\.py|main\.py)" | grep -v grep | wc -l)
if [ "$remaining_processes" -gt 0 ]; then
    print_warning "$remaining_processes related processes still running:"
    ps aux | grep -E "(vllm|accelerate|deepspeed|train\.py|main\.py)" | grep -v grep | head -5
else
    print_success "No related processes found running"
fi

# Check GPU memory usage
if command -v nvidia-smi &> /dev/null; then
    gpu_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$gpu_usage" -lt 500 ]; then  # Less than 500MB used
        print_success "GPU memory usage is low ($gpu_usage MB)"
    else
        print_warning "GPU memory usage is still high ($gpu_usage MB)"
    fi
fi

echo ""
echo "======================================"
print_success "EMERGENCY CLEANUP COMPLETED"
echo "======================================"
echo "Timestamp: $(date)"
echo ""
print_status "Summary of actions taken:"
echo "  ✓ Killed all related processes"
echo "  ✓ Reset GPUs and cleared GPU memory"
echo "  ✓ Cleared PyTorch, HuggingFace, pip, and uv caches"
echo "  ✓ Removed distributed training temporary files"
echo "  ✓ Freed up ports 8000 and 8001"
echo "  ✓ Reset environment variables"
echo "  ✓ Cleared shared memory and lock files"
echo "  ✓ Performed system cleanup"
echo ""
print_status "You may now safely restart your training or other processes."
echo ""

# Optional: Show current system status
read -p "Do you want to see current system status? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    print_status "Current system status:"
    echo "--- GPU Status ---"
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
    echo ""
    echo "--- Port Status ---"
    netstat -tuln | grep -E ":800[01] " || echo "Ports 8000 and 8001 are free"
    echo ""
    echo "--- Memory Usage ---"
    free -h
    echo ""
fi

exit 0