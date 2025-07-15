# Get root dir (current) and set it as ROOT_DIR
ROOT_DIR=$(pwd)

# update lists
sudo apt-get update

# install llvm-dev and uv
sudo apt-get install -y llvm-dev

# install Rust toolchain non-interactively
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# make Rust env immediately available
source "$HOME/.cargo/env"

# ensure new shells pick it up
grep -qxF 'source $HOME/.cargo/env' ~/.bashrc \
  || echo 'source $HOME/.cargo/env' >> ~/.bashrc

# --- disable tmux auto-attach ---
touch ~/.no_auto_tmux

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone repo
cd $ROOT_DIR

git clone -b mono-prover-fix-vllm https://github.com/joanvelja/learn_to_verify.git

cd learn_to_verify

# Make alias for uv activate
alias uv-act='source .venv/bin/activate'

# make uv venv
uv venv --python 3.11

# install dependencies
uv sync
uv add flash-attn --no-build-isolation

# Download models locally
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $ROOT_DIR/local_models/Qwen/Qwen2.5-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir $ROOT_DIR/local_models/Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download jvelja/dummy-verifier-regressor --local-dir $ROOT_DIR/local_models/jvelja/dummy-verifier-regressor