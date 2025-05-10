# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart the tmux pane
tmux attach

# Reload the rc to have uv available
source ~/.bashrc

# Change dir to where toml is
cd src/verifiers

# Install the dependencies
uv sync

# Alias the activate command
alias activate=".venv/bin/activate"

