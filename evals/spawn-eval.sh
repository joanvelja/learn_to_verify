#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=evals/output/logs/evaluation.%j.out
#SBATCH --error=evals/output/errors/evaluation.%j.err

# <--- Get Current Path --->
CURRENT_PATH=$(pwd)
echo "Current path: $CURRENT_PATH"
cd /home/jvelja/learn_to_verify

# <--- Load Modules --->
module purge
module load 2023
module load CUDA/12.4.0 # Why not load 2024? 2024 requires CUDA 12.6.0 (A hassle to remake the whole environment)
module load py # aliased to Python/3.11.3-GCCcore-12.3.0

# <--- Activate Virtual Environment --->
source .venv/bin/activate
echo "Activated virtual environment with uv"

module load CUDA/12.4.0


# <--- Check if Jupyter is installed --->
command -v jupyter >/dev/null || {
    echo "ERROR: 'jupyter' not found – check your module loads"; exit 1; }


# <--- Set Port and Token --->
PORT=8888           # always the same → VS Code can remember it
TOKEN=snellius      # any string; security comes from the SSH tunnel

LOGIN=${SLURM_SUBMIT_HOST}-pub.snellius.surf.nl
NODE=$(hostname)

# <--- Print SSH Command --->
cat <<EOF

╭───────────────────  LOCAL TERMINAL  ───────────────────╮
  ssh -N -J ${USER}@${LOGIN} ${USER}@${NODE} \
      -L ${PORT}:localhost:${PORT}
╰───────────────── leave that window running ────────────╯

Paste this once in VS Code’s kernel picker (Existing Jupyter Server):
    http://localhost:${PORT}/?token=${TOKEN}
    → choose “Always connect”
EOF

# <--- Launch Jupyter Lab --->
jupyter lab --no-browser --ip=0.0.0.0 --port=${PORT} \
            --NotebookApp.token=${TOKEN} \
            --NotebookApp.password=''