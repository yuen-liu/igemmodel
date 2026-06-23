#!/bin/bash
set -euo pipefail

# Waluigi: no SLURM scheduler here, and no Schrodinger installed.
# Run this directly, ideally inside tmux/screen — first-time setup is slow:
#   tmux new -s genie3
#   bash start_jupyter_genie3.bash
#
# Waluigi's 2 GPUs are shared with another user — check `nvidia-smi` first
# and pick a free one if needed:
#   CUDA_VISIBLE_DEVICES=1 bash start_jupyter_genie3.bash
#
# First run does a lot, all scoped to your home directory (nothing system-wide,
# nothing Schrodinger-related):
#   - installs Miniconda to ~/miniconda3 (genie3's setup.sh requires conda)
#   - clones aqlaboratory/genie3 to ~/genie3
#   - runs genie3's own scripts/setup/setup.sh, which builds a `genie3` conda
#     env containing genie3 + ESMFold + ColabFold + FoldSeek + ProteinMPNN
#     (downloads several GB of AlphaFold weights along the way)
#   - downloads pretrained genie3 weights
# This can take well over an hour the first time. Reruns skip completed steps.

GENIE3_DIR="$HOME/genie3"
CONDA_DIR="$HOME/miniconda3"

if ! command -v conda &>/dev/null; then
    if [ ! -d "$CONDA_DIR" ]; then
        echo "Installing Miniconda to $CONDA_DIR (first run)..."
        curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
        rm /tmp/miniconda.sh
    fi
    export PATH="$CONDA_DIR/bin:$PATH"
fi

eval "$(conda shell.bash hook)"

if [ ! -d "$GENIE3_DIR" ]; then
    echo "Cloning genie3..."
    git clone https://github.com/aqlaboratory/genie3.git "$GENIE3_DIR"
fi

cd "$GENIE3_DIR"

# Builds the `genie3` conda env (skips already-completed steps on reruns).
if ! conda env list | awk '{print $1}' | grep -Fxq genie3; then
    echo "Running genie3 setup.sh (first run, this takes a while)..."
    bash scripts/setup/setup.sh
fi

conda activate genie3

if [ ! -d "$GENIE3_DIR/pretrained" ]; then
    echo "Downloading pretrained genie3 weights..."
    bash scripts/setup/download.sh --weights
fi

python -m pip install --quiet jupyterlab

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

jupyter lab --no-browser --ip=$(hostname -s)
