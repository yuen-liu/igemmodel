#!/bin/bash
set -euo pipefail

# Waluigi: no SLURM scheduler here, and no Schrodinger installed.
# Run this directly, ideally inside tmux/screen so it survives disconnects:
#   tmux new -s esmfold2
#   bash start_jupyter_esmfold2.bash
#
# Note: the esmfold2_s100b notebook runs design jobs on this machine's GPU
# (not Modal's cloud GPUs — that costs money). binder_design.py is pointed at
# Biohub's smaller scaling-study checkpoints (paired with ESMC-300M, not the
# flagship's ESMC-6B) to avoid a ~24GB download/VRAM commitment — lower
# design quality than the flagship, see the notebook intro for the tradeoff.
# Waluigi's 2 GPUs are shared with another user — check `nvidia-smi` first:
#   CUDA_VISIBLE_DEVICES=1 bash start_jupyter_esmfold2.bash

VENV_DIR="$HOME/esmfold2_venv"
UV_BIN="$HOME/.local/bin/uv"

# esm requires Python >=3.12,<3.13, but waluigi's system python3 is 3.10 and
# there's no sudo to install 3.12. uv downloads its own self-contained Python
# build instead — no sudo, no system packages needed.
if [ ! -x "$UV_BIN" ]; then
    echo "Installing uv (no sudo needed)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# First run only: create a clean venv with a uv-managed Python 3.12.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating esmfold2 venv with Python 3.12 (first run)..."
    "$UV_BIN" python install 3.12
    "$UV_BIN" venv "$VENV_DIR" --python 3.12
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
        "esm@git+https://github.com/Biohub/esm.git@main" modal py3dmol pyarrow jupyterlab
    # Waluigi's driver (525.125.06) only supports up to CUDA 12.0; pip pulls
    # in the latest torch by default, which needs a newer driver. Pin to the
    # last torch release with a CUDA 11.8 build (see boltzgen script for the
    # full story on this).
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
        "torch==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118
fi

source "$VENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# binder_design.py lives alongside the notebooks, not next to this script —
# launch from there so `from binder_design import ...` resolves without
# needing a manual %cd in the notebook.
cd "$HOME/notebooks"
jupyter lab --no-browser --ip=$(hostname -s)
