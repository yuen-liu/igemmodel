#!/bin/bash
set -euo pipefail

# Waluigi: no SLURM scheduler here, and no Schrodinger installed.
# Run this directly, ideally inside tmux/screen so it survives disconnects:
#   tmux new -s boltzgen
#   bash start_jupyter_boltzgen.bash
#
# Waluigi's 2 GPUs are shared with another user — check `nvidia-smi` first
# and pick a free one if needed:
#   CUDA_VISIBLE_DEVICES=1 bash start_jupyter_boltzgen.bash

VENV_DIR="$HOME/boltzgen_venv"

# First run only: create a clean venv with the system python. `venv` is
# isolated from system/site-packages by default, so this stays self-contained.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating boltzgen venv (first run, ~10 min — torch is large)..."
    # --without-pip + get-pip.py avoids depending on the system's ensurepip
    # (often missing on minimal installs, fixing it needs sudo apt).
    /usr/bin/python3 -m venv --without-pip "$VENV_DIR"
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python3"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install boltzgen pandas py3Dmol jupyterlab
    # Waluigi's driver (525.125.06) only supports up to CUDA 12.0, but `pip
    # install boltzgen` pulls in the latest torch, which ships CUDA 12.6+
    # builds — those fail with "NVIDIA driver is too old". Pin to the last
    # torch release with a CUDA 11.8 build, which is also the last one before
    # torch.load defaulted to weights_only=True (boltzgen's checkpoints trip
    # that check on newer torch).
    "$VENV_DIR/bin/pip" install "torch==2.5.1+cu118" --index-url https://download.pytorch.org/whl/cu118
fi

# boltzgen also needs --use_kernels false on this GPU: its cuequivariance
# triangular-attention kernel needs libnvrtc.so.12 (CUDA 12.x), which isn't
# installed since we're pinned to a cu118 torch build above.
#   boltzgen run ... --use_kernels false

source "$VENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

jupyter lab --no-browser --ip=$(hostname -s)
