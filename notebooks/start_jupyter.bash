#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -J boltzgen_jup
#SBATCH -o boltzgen_jup.out

LALOOSAE_PYTHON=/mnt/beegfs/home/friesner/bgl2126/schrod_envs/laloosae/bin/python3
VENV_DIR="$HOME/boltzgen_venv"

# First run only: create a clean venv with no Schrodinger contamination.
# Uses laloosae's Python binary but inherits none of its site-packages.
# Takes ~10 min because torch is large.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating clean boltzgen venv (first run, ~10 min)..."
    "$LALOOSAE_PYTHON" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install boltzgen jupyterlab
fi

source "$VENV_DIR/bin/activate"

export OMP_NUM_THREADS=$SLURM_NTASKS

jupyter lab --no-browser --ip=$(hostname -s)
