#!/bin/bash
set -euo pipefail

# Waluigi: no SLURM scheduler here. Run this directly, ideally inside
# tmux/screen so it survives disconnects:
#   tmux new -s embed_esmc
#   bash run_embed_esmc.bash --manifest ~/notebooks/manifest.csv --output ~/notebooks/embeddings.npz
#
# Reuses the esmfold2_venv set up by start_jupyter_esmfold2.bash (same
# transformers/ESM-C install, same torch cu118 pin) -- this script only runs
# embed_esmc.py, which loads ESM-C alone (no ESMFold2 inversion/critic
# ensemble), so it's much lighter than a design job.
#
# Waluigi's 2 GPUs are shared with another user -- check `nvidia-smi` first:
#   CUDA_VISIBLE_DEVICES=1 bash run_embed_esmc.bash --manifest ... --output ...

VENV_DIR="$HOME/esmfold2_venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "esmfold2_venv not found at $VENV_DIR -- run start_jupyter_esmfold2.bash first" >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "$HOME/notebooks"

echo "Running smoke test (2 sequences) before the full batch..."
python embed_esmc.py "$@" --smoke-test

echo ""
echo "Smoke test passed. Running full embedding job..."
python embed_esmc.py "$@"
