#!/bin/bash
# SLURM array job: Prime interaction energy for the structural-arm folds
# (native / steered mutant / random-substitution control) of feature 233.
#
# Same shape as run_energy_profiling.bash -- one task per structure, prepwizard
# then profile_binding_energy.py -- with only the inputs changed. Everything
# below that was learned the hard way on the 477-design run is preserved
# verbatim; see that script's comments for why.
#
#   - DEFAULT `module load schrodinger`. Never `module unload`, never a custom
#     $SCHRODINGER path (Gates env-isolation incident, sae/README.md).
#   - prepwizard -NOJOBID, in a per-task scratch dir. A shared per-node job
#     server OOM-killed tasks at scale and races on a single $HOME state file.
#
# Prerequisite:
#   python build_mutant_energy_inputs.py --cif-dir <folds dir>
#
# IMPORTANT -- test first:
#   sbatch --array=0-2  run_mutant_energy_profiling.bash
#   # confirm 3 sane CSVs in $OUTPUT_DIR, then
#   sbatch --array=0-99 run_mutant_energy_profiling.bash
#
#SBATCH --job-name=mutant_energy
#SBATCH --output=mutant_energy_logs/%A_%a.out
#SBATCH --error=mutant_energy_logs/%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

STRUCTURE_IDS_FILE="mutant_energy_ids.txt"
# Set to wherever the structural-arm folds landed. These are NOT the 477-design
# energy_sample structures.
STRUCTURES_DIR="${MUTANT_CIF_DIR:-mutant_folds}"
INTERFACE_YAML="interface_features_mutant_energy.yaml"
PREPPED_DIR="mutant_prepped"
OUTPUT_DIR="mutant_energy_results"
TARGET_CHAIN="A"
BINDER_CHAIN="C"

mkdir -p "$PREPPED_DIR" "$OUTPUT_DIR" mutant_energy_logs

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
STRUCTURE_ID=$(sed -n "${LINE_NUM}p" "$STRUCTURE_IDS_FILE")
if [ -z "$STRUCTURE_ID" ]; then
    echo "ERROR: no structure id at line $LINE_NUM of $STRUCTURE_IDS_FILE" >&2
    exit 1
fi

CIF="$STRUCTURES_DIR/${STRUCTURE_ID}.cif"
MAE="$PREPPED_DIR/${STRUCTURE_ID}.mae"
LOG="$PREPPED_DIR/${STRUCTURE_ID}.log"

if [ ! -f "$CIF" ]; then
    echo "ERROR: $CIF not found" >&2
    exit 1
fi

echo "[$STRUCTURE_ID] loading schrodinger module..."
module load schrodinger/2025-1

WORKDIR="${TMPDIR:-/tmp}/mutant_energy_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT
CIF_ABS="$(readlink -f "$CIF")"

echo "[$STRUCTURE_ID] running prepwizard (-NOJOBID) in $WORKDIR..."
set +e
( cd "$WORKDIR" && "$SCHRODINGER/utilities/prepwizard" -NOJOBID "$CIF_ABS" "${STRUCTURE_ID}.mae" > "${STRUCTURE_ID}.log" 2>&1 )
PREPWIZARD_EXIT=$?
set -e
[ -f "$WORKDIR/${STRUCTURE_ID}.log" ] && cp "$WORKDIR/${STRUCTURE_ID}.log" "$LOG"
[ -f "$WORKDIR/${STRUCTURE_ID}.mae" ] && mv "$WORKDIR/${STRUCTURE_ID}.mae" "$MAE"
if [ "$PREPWIZARD_EXIT" -ne 0 ]; then
    echo "ERROR: prepwizard exited with code $PREPWIZARD_EXIT" >&2
fi

if [ ! -f "$MAE" ]; then
    echo "ERROR: prepwizard finished but $MAE was not created" >&2
    [ -f "$LOG" ] && { echo "--- tail of $LOG ---" >&2; tail -40 "$LOG" >&2; }
    exit 1
fi

echo "[$STRUCTURE_ID] running profile_binding_energy.py..."
"$SCHRODINGER/run" python3 profile_binding_energy.py \
    --design-id "$STRUCTURE_ID" \
    --prepped-mae "$MAE" \
    --interface-yaml "$INTERFACE_YAML" \
    --target-chain "$TARGET_CHAIN" \
    --binder-chain "$BINDER_CHAIN" \
    --output-dir "$OUTPUT_DIR"

echo "[$STRUCTURE_ID] done."
