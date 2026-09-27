#!/bin/bash
# SLURM array job: one task per design. Each task preps its own structure
# (prepwizard, via jobcontrol) then computes interaction energy + interface
# H-bonds (profile_binding_energy.py).
#
# Deliberately uses the DEFAULT `module load schrodinger` install, never
# `module unload`/a custom $SCHRODINGER path -- see sae/README.md and this
# session's plan notes on the Gates env-isolation incident. Each array task
# runs on its own compute node, so each starts its OWN local jobcontrol
# server (a login-node server, like the one used interactively, is not
# reachable from a compute node).
#
# IMPORTANT -- run a small test first: sbatch --array=0-2 this script,
# confirm all 3 tasks produce a sane .csv in $OUTPUT_DIR before submitting
# the full array. Two things in profile_binding_energy.py are unverified
# with real batch data as of this writing (prepwizard behavior at scale,
# get_hydrogen_bonds' real output) -- don't submit 477 tasks blind.
#
# Usage:
#   sbatch --array=0-476 run_energy_profiling.bash            # full run
#   sbatch --array=0-2   run_energy_profiling.bash            # small test first
#
#SBATCH --job-name=energy_profile
#SBATCH --output=energy_logs/%A_%a.out
#SBATCH --error=energy_logs/%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

DESIGN_IDS_FILE="energy_sample_design_ids.txt"
STRUCTURES_DIR="energy_sample"
INTERFACE_YAML="sae/results/run4/yamls/interface_features_combined.yaml"
PREPPED_DIR="prepped"
OUTPUT_DIR="energy_results"
TARGET_CHAIN="A"
BINDER_CHAIN="C"

mkdir -p "$PREPPED_DIR" "$OUTPUT_DIR" energy_logs

# SLURM_ARRAY_TASK_ID is 0-indexed here (matches `sbatch --array=0-N`);
# sed is 1-indexed, hence +1.
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
DESIGN_ID=$(sed -n "${LINE_NUM}p" "$DESIGN_IDS_FILE")
if [ -z "$DESIGN_ID" ]; then
    echo "ERROR: no design id at line $LINE_NUM of $DESIGN_IDS_FILE" >&2
    exit 1
fi

CIF="$STRUCTURES_DIR/${DESIGN_ID}.cif"
MAE="$PREPPED_DIR/${DESIGN_ID}.mae"
LOG="$PREPPED_DIR/${DESIGN_ID}.log"

if [ ! -f "$CIF" ]; then
    echo "ERROR: $CIF not found" >&2
    exit 1
fi

echo "[$DESIGN_ID] loading schrodinger module..."
module load schrodinger/2025-1

echo "[$DESIGN_ID] starting this task's own local job server..."
"$SCHRODINGER/jsc" local-server-start

# Give the job server a moment to actually be ready to accept submissions --
# "Job server available" printing doesn't necessarily mean it can take a job
# yet. Poll `jsc list` (works once the server is truly up) instead of a
# blind sleep, with a sleep as a floor.
sleep 2
for i in 1 2 3 4 5 6 7 8 9 10; do
    if "$SCHRODINGER/jsc" list >/dev/null 2>&1; then
        break
    fi
    echo "[$DESIGN_ID] job server not ready yet, waiting..."
    sleep 3
done

echo "[$DESIGN_ID] running prepwizard..."
set +e
PREPWIZARD_OUTPUT=$("$SCHRODINGER/utilities/prepwizard" "$CIF" "$MAE" 2>&1)
PREPWIZARD_EXIT=$?
set -e
echo "$PREPWIZARD_OUTPUT"
if [ "$PREPWIZARD_EXIT" -ne 0 ]; then
    echo "ERROR: prepwizard exited with code $PREPWIZARD_EXIT -- output above" >&2
    exit 1
fi
JOBID=$(echo "$PREPWIZARD_OUTPUT" | grep -oE 'JobId: [0-9a-f-]+' | awk '{print $2}')
if [ -z "$JOBID" ]; then
    echo "ERROR: prepwizard did not return a JobId for $CIF -- output above" >&2
    exit 1
fi

echo "[$DESIGN_ID] waiting for prepwizard job $JOBID..."
ELAPSED=0
TIMEOUT=1200
while "$SCHRODINGER/jsc" list 2>/dev/null | grep -q "^$JOBID"; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "ERROR: prepwizard job $JOBID still running after ${TIMEOUT}s -- giving up" >&2
        exit 1
    fi
done

if [ ! -f "$MAE" ]; then
    echo "ERROR: prepwizard job $JOBID finished but $MAE was not created -- check for a "\
"${DESIGN_ID}*.log file in the working directory" >&2
    exit 1
fi

echo "[$DESIGN_ID] running profile_binding_energy.py..."
"$SCHRODINGER/run" python3 profile_binding_energy.py \
    --design-id "$DESIGN_ID" \
    --prepped-mae "$MAE" \
    --interface-yaml "$INTERFACE_YAML" \
    --target-chain "$TARGET_CHAIN" \
    --binder-chain "$BINDER_CHAIN" \
    --output-dir "$OUTPUT_DIR"

echo "[$DESIGN_ID] done."
