#!/bin/bash
# SLURM array job: one task per design. Each task preps its own structure
# (prepwizard -NOJOBID, no job server -- see the comment above the prepwizard
# call) then computes interaction energy + interface H-bonds
# (profile_binding_energy.py).
#
# Deliberately uses the DEFAULT `module load schrodinger` install, never
# `module unload`/a custom $SCHRODINGER path -- see sae/README.md and this
# session's plan notes on the Gates env-isolation incident.
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
# 477-design subset of interface_features_combined.yaml, made once by
# energy_bench/make_sample_yaml.py -- the full 210 MB file takes minutes and
# ~4+ GB to parse in each task (Schrodinger python has no libyaml).
INTERFACE_YAML="sae/results/run4/yamls/interface_features_energy_sample.yaml"
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

# Run prepwizard WITHOUT job control (-NOJOBID): it runs in the foreground
# inside this task, so no job server is involved at all. A per-node job
# server broke at scale (first full run, 2026-09-27): all tasks on a node
# share one jobserverd started inside whichever task got there first, so
# every prepwizard on that node ran in (and counted against the memory of)
# that one task's cgroup -- OOM kills, and SLURM killing the server when that
# task ended. The server location is also recorded in a single shared file
# in $HOME, so nodes overwrote each other's entry. -NOJOBID also dropped
# prepwizard from ~2.5 min to ~45 s per design.
#
# Work in a per-task scratch dir: prepwizard drops a <name>-001/ subjob dir
# and a log next to its output.
WORKDIR="${TMPDIR:-/tmp}/energy_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT
CIF_ABS="$(readlink -f "$CIF")"

echo "[$DESIGN_ID] running prepwizard (-NOJOBID) in $WORKDIR..."
set +e
( cd "$WORKDIR" && "$SCHRODINGER/utilities/prepwizard" -NOJOBID "$CIF_ABS" "${DESIGN_ID}.mae" > "${DESIGN_ID}.log" 2>&1 )
PREPWIZARD_EXIT=$?
set -e
[ -f "$WORKDIR/${DESIGN_ID}.log" ] && cp "$WORKDIR/${DESIGN_ID}.log" "$LOG"
[ -f "$WORKDIR/${DESIGN_ID}.mae" ] && mv "$WORKDIR/${DESIGN_ID}.mae" "$MAE"
if [ "$PREPWIZARD_EXIT" -ne 0 ]; then
    echo "ERROR: prepwizard exited with code $PREPWIZARD_EXIT" >&2
fi

if [ ! -f "$MAE" ]; then
    echo "ERROR: prepwizard finished but $MAE was not created" >&2
    for f in "$LOG"; do
        [ -f "$f" ] && { echo "--- tail of $f ---" >&2; tail -40 "$f" >&2; }
    done
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
