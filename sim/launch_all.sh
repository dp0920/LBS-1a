#!/usr/bin/env bash
# Launch all 9 training jobs (3 algos × 3 seeds) at a given epoch count.
#
# Usage:
#   bash launch_all.sh 200              # 200 generations, all combos
#   bash launch_all.sh 1000             # 1000 generations
#   bash launch_all.sh 50 200 1000 2500 # multiple epoch counts (9 jobs each)
#
# Each invocation creates a timestamped run directory:
#   results/<timestamp>/<algo>_<init>/<gens>/
#   logs/<timestamp>/

set -e

SIMDIR="/cluster/home/dparri03/robotics/LBS-1a/sim"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ $# -eq 0 ]; then
    echo "Usage: bash launch_all.sh <generations> [<generations> ...]"
    echo "  e.g.: bash launch_all.sh 50 200 1000 2500"
    exit 1
fi

RESULTS_DIR="$SIMDIR/results/$TIMESTAMP"
LOG_DIR="$SIMDIR/logs/$TIMESTAMP"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "=== Run: $TIMESTAMP ==="
echo "  Results: $RESULTS_DIR"
echo "  Logs:    $LOG_DIR"
echo ""

ALGOS="cma random de"
INITS="gait stand random"

total=0
for GENS in "$@"; do
    echo "=== Launching jobs for $GENS generations ==="
    for ALGO in $ALGOS; do
        for INIT in $INITS; do
            JOBNAME="gait_${ALGO}_${INIT}_${GENS}"
            echo "  submitting $JOBNAME ..."
            sbatch \
                --job-name="$JOBNAME" \
                --output="$LOG_DIR/${JOBNAME}_%j.out" \
                --error="$LOG_DIR/${JOBNAME}_%j.err" \
                --export=ALL,ALGO="$ALGO",INIT="$INIT",GENS="$GENS",RESULTS_DIR="$RESULTS_DIR" \
                "$SIMDIR/train_job.sbatch"
            total=$((total + 1))
        done
    done
    echo ""
done

echo "=== Submitted $total jobs ==="
echo "Monitor with: squeue -u \$USER"
echo "Results in: $RESULTS_DIR"
echo "Logs in:    $LOG_DIR"

# Save a manifest for later reference
cat > "$RESULTS_DIR/manifest.txt" <<MANIFEST
Run: $TIMESTAMP
Generations: $@
Algorithms: $ALGOS
Seeds: $INITS
Jobs: $total
MANIFEST
echo "Manifest saved to $RESULTS_DIR/manifest.txt"
