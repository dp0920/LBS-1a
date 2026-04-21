#!/usr/bin/env bash
# Launch all 9 training jobs (3 algos × 3 seeds) at a given epoch count.
#
# Usage:
#   bash launch_all.sh 200              # 200 generations, all combos
#   bash launch_all.sh 1000             # 1000 generations
#   bash launch_all.sh 50 200 1000 2500 # multiple epoch counts (9 jobs each)
#
# Results land in ~/robotics/LBS-1a/sim/results/<algo>_<init>_<gens>/
# Logs land in ~/robotics/LBS-1a/sim/slurm_logs/

set -e

SIMDIR="/cluster/home/dparri03/robotics/LBS-1a/sim"

if [ $# -eq 0 ]; then
    echo "Usage: bash launch_all.sh <generations> [<generations> ...]"
    echo "  e.g.: bash launch_all.sh 50 200 1000 2500"
    exit 1
fi

mkdir -p "$SIMDIR/slurm_logs" "$SIMDIR/results"

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
                --output="$SIMDIR/slurm_logs/${JOBNAME}_%j.out" \
                --error="$SIMDIR/slurm_logs/${JOBNAME}_%j.err" \
                --export=ALL,ALGO="$ALGO",INIT="$INIT",GENS="$GENS" \
                "$SIMDIR/train_job.sbatch"
            total=$((total + 1))
        done
    done
    echo ""
done

echo "=== Submitted $total jobs ==="
echo "Monitor with: squeue -u \$USER"
echo "Results in: $SIMDIR/results/<algo>_<init>_<gens>/"
echo "Logs in:    $SIMDIR/slurm_logs/"
