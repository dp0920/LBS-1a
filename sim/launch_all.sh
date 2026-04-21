#!/usr/bin/env bash
# Launch all 9 training jobs (3 algos × 3 seeds) at a given epoch count.
#
# Usage:
#   bash launch_all.sh 200              # 200 generations, all combos
#   bash launch_all.sh 1000             # 1000 generations
#   bash launch_all.sh 50 200 1000 2500 # multiple epoch counts (9 jobs each)
#
# This submits 9 jobs per epoch count to the batch partition.
# Results land in results/<algo>_<init>/.
# Logs land in slurm_logs/.

set -e

if [ $# -eq 0 ]; then
    echo "Usage: bash launch_all.sh <generations> [<generations> ...]"
    echo "  e.g.: bash launch_all.sh 50 200 1000 2500"
    exit 1
fi

cd "$(dirname "$0")"
mkdir -p slurm_logs results

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
                --export=ALL,ALGO="$ALGO",INIT="$INIT",GENS="$GENS" \
                train_job.sbatch
            total=$((total + 1))
        done
    done
    echo ""
done

echo "=== Submitted $total jobs ==="
echo "Monitor with: squeue -u \$USER"
echo "Results will be in results/<algo>_<init>/"
