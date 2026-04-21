#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --job-name=gait
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#
# Single training job. Called by launch_all.sh with env vars:
#   ALGO  = cma | random | de
#   INIT  = gait | stand | random
#   GENS  = number of generations
#
# Output files land in results/<ALGO>_<INIT>/
#   best_gait_<suffix>.json  — best parameters
#   tune_gait_<suffix>.jsonl — per-generation log

set -e

module load python/3.10.4
source "$HOME/LBS-1a/sim/venv_cluster/bin/activate"

ALGO="${ALGO:-cma}"
INIT="${INIT:-gait}"
GENS="${GENS:-150}"
POPSIZE="${POPSIZE:-48}"

# Create output directory per run
OUTDIR="$HOME/LBS-1a/sim/results/${ALGO}_${INIT}"
mkdir -p "$OUTDIR"

cd "$HOME/LBS-1a/sim"

echo "=== Training config ==="
echo "  algo=$ALGO  init=$INIT  gens=$GENS  popsize=$POPSIZE"
echo "  cpus=$SLURM_CPUS_PER_TASK  node=$(hostname)"
echo "  outdir=$OUTDIR"
echo "  started=$(date)"
echo ""

python mujoco_gait.py \
    --tune \
    --algo "$ALGO" \
    --init "$INIT" \
    --generations "$GENS" \
    --popsize "$POPSIZE" \
    --workers "$((SLURM_CPUS_PER_TASK - 1))"

# Copy results to the per-run output directory
echo ""
echo "=== Copying results to $OUTDIR ==="
for f in best_gait*.json tune_gait*.jsonl; do
    [ -f "$f" ] && cp -v "$f" "$OUTDIR/"
done

echo ""
echo "=== Done at $(date) ==="
