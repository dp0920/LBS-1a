#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --job-name=gait
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#
# Single training job. Called by launch_all.sh with env vars:
#   ALGO        = cma | random | de
#   INIT        = gait | stand | random
#   GENS        = number of generations
#   RESULTS_DIR = timestamped output directory

set -e

SIMDIR="/cluster/home/dparri03/robotics/LBS-1a/sim"

module load python/3.10.4
source "/cluster/home/dparri03/robotics/LBS-1a/venv/bin/activate"

ALGO="${ALGO:-cma}"
INIT="${INIT:-gait}"
GENS="${GENS:-150}"
POPSIZE="${POPSIZE:-48}"
RESULTS_DIR="${RESULTS_DIR:-$SIMDIR/results/manual}"

OUTDIR="$RESULTS_DIR/${ALGO}_${INIT}/${GENS}"
mkdir -p "$OUTDIR"

# Run training from a temp working directory so output files don't collide
# between parallel jobs writing to the same sim/ folder.
WORKDIR=$(mktemp -d)
cp "$SIMDIR/mujoco_gait.py" "$WORKDIR/"
cp "$SIMDIR/optimus_primal.urdf" "$WORKDIR/"
cp -r "$SIMDIR/meshes" "$WORKDIR/" 2>/dev/null || true
cd "$WORKDIR"

echo "=== Training config ==="
echo "  algo=$ALGO  init=$INIT  gens=$GENS  popsize=$POPSIZE"
echo "  cpus=$SLURM_CPUS_PER_TASK  node=$(hostname)"
echo "  outdir=$OUTDIR"
echo "  workdir=$WORKDIR"
echo "  started=$(date)"
echo ""

python3 mujoco_gait.py \
    --tune \
    --algo "$ALGO" \
    --init "$INIT" \
    --generations "$GENS" \
    --popsize "$POPSIZE" \
    --workers "$((SLURM_CPUS_PER_TASK - 1))"

# Move results to the output directory
echo ""
echo "=== Moving results to $OUTDIR ==="
for f in best_gait*.json tune_gait*.jsonl; do
    [ -f "$f" ] && mv -v "$f" "$OUTDIR/"
done

# Clean up temp working directory
cd /
rm -rf "$WORKDIR"

echo ""
echo "=== Done at $(date) ==="
