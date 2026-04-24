#!/usr/bin/env bash
# Launch training jobs across (algo × init × gens × interp × random-tilt-trials).
#
# For every (algo, init, gens, interp) combo we submit N_TRIALS jobs, each
# with a random fall-tilt drawn from [TILT_MIN, TILT_MAX] degrees. The
# tilt_scale scales inversely (20/tilt) so strict kill thresholds also apply
# stricter soft penalties.
#
# Usage:
#   bash launch_all.sh 200                       # default: 5 trials, all interps
#   bash launch_all.sh 50 200 1000               # multiple epoch counts
#   N_TRIALS=10 bash launch_all.sh 1000          # 10 random tilts per config
#   INTERPS="linear cosine" bash launch_all.sh 1000
#   TILT_MIN=5 TILT_MAX=30 bash launch_all.sh 1000
#
# Results land in:
#   results/<timestamp>/<algo>_<init>/<interp>/t<tilt>_trial<N>/<gens>/
#   logs/<timestamp>/

set -e

SIMDIR="/cluster/home/dparri03/robotics/LBS-1a/sim"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ $# -eq 0 ]; then
    echo "Usage: bash launch_all.sh <generations> [<generations> ...]"
    echo ""
    echo "Optional env vars:"
    echo "  N_TRIALS=5            Random tilt trials per config"
    echo "  TILT_MIN=0            Min random tilt (deg)"
    echo "  TILT_MAX=40           Max random tilt (deg)"
    echo "  INTERPS='linear cosine smoothstep'   Interpolation modes to test"
    echo "  ALGOS='cma random de'"
    echo "  INITS='gait stand random'"
    exit 1
fi

RESULTS_DIR="$SIMDIR/results/$TIMESTAMP"
LOG_DIR="$SIMDIR/logs/$TIMESTAMP"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

ALGOS="${ALGOS:-cma random de}"
INITS="${INITS:-gait stand random}"
INTERPS="${INTERPS:-linear cosine smoothstep}"
N_TRIALS="${N_TRIALS:-5}"
TILT_MIN="${TILT_MIN:-0}"
TILT_MAX="${TILT_MAX:-40}"
SIGMA_MIN="${SIGMA_MIN:-2}"
SIGMA_MAX="${SIGMA_MAX:-15}"

echo "=== Run: $TIMESTAMP ==="
echo "  Results:  $RESULTS_DIR"
echo "  Logs:     $LOG_DIR"
echo "  Algos:    $ALGOS"
echo "  Inits:    $INITS"
echo "  Interps:  $INTERPS"
echo "  Trials:   $N_TRIALS per config"
echo "  Tilt:     random in [$TILT_MIN, $TILT_MAX]°"
echo "  Sigma:    random in [$SIGMA_MIN, $SIGMA_MAX] (CMA only)"
echo ""

# Draw a random tilt in [TILT_MIN, TILT_MAX]. Derive tilt_scale inversely:
# scale = 20/tilt, clamped to [0.3, 5]. Strict (low tilt) → high penalty scale.
draw_tilt() {
    python3 -c "
import random
tilt = random.uniform($TILT_MIN, $TILT_MAX)
# Clamp to min 0.5° so we don't divide by ~0.
tilt = max(0.5, tilt)
scale = min(5.0, max(0.3, 20.0 / tilt))
print(f'{tilt:.2f} {scale:.3f}')
"
}

# Draw a random integer sigma_init in [SIGMA_MIN, SIGMA_MAX].
draw_sigma() {
    python3 -c "import random; print(random.randint(int($SIGMA_MIN), int($SIGMA_MAX)))"
}

total=0
for GENS in "$@"; do
    for INTERP in $INTERPS; do
        for ALGO in $ALGOS; do
            for INIT in $INITS; do
                for TRIAL in $(seq 1 $N_TRIALS); do
                    read -r FALL_TILT TILT_SCALE <<< "$(draw_tilt)"
                    SIGMA_INIT=$(draw_sigma)
                    JOBNAME="gait_${ALGO}_${INIT}_${INTERP}_t${FALL_TILT}_s${SIGMA_INIT}_trial${TRIAL}_g${GENS}"
                    echo "  $JOBNAME  (tilt=${FALL_TILT}° scale=${TILT_SCALE} sigma=${SIGMA_INIT})"
                    sbatch \
                        --job-name="$JOBNAME" \
                        --output="$LOG_DIR/${JOBNAME}_%j.out" \
                        --error="$LOG_DIR/${JOBNAME}_%j.err" \
                        --export=ALL,ALGO="$ALGO",INIT="$INIT",GENS="$GENS",FALL_TILT="$FALL_TILT",TILT_SCALE="$TILT_SCALE",INTERP="$INTERP",SIGMA_INIT="$SIGMA_INIT",TRIAL="$TRIAL",RESULTS_DIR="$RESULTS_DIR" \
                        "$SIMDIR/train_job.sbatch" > /dev/null
                    total=$((total + 1))
                done
            done
        done
    done
done

echo ""
echo "=== Submitted $total jobs ==="
echo "Monitor with: squeue -u \$USER"
echo "Results in:   $RESULTS_DIR"

# Save a manifest for later reference
cat > "$RESULTS_DIR/manifest.txt" <<MANIFEST
Run: $TIMESTAMP
Generations: $@
Algorithms: $ALGOS
Seeds: $INITS
Interpolations: $INTERPS
Trials per config: $N_TRIALS
Random tilt range: [$TILT_MIN, $TILT_MAX]°
Random sigma range: [$SIGMA_MIN, $SIGMA_MAX]
Jobs: $total
MANIFEST
echo "Manifest saved to $RESULTS_DIR/manifest.txt"
