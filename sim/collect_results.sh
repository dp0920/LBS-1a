#!/usr/bin/env bash
# Collect and summarize results from all training runs.
# Usage: bash collect_results.sh

SIMDIR="/cluster/home/dparri03/robotics/LBS-1a/sim"

echo "=== Training Results Summary ==="
echo ""
printf "%-30s %10s %8s %10s\n" "Run" "Reward" "Gen" "Restarts"
printf "%-30s %10s %8s %10s\n" "----" "------" "---" "--------"

for dir in "$SIMDIR"/results/*/; do
    [ -d "$dir" ] || continue
    for json in "$dir"best_gait*.json; do
        [ -f "$json" ] || continue
        name=$(basename "$dir")
        reward=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['reward']:+.2f}\")")
        gen=$(python3 -c "import json; d=json.load(open('$json')); print(d['gen'])")
        restart=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('restart', 0))")
        printf "%-30s %10s %8s %10s\n" "$name" "$reward" "$gen" "$restart"
    done
done

echo ""
echo "=== Log files for convergence plots ==="
find "$SIMDIR/results/" -name "tune_gait*.jsonl" 2>/dev/null | sort

echo ""
echo "=== Slurm logs ==="
ls "$SIMDIR/slurm_logs/"*.out 2>/dev/null | wc -l | xargs -I{} echo "{} output logs"
ls "$SIMDIR/slurm_logs/"*.err 2>/dev/null | wc -l | xargs -I{} echo "{} error logs"
