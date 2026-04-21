#!/usr/bin/env bash
# Collect and summarize results from training runs.
#
# Usage:
#   bash collect_results.sh                          # latest run
#   bash collect_results.sh 20260421_143022          # specific run
#   bash collect_results.sh all                      # all runs

SIMDIR="/cluster/home/dparri03/robotics/LBS-1a/sim"

if [ "$1" = "all" ]; then
    RUNS=$(ls -d "$SIMDIR/results"/*/ 2>/dev/null | sort)
elif [ -n "$1" ]; then
    RUNS="$SIMDIR/results/$1"
else
    # Latest run
    RUNS=$(ls -dt "$SIMDIR/results"/*/ 2>/dev/null | head -1)
fi

for RUN in $RUNS; do
    [ -d "$RUN" ] || continue
    echo "=== Run: $(basename "$RUN") ==="

    # Show manifest if present
    [ -f "$RUN/manifest.txt" ] && cat "$RUN/manifest.txt" && echo ""

    printf "%-35s %10s %8s %10s\n" "Config" "Reward" "Gen" "Restarts"
    printf "%-35s %10s %8s %10s\n" "------" "------" "---" "--------"

    for config_dir in "$RUN"/*/; do
        [ -d "$config_dir" ] || continue
        config_name=$(basename "$config_dir")
        for gens_dir in "$config_dir"/*/; do
            [ -d "$gens_dir" ] || continue
            gens=$(basename "$gens_dir")
            for json in "$gens_dir"best_gait*.json; do
                [ -f "$json" ] || continue
                reward=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['reward']:+.2f}\")")
                gen=$(python3 -c "import json; d=json.load(open('$json')); print(d['gen'])")
                restart=$(python3 -c "import json; d=json.load(open('$json')); print(d.get('restart', 0))")
                printf "%-35s %10s %8s %10s\n" "${config_name}/${gens}" "$reward" "$gen" "$restart"
            done
        done
    done
    echo ""
done

echo "=== Log files ==="
find "$SIMDIR/results/" -name "tune_gait*.jsonl" 2>/dev/null | sort
