#!/usr/bin/env bash
set -e
for leg in FL FR RL RR; do
  echo "=== Tuning $leg ==="
  mjpython mujoco_balance.py --tune --lift "$leg"
done
echo "=== All done ==="
