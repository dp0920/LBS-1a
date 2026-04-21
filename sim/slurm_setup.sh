#!/usr/bin/env bash
# Run this ONCE on the cluster to create the venv and install dependencies.
# Usage: bash slurm_setup.sh
set -e

module load python/3.10.4

VENV_DIR="$HOME/LBS-1a/sim/venv_cluster"

if [ -d "$VENV_DIR" ]; then
    echo "venv already exists at $VENV_DIR"
else
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install mujoco cma scipy numpy
echo ""
echo "=== Installed packages ==="
pip list | grep -iE "mujoco|cma|scipy|numpy"
echo ""
echo "Done. Activate with: source $VENV_DIR/bin/activate"
