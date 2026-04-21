# Optimus Primal — MuJoCo Gait Simulation

## Local Development

### Prerequisites
- Python 3.10+
- MuJoCo (`pip install mujoco`)
- CMA-ES (`pip install cma`)
- scipy, numpy

### Quick Start

```bash
# Demo the hand-tuned gait in the viewer
mjpython mujoco_gait.py --demo

# Replay a trained gait
mjpython mujoco_gait.py --replay best_gait.json

# Train locally with CMA-ES (default)
python mujoco_gait.py --tune --generations 150

# Train with different algorithm or seed
python mujoco_gait.py --tune --algo de --init random --generations 200
```

### Training Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--algo` | `cma`, `random`, `de` | `cma` | Optimization algorithm |
| `--init` | `gait`, `stand`, `random` | `gait` | Initial seed |
| `--generations` | int | 150 | Number of generations |
| `--popsize` | int | 48 | Population size per generation |
| `--workers` | int | cpu_count - 1 | Parallel rollout workers |
| `--cycles` | int | 5 | Gait cycles per evaluation |
| `--resume` | flag | - | Resume from existing best JSON |

### Other Tools

```bash
# 3-leg balance tuning
mjpython mujoco_balance.py --tune --lift FL
mjpython mujoco_balance.py --replay best_balance_FL.json

# Center of mass calculator (for body layout)
python com_calculator.py
```

## Cluster (Slurm)

### First-Time Setup

```bash
module load python/3.10.4
python3 -m venv ~/LBS-1a/sim/venv_cluster
source ~/LBS-1a/sim/venv_cluster/bin/activate
pip install --upgrade pip
pip install mujoco==3.1.6 --only-binary=:all:
pip install cma scipy numpy
```

Or just run:
```bash
bash slurm_setup.sh
```

### Launch Training Jobs

```bash
# Submit all 9 combos (3 algos x 3 seeds) at one or more epoch counts
bash launch_all.sh 50 200 1000 2500
```

This submits jobs to the `batch` partition (16 cores, 24hr limit each).
Each job writes results to `results/<algo>_<init>/`.

### Monitor

```bash
squeue -u $USER
```

### Collect Results

```bash
# Summary table of best rewards
bash collect_results.sh

# Generate convergence plots
python3 plot_results.py
```

Plots are saved as PNG files:
- `convergence_all.png` — all runs on one chart
- `convergence_seed_<init>.png` — per seed, comparing algorithms
- `convergence_algo_<algo>.png` — per algorithm, comparing seeds

## Loading a Gait onto the Robot

```bash
# Copy the best gait JSON to the robot
scp best_gait.json admin@169.254.1.2:LBS-1a/robot/

# On the robot — cautious first run (2x slower)
cd ~/LBS-1a/robot
python gait_controller.py --gait best_gait.json --n=3 --slow

# Full speed with distance measurement
python gait_controller.py --gait best_gait.json --n=10
```

## File Overview

| File | Purpose |
|------|---------|
| `mujoco_gait.py` | Main gait training script (CMA-ES, DE, random search) |
| `mujoco_balance.py` | 3-leg balance tuning |
| `optimus_primal.urdf` | Robot model |
| `com_calculator.py` | Center of mass layout tool |
| `launch_all.sh` | Submit all training jobs to slurm |
| `train_job.sbatch` | Single slurm job template |
| `slurm_setup.sh` | One-time cluster venv setup |
| `collect_results.sh` | Summarize training results |
| `plot_results.py` | Generate convergence plots |
| `slides.tex` | Beamer presentation |
| `slides.md` | Slide content (markdown version) |
