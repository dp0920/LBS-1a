# Trained policies — directory layout

PPO policy `.zip` files and their `_vecnormalize.pkl` companions are
gitignored (sizes add up to ~50 MB). The directory structure is the
durable record of what was trained — keep it organized, prune freely.

## Subfolders

| Folder | What's in it | Why kept |
|---|---|---|
| `champions/` | The policies referenced as "winners" in METHODS.md §4.5–§4.12 | Fast access for replay, eval, deployment |
| `ablations/` | v23 single-feature ablations (§4.12) and v25 confirmation runs (when complete) | Source of the "base reward wins" finding |
| `random_sweep/` | 52 random-config policies + their results JSONs | Reproduces §4.10 analysis |
| `hybrid/` | v24 BC + PPO fine-tune experiments (§4.11) | Hybrid CMA→RL approach |
| `legacy/` | v3–v18 era and other one-off experiments | Mostly buggy era (§4.8); kept for narrative reference but not load-bearing |

## Champions (the policies actually worth replaying)

| Policy | Mean dist | Survived | Use case |
|---|---|---|---|
| `ppo_v23_base.zip` | 68.49 m | 44/50 (88%) | Current best — minimal reward, "shaping is bad" demo |
| `ppo_v20.zip` | 63.92 m | 33/50 (66%) | Hand-iterated full reward, "speed champion" |
| `ppo_v21.zip` | 50.39 m | 36/50 (72%) | Survival champion (drop-gait variant of v20) |
| `ppo_v6_trig.zip` | 43.40 m | 32/50 (64%) | First "real walker" before §4.6 plateau |
| `ppo_v5.zip` | 27.87 m | 44/50 (88%) | Earliest stable RL walker (`ctrl_repeat=4`) |

Replay one in the viewer:
```bash
mjpython train_ppo.py --replay models/champions/ppo_v20.zip
```

Run 50-episode stats:
```bash
python ppo_stats.py --policy models/champions/ppo_v23_base.zip --episodes 50 --fall-tilt 30
```

## Random sweep (52 configs)

Per-config results JSONs are tracked in git (small, useful for
reproducing §4.10 analysis):
- `random_sweep/random_sweep_results.json` (12 configs, seed 42)
- `random_sweep/random_sweep_results_2.json` (40 configs, seed 43)

The 52 `.zip` policies they reference are gitignored. To regenerate
analysis from JSONs alone:
```bash
python analyze_sweep.py models/random_sweep/random_sweep_results*.json --top 10
```

## Adding new trainings

By default `train_ppo.py --out X.zip` writes to the sim/ root. To keep
things organized, use a path explicitly:
```bash
python train_ppo.py ... --out models/ablations/ppo_v26_my_experiment.zip
```

When an experiment converges into a "champion" worth replaying often,
move it to `champions/` and add a row to the table above.
