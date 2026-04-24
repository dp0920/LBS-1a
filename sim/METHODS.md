# Methods — Gait Learning for Optimus Primal

A reproducibility log of how the trained gaits in this repo were produced. Covers the physical system, the sim model, the parameterization, the reward function (and how it evolved), the training pipeline, and the final hyperparameter sweep.

For day-to-day commands, see `README.md`. This file is the *why*.

---

## 1. Physical system

**Robot:** Optimus Primal — 4-leg quadruped with 2 DoF per leg (hip + knee).

**Actuators:** LX-16A serial bus servos, 8 total (hip/knee × FL/FR/RL/RR).

**Mass:** 1.292 kg as-measured 2026-04-23 after body reprint. CoM is centered (verified by balance test with legs attached). See `com_calculator.py` for the layout-optimization tool used when planning the reprint.

**Prior state:** Before the 2026-04-23 reprint, the CoM was rear-biased (Pi + buck converter mounted far back). The reprint rearranged components to balance the chassis, which removes a major sim-to-real discrepancy.

**Hand-coded reference gait** (`robot/stance.py`): a 13-phase crawl — start pose, then for each of the 4 legs: shift weight → swing forward → plant. Used as the seed for optimization.

---

## 2. Sim model

**Simulator:** MuJoCo ≥ 3.2, built from `optimus_primal.urdf` via the `MjSpec` declarative API (see `build_model()` in `mujoco_gait.py`).

**URDF inertial model** (updated 2026-04-23 to match real mass of 1.292 kg):

| Link | Mass | Notes |
|---|---|---|
| `base_link` | 0.672 kg | Chassis + electronics + battery, centered CoM |
| `upper_link_*` ×4 | 0.085 kg | Hip servo + bracket |
| `lower_link_*` ×4 | 0.065 kg | Knee servo + foot bracket |
| `foot_*` ×4 | 0.005 kg | TPU sock, sphere collision |

Inertia tensors scaled proportionally with mass; `balanceinertia="true"` corrects residuals.

**Actuators:** MuJoCo position servos with `kp=2.5`, `kv=0.05` — a light PD tuning chosen to approximately match LX-16A compliance at the gait's typical velocities.

**Floor:** TPU sock friction model (`friction=1.0 0.01 0.002`, `condim=4`) — grippy and soft-contact.

---

## 3. Parameterization (search space)

**105 parameters:**
- 104 joint angles: 13 phases × 4 legs × 2 joints (hip + knee), in degrees.
- 1 phase time (seconds between phases) — controls cadence.

**Bounds:**
- Hip: `[5°, 55°]`
- Knee: `[-100°, -25°]` (negative = bent)
- Phase time: `[0.15 s, 1.2 s]`

**Why this parameterization:** Gait *structure* (13 phases, stride order `FR → RL → FL → RR`) is fixed by the hand-coded seed; the optimizer tunes angles and cadence, not sequencing. This constrains the search to a well-shaped region where gen-1 candidates already walk.

**Sim-to-real is a parameter copy.** Angles use the same degree convention as `leg_abs()` on the real robot. A winning `best_gait.json` loads directly onto hardware via `gait_controller.py --gait path.json`, with no coordinate transform.

---

## 4. Reward function

The reward is a scalar-sum shaping function evaluated over a full rollout (N cycles of the 13-phase gait). It has three classes of terms: forward-motion rewards, stability penalties, and hard terminations.

### 4.1 Current form

```
reward = +100 * forward_distance
         +1000 * (forward_distance / elapsed_time)      # speed bonus
         +0.5 * phases_completed                         # survival bonus
         +5 * mean_body_height                           # stay upright
         -200 * backward_distance
         -tilt_scale * (60*mean(pitch²) + 80*mean(roll²))
         -tilt_scale * 16*(max(pitch²) + max(roll²))     # peak symmetric tilt
         -tilt_scale * (800*mean(nose_down²) + 250*max(nose_down²))   # forward-flop
         -tilt_scale * (8*mean(pitch_rate²) + 3*max(pitch_rate²))     # fast-flop detection
```

**Hard terminations** (rollout ends, loses future reward):
- `body_z < 8 cm` — fallen
- `|pitch|` or `|roll| > fall_tilt_deg` — tipped (default 20°, swept in experiments)
- `nose_down > sin(fall_tilt_deg)` — forward flop specifically

### 4.2 Reward evolution (why it looks the way it does)

**v1: distance only.** Robot learned to collapse onto its shins and scoot forward. → Added tilt penalties + fall-height kill.

**v2: symmetric pitch/roll penalty + FALL_TILT 45° → 30°.** Robot balanced but still toppled forward on front legs during swing phases. The symmetric penalty didn't distinguish "nose-up lean" (sometimes useful for weight-shift) from "nose-down flop" (failure mode). → Added directional penalties.

**v3: directional nose-down penalty + pitch-rate penalty.** `nose_down = max(0, -body_forward_z)` isolates the forward-flop failure mode. Pitch-rate catches fast rotational motions that don't linger long enough to show up in the mean-tilt term (the actual "flop" moment). → Robot stayed level.

**v4 (current): speed bonus + tighter kill thresholds + tilt_scale multiplier.** Adding `1000 * distance/time` made the optimizer prefer faster cadence; phase_time dropped from ~1.0s to ~0.5s in early tests. Kill thresholds tightened to 20° and 20° nose-down. `tilt_scale` lets sweep experiments dial global penalty strictness.

### 4.3 Sampling

Attitude (`pitch`, `roll`, `nose_down`, `pitch_rate`) is sampled per **physics step**, not per phase, so penalties integrate over the whole trajectory. `pitch_rate` is taken from `data.qvel` directly (body frame).

---

## 5. Optimization algorithms

Three algorithms implemented; all share the parallel rollout infrastructure:

| Algo | Library | Mechanic | Use |
|---|---|---|---|
| **CMA-ES** | `cma` | Learns covariance structure; auto-restarts on 15-gen stall | Primary workhorse; best results |
| Random search | `numpy` | Uniform sampling in bounds + elitism | Baseline — is CMA doing real work? |
| Differential Evolution | `scipy.optimize.differential_evolution` | Population-based mutation via linear combinations | Alternative evolutionary method |

**CMA-ES hyperparameters:**
- `popsize = 48` (fixed)
- `sigma_init` ∈ `[2, 15]` (swept; see §7)
- Auto-restart: on 15-gen stall, re-init at current best, σ ×= 1.3, popsize += 8
- Bounds enforced by the CMA library itself

**Parallelism:** `multiprocessing.Pool(cpus_per_task - 1)` — each worker builds its own `MjModel`. Workers ignore `SIGINT` so `ctrl-C` reaches only the parent (fixes a deadlock where all workers died at once inside C code).

---

## 6. Evaluation

**Per-candidate evaluation:** N cycles of the 13-phase gait (default N=5 during training, N=10 for ranking). Earned distance is summed across completed steps at each plant phase.

**Fall detection:** If the robot falls mid-rollout, the rollout exits early. Survival bonus (`0.5 × phases_completed`) gives partial credit, preventing CMA-ES from discarding almost-working gaits outright.

**Ranking rollouts** (`rank_gaits.py`): re-runs each saved `best_gait.json` for N cycles and sorts by distance, speed, or training reward. This is where you pick the winning gait for deployment — training reward is noisy because it's shaped; distance-in-sim is the ground truth.

---

## 7. Final hyperparameter sweep

A deliberately diverse sweep to find winning settings. All randomized dimensions are drawn per-job and recorded inside each `best_gait.json` under `config`.

**Swept dimensions (per-job random):**
- `fall_tilt_deg` ∈ `[0, 40]` (uniform float)
- `tilt_scale` = `20 / fall_tilt_deg`, clamped to `[0.3, 5]` (strict tilt threshold ↔ stricter soft penalty)
- `sigma_init` ∈ `[2, 15]` (uniform int, CMA only)

**Gridded dimensions:**
- `algo` ∈ `{cma, random, de}`
- `init` ∈ `{gait, stand, random}`
- `interp` ∈ `{linear, cosine, smoothstep}` — joint-angle interpolation between phases
- `generations` ∈ `{200, 1000, 2500}` — short to long training budgets

**Trials:** 10–15 random draws per gridded cell (so each combo has multiple tilt/sigma samples).

**Why these were chosen:**
- *tilt:* stricter thresholds force the optimizer to find genuinely stable gaits; lenient ones may overfit to the physics forgiveness. Range spans "always kill any tilt" to "essentially no kill".
- *interp:* linear interp has velocity discontinuities at phase boundaries, which is exactly what the `pitch_rate` penalty fights. Cosine/smoothstep have zero velocity at endpoints — should be easier to optimize.
- *sigma:* σ=2 fine-tunes near the seed; σ=15 broadly explores. Without sweeping, the optimizer only ever tries one kind of search.
- *algo/init:* baselines to confirm CMA + `init=gait` dominates.

**Output layout:**
```
results/<timestamp>/<algo>_<init>/<interp>/t<tilt>_s<sigma>_trial<N>/<gens>/
    best_gait.json       # winning params + config
    tune_gait.jsonl      # per-generation log (best/mean/pop_best)
```

---

## 8. Reproduction — end-to-end

### Local dev
```bash
pip install "mujoco>=3.2" cma scipy numpy matplotlib
mjpython mujoco_gait.py --demo                            # hand-coded gait
python3 mujoco_gait.py --tune --generations 150 --algo cma # local training
mjpython mujoco_gait.py --replay best_gait.json           # watch result
```

### Cluster sweep (Slurm)
```bash
bash slurm_setup.sh    # one-time venv setup
N_TRIALS=10 bash launch_all.sh 200 1000 2500
# → 3 × 3 × 3 × 3 × 10 = 810 jobs, results under results/<timestamp>/
```

### Analysis
```bash
# Rank every gait in a sweep by walking distance
python3 rank_gaits.py results/<timestamp> --sort speed --top 20

# Convergence plots (reward vs generation)
python3 plot_results.py

# Full telemetry for one gait
python3 record_rollout.py results/<timestamp>/cma_gait/cosine/t07.13_s09_trial4/2500/best_gait.json
```

### Deploy to robot
```bash
scp best_gait.json admin@169.254.1.2:LBS-1a/robot/
ssh admin@169.254.1.2
cd LBS-1a/robot && python gait_controller.py --gait best_gait.json --n=3 --slow
```

---

## 9. Known limitations

- **Open-loop.** The gait is a fixed sequence of target angles — no sensing, no feedback, no balance correction. A nudge mid-gait that the hand-coded kill thresholds haven't trained against will knock the robot over. This is the motivation for moving to RL (PPO + closed-loop policy).
- **Sim-to-real gap.** URDF inertias are coarse estimates, not measured. Servo compliance (`kp=2.5, kv=0.05`) is approximate. Foot friction assumes hard floor; carpet/tile vary. Gaits that look stable in sim sometimes still flop on real hardware due to one of these — tightening the reward kills (§4) mitigates but doesn't eliminate.
- **Fixed gait structure.** Stride order is hard-coded as `FR → RL → FL → RR`. The optimizer cannot discover that a trot or bound might be better for this robot.
- **Deterministic rollout.** Each fitness evaluation is noise-free (no random perturbation, always same start pose). Gaits may overfit to the one sim scenario. Domain randomization is a future improvement.

---

## 10. Change log

| Date | Change |
|---|---|
| 2026-04-23 | Refactored: `sim_core.py` (shared plumbing) + `reward.py` (`RewardAccumulator`) + `gym_env.py` (RL skeleton) |
| 2026-04-23 | URDF mass updated to 1.292 kg / centered CoM after robot reprint |
| 2026-04-23 | Added `sigma_init` sweep `[2, 15]` (int) |
| 2026-04-23 | Tightened nose-down kill to 20°, doubled baseline tilt/flop/pitch-rate penalties |
| 2026-04-23 | Added speed bonus (`+1000 * distance/time`) to reward |
| 2026-04-22 | Added `--interp` sweep (linear / cosine / smoothstep) |
| 2026-04-22 | Added random-tilt sweep `[0, 40]°` with auto-derived `tilt_scale = 20/tilt` |
| 2026-04-22 | Added `rank_gaits.py` to rank by walking distance (not just reward) |
| 2026-04-22 | Reorganized results to `results/<timestamp>/<algo>_<init>/<gens>/` |
| 2026-04-21 | Directional nose-down + pitch-rate penalties; FALL_TILT 30° → 25° |
| 2026-04-21 | Fixed ctrl-C hang (workers ignore SIGINT, use `map_async().get(timeout)`) |
| 2026-04-21 | Added `--algo` (cma/random/de) and `--init` (gait/stand/random) sweep support |
| 2026-04-21 | Added Slurm pipeline (`launch_all.sh`, `train_job.sbatch`, `slurm_setup.sh`) |
