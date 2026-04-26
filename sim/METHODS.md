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

### 4.4 RL dense reward (`gym_env.py`)

The RL environment (`OptimusPrimalEnv`) uses a **dense per-step** reward rather than the one-shot rollout reward of §4.1. PPO needs a signal every physics step, not once per rollout, or the advantage estimator has nothing to work with.

```
reward = step_reward(dx)            # forward progress + attitude cost, per step
         + 2.0                       # alive bonus (every step, only if not fallen)
         + gait_reward               # contact-pattern match, scaled ±1/step (v4)
         + 5.0 · max(0, qvel[0])    # speed bonus — forward velocity (v4)
         + extension_reward          # 3.0 · mean(|a − midpoint| / half_range) (v5)
         - posture_penalty           # linear outside z ∈ [0.12, 0.18]
         - smoothness_penalty        # 0.1 · ‖Δaction‖²
         - 20.0 · is_fallen          # one-shot at episode end
```

- **`step_reward(dx)`** (`reward.py::RewardAccumulator.step_reward`): a rescaled version of §4.1 — forward distance with asymmetric backward punishment, plus attitude penalties ~10× smaller because they fire every physics step. Calibrated with `tune_step_reward.py`, which overlays per-step reward against attitude/distance to check that "standing still" ≈ 0 and "walking" is clearly positive.
- **Alive bonus `+2`/step.** Stable-standing nets ≈ 0 after §4.1 penalties; falling forfeits both this bonus for the remainder of the episode and a one-shot `-20` at termination. This makes "stay alive" always dominate "fall fast".
- **Posture penalty.** `10 · max(0, |z − 0.15| − 0.03)` — zero cost inside a 6 cm band around target body height; linear outside. Stops the policy from converging to a fully-extended "giraffe" stance (high z) or a collapsed belly-scoot (low z). Softened from the original `30 ·` multiplier and `±0.02` band once PPO was refusing to leave the band at all in early exploration.
- **Action smoothness.** `0.1 · ‖aₜ − aₜ₋₁‖²`. LX-16A servos have finite bandwidth and a high-frequency wiggle in sim won't transfer. Originally `1.0 ·`, which deterred *any* motion; softened to `0.1 ·` on 2026-04-23.
- **Gait-contact-pattern reward.** Each physics step scores `+1` per leg whose current contact state matches the expected pattern, `-1` per mismatch; max `+4`, min `-4`, scaled by `gait_reward_scale` (**v4 default 0.25**, was 1.0). "Expected" follows `FR → RL → FL → RR` at cadence `phase_period` (**v4 default 4.0 s** → 1 s per leg swing, was 2.0 s / 0.5 s). Two thresholds score contact state: the target leg earns `+1` when foot z > `FOOT_LIFT_Z = 0.015 m`; non-target legs earn `+1` when foot z < `FOOT_PLANT_TOLERANCE = 0.03 m`. The 1.5–3 cm gap is a "free zone" where a non-target leg can briefly unload for weight-shifting without penalty (new in v4). The policy observes `(sin(2πφ), cos(2πφ))` in its state, so it can anticipate which leg *should* be swinging.
- **Speed bonus (v4, parameterized in v6).** `velocity_bonus · f(max(0, qvel[0]))` — explicit per-step positive for forward velocity on top of `step_reward(dx)`. `step_reward` mixes a positive `+500 · max(0, dx)` gain with a harsher `-1000 · max(0, −dx)` backward penalty, so early in training the net signal from noisy per-step dx is often negative. The velocity term is unambiguously positive and gives PPO a clean gradient toward "move faster". v6 parameterizes the shape `f`: `linear` (`v`, v4/v5 default), `quadratic` (`v²`), `cubic` (`v³`, steepest "go big" signal), or `trig` (`sin(v) + 1 − cos(v)`, smooth monotonic super-linear curve). Scalar `velocity_bonus` is tunable via `--velocity-bonus` on `train_ppo.py`.
- **Action extension bonus (v5).** `3.0 · mean(|action − midpoint| / half_range)` — rewards commanded joint angles sitting far from the midpoint of their bound. Normalized per-joint so hip (50° range) and knee (75° range) contribute equally: each joint contributes 0 at dead center, 1 when pinned to a bound. Direction-agnostic — rewards either flex or extend, whichever the policy commits to — so it doesn't fight the posture term. Counters the "jitter at stance" local optimum where the policy hugs the midpoint and makes small oscillations instead of committing to big leg-sweep angles. Watching the v4 replay, the robot was taking tiny steps; v5 visibly swings its legs through full arcs.

**Why a forced contact-pattern reward?** Without it, PPO rapidly finds a local optimum where the robot just stands still (the alive bonus is "free" once posture is satisfied). Shaping the stride order into the reward converts the coordination problem ("discover that legs must alternate") into a tracking problem ("lift this leg at this phase"). The chosen `FR → RL → FL → RR` order matches the hand-coded `stance.py` seed and the CMA-ES gait structure (§3), and it rides the rear-biased CoM on the real robot — lifting a front leg first with rear legs planted is the most stable option.

**Why not let PPO discover the order?** Same reason as §3: with an 8-dim continuous action space, 24-dim observation space, and sparse fall signals, gait emergence is extremely slow. Forced ordering is the RL analogue of CMA's fixed 13-phase structure — both commit to the crawl as the gait family and optimize within it.

### 4.5 RL reward evolution

Configurations trained at 1 M timesteps, 4 parallel envs, default PPO hyperparameters. Each row's behavior stats come from `ppo_stats.py --episodes 50` (stochastic) *except* v1 and v2 where the old obs shape (22-dim) no longer loads — those numbers are from TensorBoard at end of training.

**Note on episode length across versions:** v1–v5 ran with `ctrl_repeat=4` (50 Hz control, 2000-step cap ≈ 40 s sim). v6 switched to `ctrl_repeat=8` (25 Hz, closer to real LX-16A command rate; 2000-step cap ≈ 80 s sim). Absolute distance numbers are therefore not directly comparable across the v5 / v6 boundary — use "Survived", which is a fraction of full-episode completions, as the consistency metric.

| Ver. | Change | ep_len (TB) | ep_rew (TB) | Mean dist | Max dist | Survived |
|---|---|---|---|---|---|---|
| v1 | `step_reward + alive + fall penalty` (reward-only baseline, ran only ~508k steps) | ~1373 | ~4145 | — | — | — |
| v2 | + posture + action-smoothness penalties | ~698 | ~2323 | — | — | — |
| v3 | + forced gait-order reward (phase=2 s, scale=1.0) | 379 | 943 | 2.93 m | 6.01 m | 1/50 (stood still, 0.63 m) |
| v4 | phase=4 s, scale=0.25, free-zone 1.5–3 cm for non-target feet, `+5·qvel[0]` speed bonus | 461 | 4,010 | 4.28 m | 14.55 m | 0/50 (every episode a genuine walk) |
| **v5** | + action-extension bonus `3.0 · mean(|a−mid|/half_range)` | **1,896** | **23,700** | **27.87 m** | **31.19 m** | **44/50** |
| v6_linear | ctrl_repeat 4→8, parameterized velocity shape (coef=5) | 1,589 | 34,885 | 38.99 m | 55.97 m | 27/50 |
| v6_quadratic | velocity `5·v²` | 1,128 | 22,942 | 23.12 m | 55.18 m | 12/50 |
| v6_cubic | velocity `5·v³` | 1,490 | 45,212 | 38.58 m | **66.33 m** | 22/50 |
| **v6_trig** | velocity `5·(sin(v) + 1 − cos(v))` | 1,542 | 38,217 | **39.31 m** | 59.13 m | **29/50** |

**Story in one sentence:** each shaping step was principled — v2 added the penalties needed to eliminate v1's degenerate collapse-and-scoot, v3 added gait order to escape the "stand still" local optimum, v4 relaxed v3's over-constraint to unlock genuine walking, v5's action-extension bonus pushed the policy past low-amplitude jitter into full-stride leg swings, and v6's velocity-shape sweep tested how strongly to pull for speed — finding that the *least extreme* shapes (linear, trig) outperform the steeper ones (quadratic, cubic) on consistency, with trig a hair ahead on survival.

**What each version taught the next:**
- **v3 → v4:** `phase_period = 2.0 s` (0.5 s per leg) was too fast — no time to pre-shift weight before a commit swing. `gait_reward_scale = 1.0` (±4/step) dwarfed the +2 alive bonus. The single `FOOT_LIFT_Z = 0.015 m` threshold penalized weight-shifting micro-lifts. And "move forward" was only signalled through `step_reward(dx)`'s noisy per-step term.
- **v4 → v5:** v4 walked but in tiny, jittery steps — watching the replay showed the policy hugging the midpoint of its action range and making small oscillations rather than sweeping leg arcs. An action-extension bonus (reward being far from midpoint, direction-agnostic) pushed the policy to commit to full-stride poses. That single term produced a 6.5× increase in mean walking distance and a 45× jump in full-episode survival rate.
- **v5 → v6:** v5 walked but at 50 Hz control, which is above the LX-16A servo's reliable command rate. Doubling `ctrl_repeat` (50 Hz → 25 Hz) halves the number of motor commands per meter, closer to real-hardware constraints. The velocity-shape sweep tested how hard to pull for speed — turns out gently. Steeper shapes (cubic) gave higher peaks but more catastrophic failures (one run did a −3.73 m backwards faceplant); quadratic sat in an unstable valley between linear and cubic; trig's smooth super-linear curve edged linear on survival.

**Velocity-coefficient sweep (v6_linear family, 4 points):** separately testing whether the `velocity_bonus` scalar itself matters, independent of shape. All runs use `velocity_shape=linear`, 1 M timesteps, otherwise-identical v6 config.

| coef | Min dist | Mean dist | Max dist | Survived |
|---|---|---|---|---|
| 0.1 | 5.54 m | 33.77 m | 56.78 m | 20/50 |
| 1.0 | 0.73 m | 35.73 m | 58.10 m | 22/50 |
| **5.0** | 4.64 m | **38.99 m** | 55.97 m | **27/50** |
| 10.0 | 1.81 m | 29.06 m | **60.28 m** | 11/50 |

Peaked curve around **coef=5**. Below 5, the policy still walks — because `step_reward(dx)`'s `+500·dx` term is already doing most of the forward-drive work — but slightly less consistently. Above 5, the same dynamic as the cubic-shape run: the velocity term becomes dominant enough that the policy over-commits to peak speed (max distance climbs to 60.28 m) but survival collapses to 11/50. **Shape matters more than magnitude, but within the linear shape there's still a clean optimum at coef ≈ 5.**

### 4.6 v7–v14: hitting the reward-engineering ceiling

After v6_trig (32/50 survived, 43.4 m mean), we attempted several more reward-shaping interventions to beat the baseline. **Every one of them regressed or broke even.** The story of this section is about what DIDN'T work and what those failures taught us — this is honest iteration data, not a success narrative.

**Methodology note — fall-tilt detector fix.** Before running these comparisons we discovered the default `fall_tilt_deg=20°` kill threshold was over-triggering during normal walking motions. A crawling quadruped hits 15–25° tilt transiently during weight shifts without actually falling. All v7+ evaluation in this section uses `ppo_stats.py --fall-tilt 30` for apples-to-apples comparison. The training-time detector was left at 20° (matches what the older policies were trained under); the post-hoc eval uses the looser threshold to remove spurious false-FELL calls.

**Also important — `ctrl_repeat` mismatch.** v5 was trained at `ctrl_repeat=4` (50 Hz); v6+ switched to `ctrl_repeat=8` (25 Hz, matching LX-16A realistic command rate). A policy evaluated at the wrong `ctrl_repeat` collapses entirely — v5 dropped from 45/50 to 4/50 when evaluated at 8. `ppo_stats.py --ctrl-repeat <N>` must match training.

**The experiments (v7 → v14):**

| Ver. | Change vs. v6_trig | Mean dist | Survived | Notes |
|---|---|---|---|---|
| **v6_trig** (baseline) | — | **43.4 m** | **32/50** | unchanged champion |
| v7 | 27-dim IMU obs (removed `body_z`, `qvel[0]`; added gyro+accel) + LR decay | 9.9 m | 15/50 | entropy collapse + lost height signal |
| v7_ent | v7 + `ent_coef=0.01` | 16.75 m | 27/50 | entropy bonus partially recovered; obs change still hurts mean dist |
| v8 | `stride_bonus=10` + `ent_coef=0.01` | 17.3 m | 8/50 | reward hacking: policy crashed feet forward for max single-plant bonus |
| v8_nogait | v8 + `gait_reward_scale=0` | 13.6 m | 9/50 | forced gait order found NOT load-bearing (within noise of v8) |
| v9 | `randomize_init` + `dynamic_posture_target` + stride_bonus=10 | 20.0 m | 7/50 | inherited v8's stride_bonus=10 poison |
| v10 | `weight_transfer_bonus=2.0` (v6_trig + one term) | 31.6 m | 26/50 | weight-transfer concept didn't help alone |
| v11 | `stride_bonus=1.5` alone (lower scale than v8) | 30.7 m | 13/50 | smaller stride bonus less catastrophic but still regression |
| v12 | weight_transfer=2 + stride=1.5 combined | _pending_ | _pending_ | testing synergy |
| v13 | weight_transfer=2 + domain randomization | _pending_ | _pending_ | DR for robustness |
| v14 | weight_transfer=4.0 (coef sweep) | _pending_ | _pending_ | is more better? |

**Five findings worth carrying into any next iteration:**

1. **v6_trig is the plateau at 1 M timesteps.** Each of 7 single-variable changes (4 new reward terms + 2 env changes + 1 hyperparam) regressed the combined metric. Reward engineering alone is not going to push past this.
2. **Single-event bonuses are reward-hackable.** The stride bonus fires once per plant with magnitude `k·displacement` — at `k=10`, a single 2 m stride was worth +20, enough to dominate the rest of the reward. Policy learned "crash foot forward" to farm that bonus once, then gave up. At `k=1.5` the hack is less extreme but still present.
3. **The forced gait-order reward (v3) is NOT load-bearing anymore.** v8_nogait (gait_reward_scale=0) is statistically indistinguishable from v8. By the time you have posture + extension + velocity + stride shaping, PPO converges to a walking-like gait without the explicit FR→RL→FL→RR cadence. Simpler story for the slide: "we tried letting PPO discover its own gait — it did."
4. **The weight-transfer concept is conceptually right but didn't ship.** Rewarding low ground-contact force on the target swing leg — the pre-stride weight-shift behavior real quadrupeds use — is biomechanically principled. But at coef=2.0 it didn't beat the baseline. Needs more tuning or a different formulation (maybe reward CoP within stance triangle instead of per-leg force).
5. **The false-FELL bug masked real behavior.** Running stats with `fall_tilt=20` (training threshold) was terminating episodes during aggressive-but-valid walking motions. All policies benefited from the looser eval threshold, but the ranking stayed roughly the same — suggesting the training-time kill signal was correct for *learning* while mismatched for *evaluation*.

**Path forward (v15+):** reward engineering hit diminishing returns at v6_trig. Next productive step is **longer training** (3 M timesteps) on the top-performing configurations to see if the plateau is a local optimum or a training-budget ceiling. See §4.7 (if / when populated).

**TensorBoard cross-reference.** All runs are in `ppo_tb/<tb_name>_N/` where `N` is SB3's auto-increment (usually `_1`). To overlay any subset for comparison:
```bash
tensorboard --logdir ./ppo_tb
```
Run names, chronologically: `PPO_1`, `PPO_2` (early v1/v2 exploratory), `v3_1`, `v4_1`, `v5_1`, `v6_linear_1`, `v6_quadratic_1`, `v6_cubic_1`, `v6_trig_1`, `v6_linear_0p1_1`, `v6_linear_1p0_1`, `v6_linear_10p0_1`, `v7_1`, `v7_ent_1`, `v8_1`, `v8_nogait_1`, `v9_1`, `v10_1`, `v11_1`, `v12_1`, `v13_1`, `v14_1`, `v15_1`, `v16_1`, `v17_1`, `v18_1`.

### 4.7 Finding the reward's "base vectors" — correlation + PCA

By v14 we had ~10 per-step reward components, each with its own coefficient. Iteration had plateaued. The instinct at that point is to add an 11th term or tune a coefficient more carefully — but **a 22-dimensional reward-knob space is not feasibly tunable by hand**, and we had no principled way to decide which of the 10 terms were still earning their keep.

We turned the question around: **which components are independent signals, and which are duplicating work or sitting constant?** Linear-algebra diagnosis answers this directly.

**The approach:**

1. Roll out a trained policy; record every reward component at every timestep. Our `env.step()` returns an `info["reward_components"]` dict:
   ```python
   {"alive": ..., "step_reward": ..., "gait": ..., "velocity": ...,
    "extension": ..., "stride": ..., "weight_transfer": ...,
    "posture": ..., "smoothness": ..., "fall": ...}
   ```
2. Stack into a `T × N` matrix (T = timesteps in the rollout, N = components).
3. Standardize per column (subtract mean, divide by std) so high-magnitude terms don't dominate.
4. Compute the correlation matrix `C = Corr(X)` — `N × N`.
5. Eigendecompose `C = U · diag(λ) · Uᵀ`.

**What the output tells you:**
- **Pairs with |corr| > 0.7** — redundant. Drop one, or merge them into a single term.
- **Eigenvectors** — the "base vectors" of the reward space. The top ones are real independent signals the policy is responding to.
- **Eigenvalues** — variance along each base direction. Small eigenvalues flag dimensions that are near-constant or noise.

Implementation: `reward_pca.py`. Takes a policy, rolls out N episodes, prints the correlation matrix + ranked eigenvectors with loadings. Pass the training-time reward coefs via CLI so the eval env matches what the policy learned against.

**Findings from the v12 champion:**

| Discovery | What it told us |
|---|---|
| `step_reward` ↔ `velocity_bonus`: **corr = 0.93** | Both measure forward motion (`+500·dx` vs `+5·v²`). Redundant — drop `velocity_bonus`. |
| `gait_reward`: **std ≈ 0.008 per step** | With gait_reward_scale=0.25, the term barely deviates from its mean. Contributing near-zero gradient signal. Drop. |
| `extension`, `smoothness`, `posture` load on separate PCs | Independent signals, each doing real work. Keep. |
| Top 2 PCs capture 70% of variance; PCs 6+ have ≈0 variance | Real reward dimensionality is ≤ 5, not 10. |

**Gotcha — measurement artifact to watch for.** Our first-pass PCA showed perfect ±1 correlations between `alive`, `fall`, `gait`, and `weight_transfer`. That wasn't real reward structure — it was because we were gating all four on `not fell` in the info-dict reporting (so they all zeroed together on episode end). Fix: report each term's computed value *before* the fall-gate in the diagnostic trace. The gating still happens in the scalar reward returned to SB3, but the trace stays faithful to the underlying signal.

**Ablation follow-ups driven by PCA findings:**
- v16: v12 config minus `velocity_bonus` → tests "drop 0.93-correlated redundant term"
- v17: v12 config minus `gait_reward` → tests "drop low-variance dead weight"
- v18: minimal-reward v15 + stride_bonus → tests "is stride adding speed to the barebones?"

Results in §4.8 (populated once runs complete).

**The narrative — "start wide, analyze, narrow":**
1. **Start wide** (v3→v14): add reward components reactively, each justified by a replay failure mode.
2. **Plateau.** More reward terms stop helping or actively hurt (v7–v14 in §4.6).
3. **Analyze instead of adding.** Correlation matrix + PCA tell you which terms are doing duplicate work, which are dead, and which are the real load-bearing signals.
4. **Narrow.** Drop redundant/dead components, keep the base vectors. Re-train and verify with ablation runs.
5. **Result:** fewer knobs, each empirically load-bearing, equal or better performance with a simpler reward to explain and tune.

The broader lesson: **reward engineering drifts toward over-specification** because every added term fixes a *specific* replay failure, and you rarely audit whether a previously-added term is still earning its keep after the policy learned around it. A periodic correlation/PCA pass is a cheap audit that surfaces exactly that drift.

### 4.8 The silent-signal bug — "always verify your reward actually fires"

While running the v16/v17/v18 ablation follow-ups, the correlation PCA showed `stride_reward` as always-zero in the trace even though the policy was nominally trained with `stride_bonus=1.5`. That was the thread that unraveled a much bigger finding.

**Root causes, in order of discovery:**

1. **Foot body z-thresholds never fired.** The original foot-contact detection (`fz_body_in_world > FOOT_LIFT_Z = 0.015 m`) assumed body z was ground-referenced. It wasn't — MuJoCo's `data.xpos` is in world coordinates, and the foot body frame sits at ~0.1 m in world space (anatomical foot height, not contact point). So `foot_z > 0.015` was always true for every foot every step.

2. **The URDF `foot_*` body names didn't exist in the MuJoCo model.** MuJoCo's URDF loader merges fixed-joint children into their parent bodies. The URDF-defined `foot_*` links were absorbed into `lower_link_*`, leaving no bodies with the `foot_` prefix. `mj_name2id(model, mjOBJ_BODY, "foot_fr")` returned −1 for every leg, and `data.xpos[-1]` returned the last body's position (the same value every time). All four "foot z" readings were actually the same world-z of `lower_link_rr`.

3. **`data.cfrc_ext` was the wrong signal for ground contacts.** Our weight-transfer reward read `cfrc_ext[foot][5]` assuming it held ground reaction force. It doesn't — `cfrc_ext` is populated only by `mj_applyFT` (user-applied external forces). Ground contacts live in MuJoCo's contact list (`data.contact`), accessible via `mj_contactForce`.

**Implications — how much of v3–v18's reward engineering was real:**

| Reward term | Worked from v3? | Detail |
|---|---|---|
| alive, fall, step_reward(dx), posture, smoothness, extension, velocity_bonus | ✅ YES | These never touched the broken foot logic. |
| **gait_reward** | ❌ silent | Evaluated to constant −0.5 for every step of every v3+ training run. The "forced gait-order" shaping never actually shaped. |
| **stride_bonus** (v8+) | ❌ silent | Plant-event detection never triggered → always 0. |
| **weight_transfer_bonus** (v10+) | ❌ silent | `cfrc_ext` always 0 → unload fraction always 1.0 but the whole term was effectively dominated by constant bias, not signal. |

**What this means for the v8–v18 narrative:** every experiment that we attributed to stride or weight-transfer (v8 / v8_nogait / v9 / v10 / v11 / v12 / v13 / v14 / v15 / v18) was measuring **training-seed variance**, not reward-shaping effects. The "v12 synergy" between stride + weight_transfer that looked like evidence of paired-signal interaction was noise. Same for the "v10 solo regressed but v14 coef=4 improved" coefficient sensitivity claim — random seeds.

**What this means for the v3–v7 and v6 sweeps:** those stand. v3→v5's big improvements were driven by working terms (posture, extension, alive, step_reward). The v6 velocity-shape sweep tested `f(qvel[0])` which works correctly. The v7 IMU obs change was real.

**The fix** (`gym_env.py` 2026-04-24):
- Update `FOOT_BODY_NAMES` to point at `lower_link_*` (the merged parent bodies).
- Introduce `_foot_contact_forces()` — iterates `data.contact`, calls `mj_contactForce`, sums normal force per foot body.
- Replace all z-threshold checks in `_gait_reward`, stride tracking, and weight-transfer with `foot_forces[leg] > FOOT_CONTACT_THRESHOLD = 0.5 N`.
- Verified post-fix: 199 stride events / 500 steps, gait reward varies [−1, +1] with std 0.45 (was 0.008), weight-transfer varies step-to-step.

**The methodology lesson** — this is the punchline for the slide:
> Before trusting a reward term, **print its values during a rollout**. Don't assume that because you added a term to the reward it's contributing signal. A term with `std ≈ 0` across a rollout is either (a) a constant that gradient descent ignores, (b) silently disabled by a bug, or (c) never activated because its triggering condition never fires. All three are invisible to your learning curve — the policy just learns the terms that DO vary, and you credit the whole reward function for the result.
>
> The PCA analysis (§4.7) found this bug because it surfaced exactly this `std ≈ 0` anomaly. Had we not instrumented per-component rewards, we would have shipped a "winning" policy built on signals that never fired and drawn incorrect conclusions about which reward terms matter.

**Relaunched experiments post-fix:** see §4.9 below.

### 4.9 Post-fix sweep — what the rewards actually do

After the silent-signal fix, every reward shaping experiment from v3 onwards needed re-validation since stride / weight_transfer / gait had been disabled or constant. Re-trained the most-cited configurations at 3 M timesteps each with the working signals.

**Apples-to-apples comparison at fall_tilt=30, ctrl_repeat=8, 50-episode stochastic:**

| Run | Config | Mean | Max | Survived | Combined |
|---|---|---|---|---|---|
| **v20** | trig vel=5 + stride=**5** + WT=2 + gait=0.25 | **63.92 m** | **80.84 m** | 33/50 | **42.2** |
| **v21** | trig vel=5 + stride=**5** + WT=2 + gait=**0** | 50.39 m | 70.09 m | **36/50** | 36.3 |
| v19 | trig vel=5 + stride=1.5 + WT=2 + gait=**0** | 52.14 | 76.80 | 26/50 | 27.1 |
| v12_3M_fixed | trig vel=5 + stride=1.5 + WT=2 + gait=0.25 | 49.26 | 68.57 | 24/50 | 23.6 |
| v16_3M_fixed | trig vel=**0** + stride=1.5 + WT=2 + gait=0.25 | 32.28 | 63.64 | 13/50 | 8.4 |
| v6_trig_3M_fixed | trig vel=5 only, no stride/WT | 25.06 | 67.06 | 20/50 | 10.0 |

**Findings that revise §4.6's "v6_trig is the plateau" claim:**

1. **The plateau wasn't real.** v6_trig was hitting 43 m mean / 32 survived at 1 M (in §4.6) entirely on the working terms (alive, step_reward, posture, smoothness, extension, velocity). At 3 M with the *fixed* gait-reward firing, v6_trig actually **regressed** to 25 m / 20 — because the previously-constant gait_reward now imposes a real coordination penalty without complementary stride/WT signals to balance it.

2. **Once stride and weight_transfer actually fire, they help substantially.** v12 vs v6_trig (both with same vel + gait, only stride+WT differing): 49 m vs 25 m mean, 24 vs 20 survived. The "synergy" we thought we saw at 1M was random-seed noise; the real synergy emerges with working signals at 3M.

3. **`stride_bonus = 1.5` was too conservative.** Bumping to 5 (v20) gave +14 m mean and +9 survivals over v12. v12's 1.5 was a defensive response to v8's apparent "reward hacking" — but v8 didn't actually have stride firing, so the hacking was random-seed variance.

4. **Forced `gait_reward_scale=0.25` is now an aggression switch, not a stability shaper.** v20 vs v21 differ only here:
   - **Gait ON (0.25)**: policy commits to FR→RL→FL→RR pattern more rigidly → bigger strides (mean 64 m, max 81 m), but more falls (33/50 survived).
   - **Gait OFF (0.0)**: policy free-style → smaller strides (mean 50 m), but stays upright more (36/50).
   - The forced cadence rewards aggressive committing-to-step behavior; remove it and the policy hedges.

5. **`velocity_bonus` is load-bearing despite being 0.97-correlated with step_reward.** v12 vs v16 (only difference is velocity_bonus 5 vs 0): 49 vs 32 m mean. PCA's "redundant by correlation" reading didn't capture that velocity's `trig` shape gives the gradient distinct curvature even when the values track step_reward.

**Two policies worth showcasing on slides / deployment:**

| Use case | Pick | Reasoning |
|---|---|---|
| **"Watch the robot walk far"** demo footage | **v20** | Distance champion: 64 m mean, 81 m peak, 1.0 m/s sustained. Aggressive committed-stride gait. |
| **Sim-to-real deployment candidate** | **v21** | Survival champion: 36/50 full-episode (72% survival). 50 m mean still strong. Less fall-prone makes it safer for hardware deployment. |

**Calibrated PCA on v12_3M_fixed (first trustworthy correlation analysis):**

Real principal components of the 10-dim reward, finally with working signals:

| PC | Variance | Dominant terms | Interpretation |
|---|---|---|---|
| PC1 | 25% | velocity, step_reward, extension, posture | "walking forward with form" |
| PC2 | 21% | fall, alive, posture | "stability axis" |
| PC3 | 16% | gait + weight_transfer (r=0.64) | "contact pattern coordination" |
| PC4 | 12% | stride + velocity + step_reward | "big strides axis" |
| PC5 | 10% | smoothness alone | independent regularizer |
| PC6 | 8% | extension + posture | "stance commitment" |
| PC7–10 | <8% each | mostly noise | redundant directions |

**Methodology lesson — for the slide deck:**
- **PCA on broken instrumentation gives misleading conclusions.** Our first PCA (§4.7) ran on a policy where 3 of 10 reward terms were silently disabled. It "correctly" flagged velocity as redundant (r=0.93 in that data) — but that conclusion was an artifact of the broken stride/WT signals leaving step_reward and velocity as the only forward-drive terms. With fixed signals, dropping velocity costs 17 m mean. **Always re-validate PCA findings with ablation runs.**
- **Correlated values can be non-redundant gradients.** Even after fixing signals, step_reward ↔ velocity remains 0.97 correlated, but velocity's trig shape provides distinct curvature in the gradient that step_reward's linear `dx` doesn't. PCA on values alone can't tell you this — only an ablation experiment can.

**v22 — confirms stride=5 is near-optimal.** Bumping `stride_bonus` from 5 → 20 (4×) regressed across every metric: mean 64 → 50 m, max 81 → 76 m, survived 33 → 27. Higher coefficient ≠ better. The reward-coefficient landscape is non-monotonic — tiny stride (1.5) is too weak, huge stride (20) destabilizes, sweet spot was 5. This kind of one-axis test is the cheap way to bound a coefficient before doing a full random search.

| stride_bonus | Run | Mean | Survived |
|---|---|---|---|
| 1.5 | v12_3M_fixed | 49.3 m | 24/50 |
| **5** | **v20** | **63.9 m** | **33/50** |
| 20 | v22 | 50.5 m | 27/50 |

### 4.10 Random coefficient sweep — when log-uniform sampling helps and when it doesn't

After v22 confirmed `stride=5` was near-optimal on a 1-D sweep, the next question was whether jointly tuning all 5 reward coefficients could find a better basin. Hand-iteration had bounded each coef in isolation; random search over the 5-D space could surface combinations we wouldn't think to try.

**Sweep design (`random_sweep.py`):** sample log-uniform from
- `velocity_bonus`, `extension_bonus`, `stride_bonus`, `weight_transfer_bonus` ∈ [0.1, 10]
- `gait_reward_scale` ∈ [0.01, 1] (smaller range — the underlying score is already ±4)

with `velocity_shape=trig` fixed (winner of v6 sweep). 52 random configurations total (12 + 40 across two seeds), each trained for 1 M timesteps, evaluated via `ppo_stats.py --episodes 30 --fall-tilt 30`. Final ranking by `combined = mean_dist × (survived/30)`.

**Outcome (across 28+ samples — full results pending):**
- **No random configuration beat v20** (3 M training, 64 m mean / 33-of-50 survived).
- Best random config (`rand_05`, sweep 1): vel=0.25, ext=0.86, stride=0.12, wt=0.20, gait=0.23 → 51 m / 22-of-30. Roughly v21-class performance with a "near-minimal" reward.
- Per-coefficient correlation with `combined` score:

| coef | r(mean) | r(survival) | r(combined) |
|---|---|---|---|
| **gait** | **+0.43** | **+0.55** | **+0.45** |
| stride | −0.36 | −0.24 | −0.30 |
| ext | +0.11 | +0.12 | +0.13 |
| wt | +0.00 | +0.31 | +0.15 |
| vel | −0.06 | +0.03 | ~0 |

**The two surprising findings:**

1. **`gait_reward_scale` is the most predictive single coefficient at 1 M.** Higher gait → better random config performance. This contradicts §4.9's v19/v21 result (where dropping gait helped) — but those were 3 M runs starting from v20-class configurations. At 1 M from a random start, gait acts as a useful inductive bias that helps PPO converge faster.

2. **`stride_bonus` correlates *negatively* with score at 1 M**, even though v20 (stride=5) is the 3 M champion. Reading: high stride coefficients require longer training to converge usefully. At 1 M, a low stride coef converges to a stable-but-modest gait; at 3 M, a high stride coef has time to develop into v20's longer-stride gait.

**The methodology lesson** — for the slide deck:
> **Optimal coefficients depend on training budget.** Random search at 1 M finds different "good configs" than careful iteration at 3 M, and both are finding *real* local optima within their respective budgets. A coefficient sweep that ignores training-time interactions can mislead. To find the best 3 M config, you'd need to do random search at 3 M (~3× the compute) or use a multi-fidelity method like Hyperband / SMAC. We did the cheap version — and learned that the cheap version answers a different question than we thought it was answering.

**What this rules in vs. out for v20+ deployment:**
- ✅ v20 (3 M, hand-iterated) remains the distance champion; v21 the survival champion.
- ❌ A 1 M random search isn't going to outperform either of them. Don't try to.
- 🧪 Re-running the *top 3 random configs* at 3 M would test whether their "lower-stride / higher-gait" recipes scale with training time. That's the next experiment if we want to push the frontier.

Implementation references:
- `random_sweep.py`: log-uniform sampler + per-config train+eval orchestrator. Saves results JSON incrementally so partial sweeps aren't lost.
- `analyze_sweep.py`: ingests one or more sweep JSON files, prints top-N table, per-coefficient correlations, and binned marginal trends. Useful for deciding which configs to promote to longer-budget runs.

### 4.11 CMA vs. RL head-to-head — and the hybrid path

The cluster sweep (`results/20260423_221949/`) trained **90 CMA / DE gaits** across `{cma, de}` × `{gait, stand, random}` × `{linear, cosine, smoothstep}` interpolation × randomized tilt/sigma. Selected the top-5 by walking distance via `rank_gaits.py --sort speed`.

**`cma_stats.py`** — apples-to-apples evaluator for CMA gaits matching the RL eval protocol: 50 stochastic episodes, ±2° initial joint noise, ±5 mm body-z noise, fall_tilt=30°. The init-pose noise is what gives CMA gaits a stochastic distribution to compare against (otherwise open-loop replay is exactly deterministic).

**Top 5 CMA gaits, 50-episode evaluation:**

| Rank | Mean dist | Speed | Survived | Combined |
|---|---|---|---|---|
| 1 | 8.78 m / 18.7 s | **0.479 m/s** | 45/50 (90%) | 7.90 |
| 2 | 7.29 m | 0.391 m/s | 50/50 (100%) | 7.29 |
| 3 | 7.28 m | 0.385 m/s | 50/50 (100%) | 7.28 |
| 4 | 7.68 m | 0.352 m/s | 50/50 (100%) | 7.68 |
| 5 | 8.07 m | 0.343 m/s | 50/50 (100%) | 8.07 |

**Direct head-to-head with our RL champions** (composite metric `speed × survival_rate` since rollout durations differ):

| Policy | Speed (m/s) | Survival | speed × surv |
|---|---|---|---|
| **v20 (RL)** | **0.80** | 66% | **0.528** |
| v21 (RL) | 0.63 | 72% | 0.453 |
| CMA #1 (best) | 0.48 | 90% | 0.431 |
| CMA #5 (most reliable) | 0.34 | 100% | 0.343 |

**RL wins by 22% on the composite.** v20 is 1.7× faster than the best CMA, and 2.3× faster than the most reliable CMA gait. RL's headroom comes entirely from speed; CMA's strength is reliability.

**The CMA frontier shows the same internal trade-off as RL:**
- CMA #1 trades survival (90%) for speed (0.48 m/s)
- CMA #5 trades speed (0.34 m/s) for full survival (100%)
- v20 trades survival (66%) for speed (0.80 m/s)
- v21 trades speed (0.63 m/s) for survival (72%)

So both methods sit on a Pareto frontier; the question is whose frontier is further out. **RL's frontier is uniformly faster.**

**Why CMA loses speed but wins reliability:**
- **CMA is open-loop deterministic** — same trajectory every time, no closed-loop adjustment to perturbation.
- **RL is closed-loop stochastic** — observation-conditioned policy that *can* react to disturbance, but introduces variance that occasionally pushes it past the stability margin.
- For nominal-condition demos in sim, CMA is a perfectly tuned trajectory that hits the floor every time. For real-world deployment with sensor noise + ground variations, this brittleness becomes a liability — the user's instinct.

**The hybrid path — BC + PPO fine-tune (`bc_pretrain.py`):**

Goal: get RL's adaptive closed-loop control with CMA's reliable trajectory as a baseline.

Pipeline:
1. **Behavioral cloning (`bc_pretrain.py`)** — roll out the best CMA gait in the RL env (with init-pose noise for diversity), record (obs, action) pairs at each control step. Train PPO's actor network via supervised MSE to predict CMA's action given the RL obs. This embeds CMA's stride pattern into the policy.
2. **PPO fine-tune (`train_ppo.py --init-from`)** — load the BC-pretrained policy and continue with normal PPO learning. The policy starts at "CMA's trajectory" and is free to adjust based on closed-loop feedback (roll, pitch, body velocity, pitch rate, body_fwd_z — all IMU-derivable signals already in the 24-dim obs).

What the hybrid is supposed to fix:
- **Brittleness**: when sim physics drift (sim-to-real, terrain change), the closed-loop part of the policy compensates. Open-loop CMA cannot.
- **Convergence speed**: PPO from random init takes 3 M timesteps to reach v20-level. PPO from CMA seed should converge much faster because it starts on the reward landscape's good basin instead of in random territory.
- **Worst-case failure mode**: if PPO's exploration finds a regression, the BC-cloned baseline still encodes "fall back to CMA-like behavior."

Results pending — `ppo_v24_bc.zip` is the BC checkpoint; `ppo_v24_bc_ppo.zip` is the fine-tuned policy (currently training).

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
| 2026-04-25 | **CMA vs RL + hybrid (§4.11)**: evaluated cluster's top-5 CMA gaits at 50 episodes via new `cma_stats.py` (with init-pose noise for stochasticity). Best CMA: 0.479 m/s / 90% survival. v20 (RL) beats best CMA by 1.7× on speed and 22% on the `speed × survival_rate` composite. Both methods sit on a Pareto frontier; RL's is uniformly faster. Started hybrid: BC pretrain (`bc_pretrain.py`) imitates CMA gait into PPO's actor, then `train_ppo.py --init-from` fine-tunes with closed-loop PPO. Pending v24_bc_ppo result. |
| 2026-04-25 | **Random coefficient sweep (§4.10)**: 52 log-uniform random configs over the 5 reward coefficients at 1 M timesteps each. No random config beat v20. Best random sample at ~51 m mean (v21-class). Strong finding: at 1 M, `gait_reward_scale` correlates +0.45 with combined score (highest predictor) and `stride_bonus` correlates −0.30 (opposite sign from 3 M optimum). Conclusion: optimal coefficients depend on training budget, and random search at 1 M answers a different question than 3 M iteration. `random_sweep.py` + `analyze_sweep.py` added for repro. |
| 2026-04-24 | **Post-fix re-sweep (§4.9)**: trained v12/v16/v6_trig at 3M timesteps + v19 (drop gait) + v20 (stride=5) + v21 (drop gait & stride=5). New distance champion **v20** at 63.92 m mean / 33-of-50 survived. New survival champion **v21** at 50.39 m / 36-of-50 (72% full-episode). With working signals, `stride_bonus=5` ≫ `1.5`, and `gait_reward` acts as an aggression switch (on = bigger strides + more falls, off = conservative gait + better survival). PCA on fixed v12 confirms 6 meaningful base axes including a coupled gait+WT contact-pattern axis (PC3, r=0.64). |
| 2026-04-24 | **Silent-signal bug discovered and fixed (§4.8)**: `FOOT_BODY_NAMES` referenced URDF link names that MuJoCo had merged away (mj_name2id → −1), and `cfrc_ext` is for user-applied forces, not ground contacts. Result: gait_reward was constant −0.5, stride_bonus and weight_transfer_bonus were effectively 0 for every v3–v18 training run. Fix: look up `lower_link_*` bodies (the merged parents), iterate MuJoCo contacts via `mj_contactForce`, use `FOOT_CONTACT_THRESHOLD=0.5 N` for lifted/planted distinction. All v8–v18 experiments that nominally tested stride/WT are now re-attributed to random-seed variance. v3–v7 results stand (didn't use the broken signals). |
| 2026-04-24 | `reward_pca.py` + `info["reward_components"]`: correlation + eigendecomposition diagnostic for the per-step reward components. On v12, finds that `step_reward` ↔ `velocity_bonus` are 0.93 correlated (redundant) and `gait_reward` has near-zero variance (dead weight). The near-zero variance was what uncovered the silent-signal bug (§4.8). |
| 2026-04-24 | **v12 becomes new champion** via synergy: weight_transfer=2.0 + stride_bonus=1.5 COMBINED beats v6_trig (45.48 m mean / 39/50 survived vs. 43.4 m / 32/50). Both terms regress individually (v10, v11) but together they interact positively. Key learning for ablation design: terms can be drowned in isolation but become load-bearing when paired with a complementary signal. |
| 2026-04-24 | RL v7–v14 reward-engineering plateau: attempted 7 single-variable changes on top of v6_trig (IMU obs, stride bonus, weight-transfer bonus, gait-reward removal, DR); all regressed or broke even. Documented findings in §4.6. Conclusion: more shaping unlikely to help; longer training is the next lever. Also fixed false-FELL bug via `--fall-tilt` flag in `ppo_stats.py` (default was 20°, over-triggering on aggressive gaits; eval runs now use 30°) |
| 2026-04-24 | RL v6 coefficient sweep (linear shape, `velocity_bonus` ∈ {0.1, 1.0, 5.0, 10.0}): peaked curve around coef=5, with coef=10 over-committing and dropping survival to 11/50. Below 5 still walks — most forward-drive comes from `step_reward(dx)`, not the velocity bonus (§4.5) |
| 2026-04-24 | RL v6: `ctrl_repeat` 4→8 (50 Hz → 25 Hz, matching LX-16A command rate), parameterized velocity-reward shape (`linear` / `quadratic` / `cubic` / `trig`) with `--velocity-shape` and scalar `--velocity-bonus` CLI flags. Shape sweep at coef=5 showed trig (29/50) and linear (27/50) beat quadratic (12/50) and cubic (22/50) on consistency (§4.4, §4.5) |
| 2026-04-24 | RL reward v5: added action-extension bonus `3.0 · mean(|a−mid|/half_range)` to escape v4's "jitter-at-stance" mode; 50-ep mean walking distance 4.28 m → 27.87 m, full-episode survival 0/50 → 44/50 (§4.4, §4.5) |
| 2026-04-24 | RL reward v4: lengthened `phase_period` 2→4 s, dropped `gait_reward_scale` 1.0→0.25, added `FOOT_PLANT_TOLERANCE=0.03` free-zone for non-target feet, added `+5·max(0, qvel[0])` speed bonus; 50-ep mean walking distance +46% over v3, max 14.55 m (§4.4, §4.5) |
| 2026-04-23 | RL reward: added forced gait-order (`FR → RL → FL → RR`) contact-pattern reward with `phase_period`/`gait_reward_scale` knobs and phase clock in observation (§4.4) |
| 2026-04-23 | RL reward: softened posture/smoothness penalties (band widened to ±0.03 m, smoothness coeff 1.0 → 0.1) so early PPO exploration isn't punished into standing still |
| 2026-04-23 | RL reward: added posture (z-band) + action-smoothness penalties + alive bonus + one-shot fall penalty on top of `step_reward` |
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
