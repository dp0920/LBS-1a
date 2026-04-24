#!/usr/bin/env python3
"""
Gait learning via CMA-ES — seeded from stance.py real-robot crawl gait.

The gait is a sequence of 13 phases (shift/swing/plant for FR→RL→FL→RR)
with all angles taken directly from the working real-robot gait in stance.py.
CMA-ES optimizes those angles + phase timing to maximize forward distance.

Usage:
  mjpython mujoco_gait.py --demo                     # run stance.py gait as-is
  python   mujoco_gait.py --tune                     # CMA-ES optimize
  mjpython mujoco_gait.py --replay best_gait.json    # visualize result



  ADD a discount factor
"""
import argparse
import json
import sys
import os
import multiprocessing as mp
import time
import numpy as np
import mujoco

from sim_core import (JOINTS, build_model, get_qadr, get_ctrl_idx,
                      get_base_body_id)
from reward import RewardAccumulator


def _ease(t, mode):
    """Map phase progress t ∈ [0,1] through an easing curve.
    linear:      t                              (constant velocity)
    cosine:      0.5 * (1 - cos(pi*t))          (C¹: zero vel at endpoints)
    smoothstep:  t**2 * (3 - 2*t)               (C¹, cubic polynomial)
    smootherstep: t**3 * (t*(6*t-15)+10)        (C², zero accel at endpoints)
    """
    if mode == "linear":
        return t
    if mode == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * t))
    if mode == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if mode == "smootherstep":
        return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    raise ValueError(f"unknown interp mode: {mode!r}")


def lerp_pose(a, b, t, interp="linear"):
    u = _ease(t, interp)
    return {j: a[j] * (1 - u) + b[j] * u for j in a}


def deg_pose(FL, FR, RL, RR):
    """Build a full pose dict from (hip_deg, knee_deg) per leg."""
    return {
        "hip_fl": np.radians(FL[0]), "knee_fl": np.radians(FL[1]),
        "hip_fr": np.radians(FR[0]), "knee_fr": np.radians(FR[1]),
        "hip_rl": np.radians(RL[0]), "knee_rl": np.radians(RL[1]),
        "hip_rr": np.radians(RR[0]), "knee_rr": np.radians(RR[1]),
    }


# ---------------------------------------------------------------------------
# Gait phases — traced from stance.py crawl gait (cumulative state)
# ---------------------------------------------------------------------------
# Each phase is a full 8-joint pose in degrees: (FL, FR, RL, RR) × (hip, knee)
# Sequence: FR → RL → FL → RR (diagonal crawl)
#
# Phase layout per leg:
#   shift  — redistribute weight so the target leg is unloaded
#   swing  — lift + swing the target leg forward
#   plant  — lower foot to ground at new position

# These are the default angles from the real robot. CMA-ES optimizes them.
# fmt: off
PHASE_DEFAULTS = {
    #                    FL          FR          RL          RR
    "start":      ((35, -80), (35, -80), (35, -50), (35, -50)),
    "shift_FR":   ((35, -95), (35, -80), (35, -65), (35, -40)),
    "swing_FR":   ((35, -95), (10,-100), (35, -65), (35, -40)),
    "plant_FR":   ((35, -95), (10, -65), (35, -65), (35, -40)),
    "shift_RL":   ((35, -65), (10, -80), (35, -65), (35, -65)),
    "swing_RL":   ((35, -65), (10, -80), (10, -75), (35, -65)),
    "plant_RL":   ((35, -65), (10, -80), (10, -35), (35, -65)),
    "shift_FL":   ((35, -65), (10, -95), (10, -35), (35, -65)),
    "swing_FL":   ((10,-100), (10, -95), (10, -35), (35, -65)),
    "plant_FL":   ((10, -65), (10, -95), (10, -35), (35, -65)),
    "shift_RR":   ((10, -95), (10, -65), (10, -50), (35, -65)),
    "swing_RR":   ((10, -95), (10, -65), (10, -50), (10, -75)),
    "plant_RR":   ((10, -95), (10, -65), (10, -50), (10, -35)),
}
# fmt: on

PHASE_ORDER = [
    "start",
    "shift_FR", "swing_FR", "plant_FR",
    "shift_RL", "swing_RL", "plant_RL",
    "shift_FL", "swing_FL", "plant_FL",
    "shift_RR", "swing_RR", "plant_RR",
]


def phases_to_vector(phases):
    """Flatten phase defs to a parameter vector (angle values only)."""
    vec = []
    for name in PHASE_ORDER:
        FL, FR, RL, RR = phases[name]
        vec.extend([FL[0], FL[1], FR[0], FR[1],
                    RL[0], RL[1], RR[0], RR[1]])
    return np.array(vec, dtype=float)


def vector_to_phases(vec):
    """Rebuild phase defs from a parameter vector."""
    phases = {}
    idx = 0
    for name in PHASE_ORDER:
        FL = (vec[idx], vec[idx+1])
        FR = (vec[idx+2], vec[idx+3])
        RL = (vec[idx+4], vec[idx+5])
        RR = (vec[idx+6], vec[idx+7])
        phases[name] = (FL, FR, RL, RR)
        idx += 8
    return phases


def phases_to_poses(phases):
    """Convert phase defs (degrees) to full pose dicts (radians)."""
    poses = {}
    for name in PHASE_ORDER:
        FL, FR, RL, RR = phases[name]
        poses[name] = deg_pose(FL, FR, RL, RR)
    return poses


# Build defaults
X0_ANGLES = phases_to_vector(PHASE_DEFAULTS)
# Append phase_time param (seconds per phase)
X0 = np.append(X0_ANGLES, [0.6])  # 0.6s per phase = stance.py PHASE_PAUSE

N_ANGLE_PARAMS = len(X0_ANGLES)  # 13 phases × 8 joints = 104
PARAM_LABELS = []
for name in PHASE_ORDER:
    for leg in ["FL", "FR", "RL", "RR"]:
        PARAM_LABELS += [f"{name}_{leg}_hip", f"{name}_{leg}_knee"]
PARAM_LABELS.append("phase_time")

# Bounds
BOUNDS_LO_ANGLES = np.tile([5, -100, 5, -100, 5, -100, 5, -100],
                           len(PHASE_ORDER))
BOUNDS_HI_ANGLES = np.tile([55, -25, 55, -25, 55, -25, 55, -25],
                           len(PHASE_ORDER))
BOUNDS_LO = np.append(BOUNDS_LO_ANGLES, [0.15])
BOUNDS_HI = np.append(BOUNDS_HI_ANGLES, [1.2])


def decode_params(x):
    """Decode flat param vector into poses + timing."""
    phases = vector_to_phases(x[:N_ANGLE_PARAMS])
    poses = phases_to_poses(phases)
    phase_time = float(x[N_ANGLE_PARAMS])
    return poses, phase_time


def make_x0(init, rng=None):
    """Build the initial CMA-ES parameter vector.

    init:
      "gait"   — seed from the hand-coded PHASE_DEFAULTS crawl
      "stand"  — every phase pinned to the standing pose (no motion)
      "random" — uniform draw within bounds
    """
    if init == "gait":
        return X0.copy()
    if init == "stand":
        stand = PHASE_DEFAULTS["start"]
        phases = {name: stand for name in PHASE_ORDER}
        return np.append(phases_to_vector(phases), [0.6])
    if init == "random":
        rng = rng if rng is not None else np.random.default_rng()
        return rng.uniform(BOUNDS_LO, BOUNDS_HI)
    raise ValueError(f"unknown init mode: {init!r}")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(model, poses, phase_time, n_cycles=5, viewer=None, data=None,
            fall_tilt_deg=20.0, tilt_scale=1.0, interp="linear"):
    """Run gait for n_cycles. Returns (reward, info).

    fall_tilt_deg: kill threshold (pitch/roll); also scales the nose-down kill
                   threshold proportionally.
    tilt_scale:    multiplier on soft tilt/flop/pitch-rate penalties.
    interp:        'linear' | 'cosine' | 'smoothstep' | 'smootherstep' —
                   controls how joint angles are interpolated between phases.
    """
    if data is None:
        data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qadr = get_qadr(model)
    ctrl_idx = get_ctrl_idx(model)
    dt = model.opt.timestep
    base_body_id = get_base_body_id(model)

    acc = RewardAccumulator(dt, fall_tilt_deg=fall_tilt_deg,
                            tilt_scale=tilt_scale)

    def set_ctrl(pose):
        for name, angle in pose.items():
            data.ctrl[ctrl_idx[name]] = angle

    def set_qpos(pose):
        for name, angle in pose.items():
            data.qpos[qadr[name]] = angle

    def run_phase(from_pose, to_pose, duration):
        steps = max(1, int(duration / dt))
        phase_wall = time.time() if viewer else 0
        for i in range(steps):
            t = i / steps
            set_ctrl(lerp_pose(from_pose, to_pose, t, interp))
            mujoco.mj_step(model, data)
            acc.sample_step(data, base_body_id)
            if acc.is_fallen(data, base_body_id):
                return False
            if viewer:
                viewer.sync()
                elapsed = (i + 1) * dt
                wall = time.time() - phase_wall
                if elapsed > wall:
                    time.sleep(elapsed - wall)
                if not viewer.is_running():
                    return False
        set_ctrl(to_pose)
        mujoco.mj_step(model, data)
        acc.sample_step(data, base_body_id)
        return not acc.is_fallen(data, base_body_id)

    # Init at start pose
    start_pose = poses["start"]
    set_qpos(start_pose)
    data.qpos[2] = 0.16
    data.qpos[3] = 1.0
    set_ctrl(start_pose)
    mujoco.mj_forward(model, data)

    # Settle (2s)
    if viewer:
        print(">>> Settling (2s)...")
    settle_wall = time.time() if viewer else 0
    for i in range(int(2.0 / dt)):
        set_ctrl(start_pose)
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            elapsed = (i + 1) * dt
            wall = time.time() - settle_wall
            if elapsed > wall:
                time.sleep(elapsed - wall)

    x_start = float(data.qpos[0])
    x_at_last_phase = x_start
    earned_distance = 0.0
    z_sum = 0.0
    z_n = 0
    min_z = float(data.qpos[2])
    phases_completed = 0
    total_phases = n_cycles * (len(PHASE_ORDER) - 1)  # exclude "start"
    alive = True

    # The gait phases (skip "start" — that's where we already are)
    stride_phases = PHASE_ORDER[1:]  # shift_FR through plant_RR

    for cycle in range(n_cycles):
        if viewer:
            print(f">>> Cycle {cycle + 1}/{n_cycles}  "
                  f"x={data.qpos[0]:+.4f}  z={data.qpos[2]:.3f}")

        prev_name = "start" if cycle == 0 else "start"

        for phase_name in stride_phases:
            if not run_phase(poses[prev_name], poses[phase_name], phase_time):
                alive = False
                break

            prev_name = phase_name
            phases_completed += 1

            # Track distance at each plant phase (completed step)
            if phase_name.startswith("plant_"):
                x_now = float(data.qpos[0])
                earned_distance += x_now - x_at_last_phase
                x_at_last_phase = x_now

            # Height tracking (attitude is sampled per physics step inside run_phase)
            cur_z = float(data.qpos[2])
            z_sum += cur_z
            z_n += 1
            min_z = min(min_z, cur_z)

        if not alive:
            break

        # Recenter: return to start pose for next cycle
        if alive and cycle < n_cycles - 1:
            if not run_phase(poses[stride_phases[-1]], poses["start"],
                             phase_time):
                alive = False
                break

    mean_z = z_sum / max(1, z_n)
    return acc.finalize(earned_distance, phases_completed, total_phases,
                        mean_z=mean_z, final_z=float(data.qpos[2]),
                        min_z=min_z)


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

_worker_model = None
_worker_data = None

_worker_fall_tilt_deg = 20.0
_worker_tilt_scale = 1.0
_worker_interp = "linear"


def _worker_init(fall_tilt_deg=20.0, tilt_scale=1.0, interp="linear"):
    """Each worker builds its own MuJoCo model (can't pickle them).
    Also ignore SIGINT so ctrl-C goes only to the parent — otherwise every
    worker dies at once and pool.map hangs inside C code."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _worker_model, _worker_data
    global _worker_fall_tilt_deg, _worker_tilt_scale, _worker_interp
    _worker_model = build_model()
    _worker_data = mujoco.MjData(_worker_model)
    _worker_fall_tilt_deg = fall_tilt_deg
    _worker_tilt_scale = tilt_scale
    _worker_interp = interp


def _worker_eval(args):
    """Evaluate one candidate. Returns (reward, info)."""
    x, n_cycles = args
    poses, phase_time = decode_params(x)
    return rollout(_worker_model, poses, phase_time,
                   n_cycles=n_cycles, data=_worker_data,
                   fall_tilt_deg=_worker_fall_tilt_deg,
                   tilt_scale=_worker_tilt_scale,
                   interp=_worker_interp)


# ---------------------------------------------------------------------------
# Viewer / demo
# ---------------------------------------------------------------------------

def run_viewer(model, poses, phase_time, n_cycles=5, interp="linear"):
    import mujoco.viewer
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        r, info = rollout(model, poses, phase_time,
                          n_cycles=n_cycles, viewer=viewer, data=data,
                          interp=interp)
        print(f"reward={r:+.4f}  {info}")
        print("Done. Close viewer to exit.")
        while viewer.is_running():
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# CMA-ES tuning
# ---------------------------------------------------------------------------

def tune(model, out=None, generations=150, popsize=48,
         n_cycles=5, resume=False, workers=None, init="gait",
         fall_tilt_deg=20.0, tilt_scale=1.0, interp="linear",
         sigma_init=5.0):
    try:
        import cma
    except ImportError:
        print("pip install cma", file=sys.stderr)
        sys.exit(1)

    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    # Auto-name outputs per init mode so runs don't clobber each other.
    # "gait" keeps the historical filenames for back-compat.
    suffix = "" if init == "gait" else f"_{init}"
    if out is None:
        out = f"best_gait{suffix}.json"
    log_path = f"tune_gait{suffix}.jsonl"

    # Resume from saved best if available; otherwise seed from --init mode.
    x0 = make_x0(init)
    best_r = -1e9
    if resume and os.path.exists(out):
        with open(out) as f:
            d = json.load(f)
        x0 = np.clip(np.array(d["params"]), BOUNDS_LO, BOUNDS_HI)
        best_r = d.get("reward", -1e9)
        print(f"Resuming from {out} (reward={best_r:+.4f}, clamped to bounds)")
    else:
        print(f"Initial seed: {init}")

    best_x = x0.copy()
    gen = 0
    stall_count = 0
    prev_best = best_r
    restart = 0

    n_params = len(x0)
    print(f"Tuning gait ({n_params} params: {N_ANGLE_PARAMS} angles + 1 timing, "
          f"{n_cycles} cycles/eval)")

    sigma = float(sigma_init)
    es = cma.CMAEvolutionStrategy(x0, sigma, {
        "popsize": popsize,
        "bounds": [list(BOUNDS_LO), list(BOUNDS_HI)],
        "maxiter": generations,
        "verbose": -9,
        "tolstagnation": 0,
    })

    pool = mp.Pool(workers, initializer=_worker_init,
                   initargs=(fall_tilt_deg, tilt_scale, interp))
    print(f"  workers: {workers}  fall_tilt={fall_tilt_deg}°  "
          f"tilt_scale={tilt_scale}  interp={interp}")

    cfg = {"algo": "cma", "init": init, "fall_tilt_deg": fall_tilt_deg,
           "tilt_scale": tilt_scale, "interp": interp,
           "sigma_init": sigma,
           "generations": generations, "popsize": popsize,
           "n_cycles": n_cycles}

    try:
      while not es.stop():
        xs = es.ask()
        # map_async + get(timeout) so ctrl-C lands in Python, not C-level map.
        async_result = pool.map_async(_worker_eval,
                                      [(x, n_cycles) for x in xs])
        results = async_result.get(timeout=1e9)
        fs = []
        best_info_this_gen = None
        for x, (r, info) in zip(xs, results):
            fs.append(-r)
            if r > best_r:
                best_r = r
                best_x = x.copy()
                best_info_this_gen = info
        es.tell(xs, fs)
        gen += 1
        pop_mean = float(-np.mean(fs))
        pop_best = float(-min(fs))

        status = ""
        if best_info_this_gen:
            status = (f"  dist={best_info_this_gen['earned_dist_m']:+.4f}m "
                      f"speed={best_info_this_gen['speed_mps']:.3f}m/s "
                      f"phases={best_info_this_gen['phases']} "
                      f"z={best_info_this_gen['mean_z']:.3f} "
                      f"nose_down={best_info_this_gen['mean_nose_down_deg']:.1f}°/"
                      f"{best_info_this_gen['max_nose_down_deg']:.1f}° "
                      f"roll={best_info_this_gen['mean_roll_deg']:.1f}°/"
                      f"{best_info_this_gen['max_roll_deg']:.1f}° "
                      f"d_pitch_max={best_info_this_gen['max_pitch_rate']:.1f}")
        print(f"gen {gen:3d}  best={best_r:+.4f}  mean={pop_mean:+.4f}  "
              f"pop_best={pop_best:+.4f}{status}", flush=True)

        if abs(best_r - prev_best) < 0.01:
            stall_count += 1
        else:
            stall_count = 0
            prev_best = best_r

        if stall_count >= 15:
            restart += 1
            stall_count = 0
            new_sigma = sigma * (1 + 0.3 * restart)
            print(f">>> RESTART {restart} — stalled for 15 gens, "
                  f"sigma={new_sigma:.1f}", flush=True)
            es = cma.CMAEvolutionStrategy(best_x, new_sigma, {
                "popsize": popsize + 8 * restart,
                "bounds": [list(BOUNDS_LO), list(BOUNDS_HI)],
                "maxiter": generations - gen,
                "verbose": -9,
                "tolstagnation": 0,
            })

        _save_and_log(best_x, best_r, gen, restart, out, log_path,
                      pop_mean, pop_best, config=cfg)
    except KeyboardInterrupt:
        print("\n>>> Interrupted — saving best and exiting.", flush=True)
    finally:
        pool.terminate()
        pool.join()

    _print_best(best_x, best_r, restart, out)


def _print_best(best_x, best_r, restart, out):
    """Print final phase angles — shared by all tuning algorithms."""
    phases = vector_to_phases(best_x[:N_ANGLE_PARAMS])
    pt = best_x[N_ANGLE_PARAMS]
    print(f"\nDone. best reward {best_r:.4f}  ({restart} restarts)")
    print(f"  phase_time={pt:.2f}s")
    for name in PHASE_ORDER:
        FL, FR, RL, RR = phases[name]
        print(f"  {name:12s}  FL({FL[0]:+5.1f},{FL[1]:+6.1f})  "
              f"FR({FR[0]:+5.1f},{FR[1]:+6.1f})  "
              f"RL({RL[0]:+5.1f},{RL[1]:+6.1f})  "
              f"RR({RR[0]:+5.1f},{RR[1]:+6.1f})")
    print(f"Saved to {out}")


def _save_and_log(best_x, best_r, gen, restart, out, log_path,
                  pop_mean=None, pop_best=None, config=None):
    """Write best JSON + append to JSONL log — shared by all algorithms.
    `config` is an optional dict of training settings (algo, init, fall_tilt,
    tilt_scale, interp, ...) persisted inside the saved JSON."""
    result = {
        "params": list(best_x),
        "reward": best_r,
        "gen": gen,
        "restart": restart,
    }
    if config:
        result["config"] = dict(config)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    entry = {"gen": gen, "best": best_r}
    if pop_mean is not None:
        entry["mean"] = pop_mean
    if pop_best is not None:
        entry["pop_best"] = pop_best
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def tune_random(model, out=None, generations=150, popsize=48,
                n_cycles=5, workers=None, init="gait",
                fall_tilt_deg=20.0, tilt_scale=1.0, interp="linear"):
    """Baseline: sample popsize candidates per generation, keep the best.
    No learning, no covariance adaptation — just random sampling + elitism."""
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    suffix = f"_rand_{init}" if init != "gait" else "_rand"
    if out is None:
        out = f"best_gait{suffix}.json"
    log_path = f"tune_gait{suffix}.jsonl"

    x0 = make_x0(init)
    best_r = -1e9
    best_x = x0.copy()
    rng = np.random.default_rng()

    print(f"Random search ({len(x0)} params, popsize={popsize}, "
          f"gens={generations}, {n_cycles} cycles/eval)")

    pool = mp.Pool(workers, initializer=_worker_init,
                   initargs=(fall_tilt_deg, tilt_scale, interp))
    print(f"  workers: {workers}  fall_tilt={fall_tilt_deg}°  "
          f"tilt_scale={tilt_scale}  interp={interp}")

    cfg = {"algo": "random", "init": init, "fall_tilt_deg": fall_tilt_deg,
           "tilt_scale": tilt_scale, "interp": interp,
           "generations": generations, "popsize": popsize,
           "n_cycles": n_cycles}

    try:
        for gen in range(1, generations + 1):
            # Generate population: first candidate is always the current best
            # (elitism), rest are uniform random within bounds.
            xs = [best_x.copy()]
            for _ in range(popsize - 1):
                xs.append(rng.uniform(BOUNDS_LO, BOUNDS_HI))

            async_result = pool.map_async(_worker_eval,
                                          [(x, n_cycles) for x in xs])
            results = async_result.get(timeout=1e9)

            rewards = []
            best_info_this_gen = None
            for x, (r, info) in zip(xs, results):
                rewards.append(r)
                if r > best_r:
                    best_r = r
                    best_x = x.copy()
                    best_info_this_gen = info

            pop_mean = float(np.mean(rewards))
            pop_best = float(max(rewards))

            status = ""
            if best_info_this_gen:
                status = (f"  dist={best_info_this_gen['earned_dist_m']:+.4f}m "
                          f"phases={best_info_this_gen['phases']} "
                          f"z={best_info_this_gen['mean_z']:.3f}")
            print(f"gen {gen:3d}  best={best_r:+.4f}  mean={pop_mean:+.4f}  "
                  f"pop_best={pop_best:+.4f}{status}", flush=True)

            _save_and_log(best_x, best_r, gen, 0, out, log_path,
                          pop_mean, pop_best, config=cfg)
    except KeyboardInterrupt:
        print("\n>>> Interrupted — saving best and exiting.", flush=True)
    finally:
        pool.terminate()
        pool.join()

    _print_best(best_x, best_r, 0, out)


# ---------------------------------------------------------------------------
# Differential Evolution (scipy)
# ---------------------------------------------------------------------------

def tune_de(model, out=None, generations=150, popsize=48,
            n_cycles=5, workers=None, init="gait",
            fall_tilt_deg=20.0, tilt_scale=1.0, interp="linear"):
    """Differential Evolution via scipy — population-based, mutation by
    combining existing solutions. No covariance learning."""
    from scipy.optimize import differential_evolution

    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    suffix = f"_de_{init}" if init != "gait" else "_de"
    if out is None:
        out = f"best_gait{suffix}.json"
    log_path = f"tune_gait{suffix}.jsonl"

    x0 = make_x0(init)
    bounds = list(zip(BOUNDS_LO, BOUNDS_HI))

    print(f"Differential Evolution ({len(x0)} params, popsize={popsize}, "
          f"maxiter={generations}, {n_cycles} cycles/eval)")

    pool = mp.Pool(workers, initializer=_worker_init,
                   initargs=(fall_tilt_deg, tilt_scale, interp))
    print(f"  workers: {workers}  fall_tilt={fall_tilt_deg}°  "
          f"tilt_scale={tilt_scale}  interp={interp}")

    cfg = {"algo": "de", "init": init, "fall_tilt_deg": fall_tilt_deg,
           "tilt_scale": tilt_scale, "interp": interp,
           "generations": generations, "popsize": popsize,
           "n_cycles": n_cycles}

    gen_counter = [0]
    best_r = [-1e9]
    best_x = [x0.copy()]

    def callback(xk, convergence=0):
        """Called after each DE generation."""
        gen_counter[0] += 1
        r, info = rollout(model, *decode_params(xk), n_cycles=n_cycles,
                          fall_tilt_deg=fall_tilt_deg,
                          tilt_scale=tilt_scale,
                          interp=interp)
        if r > best_r[0]:
            best_r[0] = r
            best_x[0] = xk.copy()
        status = (f"  dist={info['earned_dist_m']:+.4f}m "
                  f"phases={info['phases']} "
                  f"z={info['mean_z']:.3f}")
        print(f"gen {gen_counter[0]:3d}  best={best_r[0]:+.4f}  "
              f"conv={convergence:.4f}{status}", flush=True)
        _save_and_log(best_x[0], best_r[0], gen_counter[0], 0, out,
                      log_path, config=cfg)

    def objective(x):
        """Evaluate a single candidate (called by DE vectorized=False)."""
        poses, phase_time = decode_params(x)
        r, _ = rollout(_worker_model, poses, phase_time,
                       n_cycles=n_cycles, data=_worker_data,
                       fall_tilt_deg=_worker_fall_tilt_deg,
                       tilt_scale=_worker_tilt_scale,
                       interp=_worker_interp)
        if r > best_r[0]:
            best_r[0] = r
            best_x[0] = x.copy()
        return -r

    try:
        # Seed the initial population: put x0 in and let DE fill the rest.
        # DE's init can take an array of shape (popsize*len(x), ndim).
        n_pop = popsize * len(x0)  # scipy's popsize is a multiplier
        init_pop = np.array([np.clip(
            x0 + np.random.randn(len(x0)) * 5.0,
            BOUNDS_LO, BOUNDS_HI
        ) for _ in range(max(n_pop, popsize))])
        init_pop[0] = x0  # ensure seed is in the population

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=generations,
            popsize=popsize,  # scipy multiplies this by ndim for pop count
            init=init_pop[:popsize * len(x0)],
            callback=callback,
            tol=0,
            atol=0,
            seed=42,
            workers=-1,  # use scipy's own parallelism
            updating="deferred",  # required for workers=-1
        )
        # Final best from scipy
        r, info = rollout(model, *decode_params(result.x), n_cycles=n_cycles,
                          fall_tilt_deg=fall_tilt_deg,
                          tilt_scale=tilt_scale,
                          interp=interp)
        if r > best_r[0]:
            best_r[0] = r
            best_x[0] = result.x.copy()
        _save_and_log(best_x[0], best_r[0], gen_counter[0], 0, out, log_path,
                      config=cfg)
    except KeyboardInterrupt:
        print("\n>>> Interrupted — saving best and exiting.", flush=True)
    finally:
        pool.terminate()
        pool.join()

    _print_best(best_x[0], best_r[0], 0, out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Resume tuning from best_gait.json")
    ap.add_argument("--replay", type=str, default=None)
    ap.add_argument("--demo", action="store_true",
                    help="Run the stance.py gait unmodified")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--popsize", type=int, default=48)
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallel workers (default: cpu_count - 1)")
    ap.add_argument("--init", choices=["gait", "stand", "random"],
                    default="gait",
                    help="Seed: 'gait' (hand-coded crawl, default), "
                         "'stand' (every phase at standing pose), "
                         "'random' (uniform within bounds)")
    ap.add_argument("--algo", choices=["cma", "random", "de"],
                    default="cma",
                    help="Optimization algorithm: 'cma' (CMA-ES, default), "
                         "'random' (random search baseline), "
                         "'de' (Differential Evolution)")
    ap.add_argument("--fall-tilt", type=float, default=20.0,
                    help="Tilt kill threshold in degrees (pitch/roll/nose-down "
                         "scaled from this). Lower = stricter. Default 20.")
    ap.add_argument("--tilt-scale", type=float, default=1.0,
                    help="Multiplier on soft tilt/flop/pitch-rate penalties. "
                         "Higher = stricter stability. Default 1.0.")
    ap.add_argument("--interp",
                    choices=["linear", "cosine", "smoothstep", "smootherstep"],
                    default="linear",
                    help="Joint-angle interpolation between phases. "
                         "'cosine'/'smoothstep' have zero velocity at phase "
                         "boundaries (smoother transitions). Default linear.")
    ap.add_argument("--sigma-init", type=float, default=5.0,
                    help="CMA-ES initial step size. Small (~2) = fine-tune "
                         "near seed, large (~15) = broad exploration. "
                         "Default 5.0. Ignored by random/de algorithms.")
    args = ap.parse_args()

    model = build_model()

    if args.replay:
        with open(args.replay) as f:
            d = json.load(f)
        poses, phase_time = decode_params(np.array(d["params"]))
        # Use the interp from the saved config if it was recorded; else CLI.
        replay_interp = d.get("config", {}).get("interp", args.interp)
        run_viewer(model, poses, phase_time, n_cycles=args.cycles,
                   interp=replay_interp)
    elif args.tune or args.resume:
        common = dict(generations=args.generations, popsize=args.popsize,
                      n_cycles=args.cycles, workers=args.workers,
                      init=args.init,
                      fall_tilt_deg=args.fall_tilt,
                      tilt_scale=args.tilt_scale,
                      interp=args.interp)
        if args.algo == "cma":
            tune(model, resume=args.resume, sigma_init=args.sigma_init,
                 **common)
        elif args.algo == "random":
            tune_random(model, **common)
        elif args.algo == "de":
            tune_de(model, **common)
    elif args.demo:
        poses, phase_time = decode_params(X0)
        run_viewer(model, poses, phase_time, n_cycles=args.cycles,
                   interp=args.interp)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
