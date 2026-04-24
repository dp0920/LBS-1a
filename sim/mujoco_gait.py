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

JOINTS = ["hip_fl", "knee_fl", "hip_fr", "knee_fr",
          "hip_rl", "knee_rl", "hip_rr", "knee_rr"]

LEGS = {
    "FL": ("hip_fl", "knee_fl"),
    "FR": ("hip_fr", "knee_fr"),
    "RL": ("hip_rl", "knee_rl"),
    "RR": ("hip_rr", "knee_rr"),
}


def build_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "optimus_primal.urdf")
    spec = mujoco.MjSpec.from_file(urdf_path)
    spec.meshdir = os.path.join(script_dir, "meshes")
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.4, 0.6, 0.9], rgb2=[0.1, 0.15, 0.25],
                     width=512, height=512)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.25, 0.26, 0.27], rgb2=[0.32, 0.33, 0.34],
                     width=512, height=512)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[10, 10],
                      reflectance=0.1)
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1],
                             diffuse=[0.9, 0.9, 0.9])
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5, 5, 0.1], material="grid",
                            friction=[1.0, 0.05, 0.001])
    base = spec.body("base_link")
    base.add_freejoint()
    for jname in JOINTS:
        act = spec.add_actuator(name=f"act_{jname}")
        act.target = jname
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=2.5, kv=0.05)
    return spec.compile()


def get_qadr(model):
    qadr = {}
    for j in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qadr[j] = model.jnt_qposadr[jid]
    return qadr


def get_ctrl_idx(model):
    ctrl = {}
    for j in JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"act_{j}")
        ctrl[j] = aid
    return ctrl


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

    def set_ctrl(pose):
        for name, angle in pose.items():
            data.ctrl[ctrl_idx[name]] = angle

    def set_qpos(pose):
        for name, angle in pose.items():
            data.qpos[qadr[name]] = angle

    FALL_Z = 0.08
    FALL_TILT = np.radians(max(0.1, fall_tilt_deg))
    # nose-down kill scales with tilt budget: sin(tilt_deg) ≈ forward-z cutoff.
    FALL_NOSE_DOWN = float(np.sin(FALL_TILT))

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     "base_link")

    # Per-step body-attitude trackers (filled inside run_phase).
    pitch_sq_sum = [0.0]
    roll_sq_sum = [0.0]
    pitch_abs_sum = [0.0]
    roll_abs_sum = [0.0]
    tilt_n_steps = [0]
    max_pitch = [0.0]
    max_roll = [0.0]
    # "Nose-down" (forward flop) — specific direction, not abs.
    # fwd_z = body-x axis's world-z component:
    #   +1 = nose pointing straight up
    #    0 = body level
    #   -1 = nose straight down (forward flop)
    # nose_down = max(0, -fwd_z) isolates the forward-flop failure mode.
    nose_down_sum = [0.0]
    nose_down_sq_sum = [0.0]
    max_nose_down = [0.0]
    # Pitch angular velocity — catches the "flop" (fast rotation) even if the
    # absolute angle hasn't built up yet.
    pitch_rate_sq_sum = [0.0]
    max_pitch_rate = [0.0]

    def body_rp():
        qw, qx, qy, qz = data.qpos[3:7]
        roll = np.arctan2(2 * (qw * qx + qy * qz),
                          1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        return roll, pitch

    def body_fwd_z():
        """World-z component of body-x axis. <0 = nose pointed down."""
        mat = data.xmat[base_body_id].reshape(3, 3)
        return float(mat[2, 0])

    def is_fallen():
        if data.qpos[2] < FALL_Z:
            return True
        roll, pitch = body_rp()
        if abs(roll) > FALL_TILT or abs(pitch) > FALL_TILT:
            return True
        # Hard-kill rollouts that pitch nose-down past the threshold — forward
        # flop is the specific failure mode we're fighting.
        if -body_fwd_z() > FALL_NOSE_DOWN:
            return True
        return False

    def sample_attitude():
        roll, pitch = body_rp()
        pitch_sq_sum[0] += pitch * pitch
        roll_sq_sum[0] += roll * roll
        pitch_abs_sum[0] += abs(pitch)
        roll_abs_sum[0] += abs(roll)
        if abs(pitch) > max_pitch[0]:
            max_pitch[0] = abs(pitch)
        if abs(roll) > max_roll[0]:
            max_roll[0] = abs(roll)

        # Forward-flop specific: nose-down component only.
        nd = max(0.0, -body_fwd_z())
        nose_down_sum[0] += nd
        nose_down_sq_sum[0] += nd * nd
        if nd > max_nose_down[0]:
            max_nose_down[0] = nd

        # Pitch rate (body-frame angular velocity about y).
        pr = float(data.qvel[4])
        pitch_rate_sq_sum[0] += pr * pr
        if abs(pr) > max_pitch_rate[0]:
            max_pitch_rate[0] = abs(pr)

        tilt_n_steps[0] += 1

    def run_phase(from_pose, to_pose, duration):
        steps = max(1, int(duration / dt))
        phase_wall = time.time() if viewer else 0
        for i in range(steps):
            t = i / steps
            set_ctrl(lerp_pose(from_pose, to_pose, t, interp))
            mujoco.mj_step(model, data)
            sample_attitude()
            if is_fallen():
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
        sample_attitude()
        return not is_fallen()

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

    n_samp = max(1, tilt_n_steps[0])
    mean_pitch_sq = pitch_sq_sum[0] / n_samp
    mean_roll_sq = roll_sq_sum[0] / n_samp
    mean_pitch_abs = pitch_abs_sum[0] / n_samp
    mean_roll_abs = roll_abs_sum[0] / n_samp
    mean_nose_down = nose_down_sum[0] / n_samp
    mean_nose_down_sq = nose_down_sq_sum[0] / n_samp
    mean_pitch_rate_sq = pitch_rate_sq_sum[0] / n_samp
    mean_z = z_sum / max(1, z_n)
    survival = phases_completed / max(1, total_phases)

    step_bonus = 0.5 * phases_completed
    # Distance reward (covers ground) + speed bonus (rewards pace).
    # Speed is m/s averaged over the whole rollout; elapsed time comes from
    # the per-physics-step sample count × timestep.
    elapsed_time = max(dt, tilt_n_steps[0] * dt)
    speed = max(0, earned_distance) / elapsed_time
    forward = max(0, earned_distance) * 100.0 + speed * 1000.0
    backward_penalty = max(0, -earned_distance) * 200.0
    height_bonus = 5.0 * mean_z if phases_completed > 0 else 0.0

    # Symmetric pitch/roll — keeps the body from hanging at an angle.
    # Baseline weights doubled from the previous version; `tilt_scale` lets
    # sweep jobs dial strictness further up (strict) or down (lenient).
    pitch_penalty = tilt_scale * 60.0 * mean_pitch_sq
    roll_penalty = tilt_scale * 80.0 * mean_roll_sq
    peak_penalty = tilt_scale * (16.0 * max_pitch[0] ** 2
                                 + 16.0 * max_roll[0] ** 2)

    # Directional forward-flop penalty — front-leg collapse is the dominant
    # failure mode we're fighting. nose-down ranges 0..1; penalize its square
    # and hammer the worst moment.
    flop_penalty = tilt_scale * (800.0 * mean_nose_down_sq
                                 + 250.0 * (max_nose_down[0] ** 2))

    # Pitch-rate penalty catches fast floppy motions that don't linger in the
    # mean. mean_pitch_rate_sq is in (rad/s)²; typical gentle walking < 2, a
    # flop spikes to 10+.
    pitch_rate_penalty = tilt_scale * (8.0 * mean_pitch_rate_sq
                                       + 3.0 * (max_pitch_rate[0] ** 2))

    reward = (step_bonus + forward - backward_penalty + height_bonus
              - pitch_penalty - roll_penalty - peak_penalty
              - flop_penalty - pitch_rate_penalty)

    info = {
        "earned_dist_m": earned_distance,
        "elapsed_s": elapsed_time,
        "speed_mps": speed,
        "mean_pitch_deg": np.degrees(mean_pitch_abs),
        "mean_roll_deg": np.degrees(mean_roll_abs),
        "max_pitch_deg": np.degrees(max_pitch[0]),
        "max_roll_deg": np.degrees(max_roll[0]),
        "mean_nose_down_deg": np.degrees(np.arcsin(min(1.0, mean_nose_down))),
        "max_nose_down_deg": np.degrees(np.arcsin(min(1.0, max_nose_down[0]))),
        "max_pitch_rate": max_pitch_rate[0],
        "mean_z": mean_z,
        "final_z": float(data.qpos[2]),
        "min_z": min_z,
        "phases": f"{phases_completed}/{total_phases}",
        "survival": f"{survival:.0%}",
    }
    return reward, info


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
         fall_tilt_deg=20.0, tilt_scale=1.0, interp="linear"):
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

    sigma = 5.0
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
            tune(model, resume=args.resume, **common)
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
