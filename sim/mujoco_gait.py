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


def lerp_pose(a, b, t):
    return {j: a[j] * (1 - t) + b[j] * t for j in a}


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


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(model, poses, phase_time, n_cycles=5, viewer=None, data=None):
    """Run gait for n_cycles. Returns (reward, info)."""
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
    FALL_TILT = np.radians(45)

    def is_fallen():
        if data.qpos[2] < FALL_Z:
            return True
        qw, qx, qy, qz = data.qpos[3:7]
        roll = abs(np.arctan2(2 * (qw * qx + qy * qz),
                              1 - 2 * (qx * qx + qy * qy)))
        pitch = abs(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
        return roll > FALL_TILT or pitch > FALL_TILT

    def run_phase(from_pose, to_pose, duration):
        steps = max(1, int(duration / dt))
        phase_wall = time.time() if viewer else 0
        for i in range(steps):
            t = i / steps
            set_ctrl(lerp_pose(from_pose, to_pose, t))
            mujoco.mj_step(model, data)
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
    tilt_sum = 0.0
    tilt_n = 0
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

            # Track metrics
            qw, qx, qy, qz = data.qpos[3:7]
            roll = np.arctan2(2 * (qw * qx + qy * qz),
                              1 - 2 * (qx * qx + qy * qy))
            pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
            tilt_sum += abs(roll) + abs(pitch)
            tilt_n += 1
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

    mean_tilt = tilt_sum / max(1, tilt_n)
    mean_z = z_sum / max(1, z_n)
    survival = phases_completed / max(1, total_phases)

    step_bonus = 0.5 * phases_completed
    forward = max(0, earned_distance) * 100.0
    backward_penalty = max(0, -earned_distance) * 200.0
    height_bonus = 5.0 * mean_z if phases_completed > 0 else 0.0
    reward = step_bonus + forward - backward_penalty + height_bonus - 2.0 * mean_tilt

    info = {
        "earned_dist_m": earned_distance,
        "mean_tilt_deg": np.degrees(mean_tilt),
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

def _worker_init():
    """Each worker builds its own MuJoCo model (can't pickle them)."""
    global _worker_model, _worker_data
    _worker_model = build_model()
    _worker_data = mujoco.MjData(_worker_model)


def _worker_eval(args):
    """Evaluate one candidate. Returns (reward, info)."""
    x, n_cycles = args
    poses, phase_time = decode_params(x)
    return rollout(_worker_model, poses, phase_time,
                   n_cycles=n_cycles, data=_worker_data)


# ---------------------------------------------------------------------------
# Viewer / demo
# ---------------------------------------------------------------------------

def run_viewer(model, poses, phase_time, n_cycles=5):
    import mujoco.viewer
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        r, info = rollout(model, poses, phase_time,
                          n_cycles=n_cycles, viewer=viewer, data=data)
        print(f"reward={r:+.4f}  {info}")
        print("Done. Close viewer to exit.")
        while viewer.is_running():
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# CMA-ES tuning
# ---------------------------------------------------------------------------

def tune(model, out="best_gait.json", generations=150, popsize=48,
         n_cycles=5, resume=False, workers=None):
    try:
        import cma
    except ImportError:
        print("pip install cma", file=sys.stderr)
        sys.exit(1)

    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    # Resume from saved best if available
    x0 = X0.copy()
    best_r = -1e9
    if resume and os.path.exists(out):
        with open(out) as f:
            d = json.load(f)
        x0 = np.clip(np.array(d["params"]), BOUNDS_LO, BOUNDS_HI)
        best_r = d.get("reward", -1e9)
        print(f"Resuming from {out} (reward={best_r:+.4f}, clamped to bounds)")

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

    pool = mp.Pool(workers, initializer=_worker_init)
    print(f"  workers: {workers}")

    try:
      while not es.stop():
        xs = es.ask()
        results = pool.map(_worker_eval, [(x, n_cycles) for x in xs])
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
                      f"phases={best_info_this_gen['phases']} "
                      f"z={best_info_this_gen['mean_z']:.3f}")
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

        result = {
            "params": list(best_x),
            "reward": best_r,
            "gen": gen,
            "restart": restart,
        }
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        with open("tune_gait.jsonl", "a") as f:
            f.write(json.dumps({"gen": gen, "best": best_r,
                                "mean": pop_mean,
                                "pop_best": pop_best}) + "\n")
    finally:
        pool.terminate()
        pool.join()

    # Print final phase angles
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
    args = ap.parse_args()

    model = build_model()

    if args.replay:
        with open(args.replay) as f:
            d = json.load(f)
        poses, phase_time = decode_params(np.array(d["params"]))
        run_viewer(model, poses, phase_time, n_cycles=args.cycles)
    elif args.tune or args.resume:
        tune(model, generations=args.generations,
             popsize=args.popsize, n_cycles=args.cycles,
             resume=args.resume, workers=args.workers)
    elif args.demo:
        poses, phase_time = decode_params(X0)
        run_viewer(model, poses, phase_time, n_cycles=args.cycles)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
