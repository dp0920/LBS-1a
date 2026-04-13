#!/usr/bin/env python3
"""
Balance on 3 legs — built directly on the working mujoco_trot.py pattern.
Uses qpos pinning (kinematic joints, free body) same as trot.

Usage:
  mjpython mujoco_balance.py              # 4-leg stand (sanity check)
  mjpython mujoco_balance.py --lift FL    # lift FL, balance on 3
  mjpython mujoco_balance.py --lift FR    # lift FR, balance on 3
  python   mujoco_balance.py --tune --lift FL   # CMA-ES tune (headless)
  python   mujoco_balance.py --tune --lift all  # tune all 4 legs
  mjpython mujoco_balance.py --replay best_balance_FL.json
"""
import argparse
import json
import sys
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

HIP_STAND = np.radians(35)
KNEE_STAND = -np.radians(80)

STAND = {
    "hip_fl": HIP_STAND, "knee_fl": KNEE_STAND,
    "hip_fr": HIP_STAND, "knee_fr": KNEE_STAND,
    "hip_rl": HIP_STAND, "knee_rl": KNEE_STAND,
    "hip_rr": HIP_STAND, "knee_rr": KNEE_STAND,
}

LIFT_HIP = np.radians(10)
LIFT_KNEE = -np.radians(130)


def build_model():
    spec = mujoco.MjSpec.from_file("optimus_primal.urdf")
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
    return spec.compile()


def get_qadr(model):
    qadr = {}
    for j in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qadr[j] = model.jnt_qposadr[jid]
    return qadr


def make_3leg_pose(lift_leg, params=None):
    """Build pose with one leg lifted. params = 6 floats (hip_deg, knee_deg)
    for each grounded leg in alphabetical order."""
    pose = dict(STAND)
    grounded = sorted([n for n in LEGS if n != lift_leg])

    if params is not None:
        for i, name in enumerate(grounded):
            hip_j, knee_j = LEGS[name]
            pose[hip_j] = np.radians(params[i * 2])
            pose[knee_j] = np.radians(params[i * 2 + 1])

    if lift_leg:
        hip_j, knee_j = LEGS[lift_leg]
        pose[hip_j] = LIFT_HIP
        pose[knee_j] = LIFT_KNEE

    return pose


def lerp_pose(pose_a, pose_b, t):
    pose = {}
    for j in pose_a:
        pose[j] = pose_a[j] * (1 - t) + pose_b[j] * t
    return pose


def rollout(model, lift_leg, params=None, hold_time=5.0, viewer=None,
            data=None):
    """Run one balance test. Returns reward + info."""
    if data is None:
        data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qadr = get_qadr(model)

    def apply_pose(pose):
        for name, angle in pose.items():
            data.qpos[qadr[name]] = angle

    # Init: spawn already crouched at correct height (no drop/transition)
    apply_pose(STAND)
    data.qpos[2] = 0.16   # ~leg height in crouched pose
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    wall_start = time.time() if viewer else None

    def pace(step_i, phase_start_wall, phase_start_sim_step):
        if viewer:
            viewer.sync()
            sim_elapsed = (step_i - phase_start_sim_step + 1) * dt
            wall_elapsed = time.time() - phase_start_wall
            if sim_elapsed > wall_elapsed:
                time.sleep(sim_elapsed - wall_elapsed)

    # Phase 1: hold standing (2s settle) — zero body velocity to prevent sliding
    if viewer: print(">>> PHASE 1: settling (2s)...")
    phase_wall = time.time() if viewer else 0
    for i in range(int(2.0 / dt)):
        apply_pose(STAND)
        data.qvel[0:6] = 0  # zero base linear + angular velocity
        mujoco.mj_step(model, data)
        pace(i, phase_wall, 0)
    if viewer:
        print(f"    settled at z={data.qpos[2]:.3f}")

    # Phase 2: smooth transition to 3-leg pose (2s)
    if viewer: print(f">>> PHASE 2: lifting {lift_leg} (2s)...")
    target_pose = make_3leg_pose(lift_leg, params)
    phase_wall = time.time() if viewer else 0
    lift_steps = int(2.0 / dt)
    for i in range(lift_steps):
        t = i / lift_steps
        pose = lerp_pose(STAND, target_pose, t)
        apply_pose(pose)
        mujoco.mj_step(model, data)
        pace(i, phase_wall, 0)

    if viewer:
        z = data.qpos[2]
        quat = data.qpos[3:7]
        print(f"  after lift: z={z:.3f}  "
              f"quat=[{quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f}]")

    # Phase 3: hold 3-leg pose (5s) — measure balance
    if viewer: print(">>> PHASE 3: holding balance (5s)...")
    tilt_sum = 0.0
    tilt_n = 0
    z_start = float(data.qpos[2])
    phase_wall = time.time() if viewer else 0

    for i in range(int(hold_time / dt)):
        apply_pose(target_pose)
        mujoco.mj_step(model, data)
        pace(i, phase_wall, 0)

        qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        tilt_sum += abs(roll) + abs(pitch)
        tilt_n += 1

        if viewer and not viewer.is_running():
            break

    mean_tilt = tilt_sum / max(1, tilt_n)
    final_z = float(data.qpos[2])
    z_drop = max(0, z_start - final_z)

    # Reward: minimize tilt and z drop
    # Must maintain standing height (~0.10m) — penalize heavily if body drops
    min_acceptable_z = 0.07
    z_penalty = max(0, min_acceptable_z - final_z) * 200.0  # massive penalty for collapsing
    reward = -5.0 * mean_tilt - 50.0 * z_drop - z_penalty

    return reward, dict(
        mean_tilt_deg=np.degrees(mean_tilt),
        final_z=final_z,
        z_drop=z_drop,
        roll_deg=np.degrees(roll),
        pitch_deg=np.degrees(pitch),
    )


def run_viewer(model, lift_leg, params=None):
    import mujoco.viewer
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(3):
            if not viewer.is_running():
                break
            r, info = rollout(model, lift_leg, params, hold_time=5.0,
                              viewer=viewer, data=data)
            print(f"[{i + 1}/3] reward={r:+.3f} {info}")
        print("Done. Close viewer to exit.")
        while viewer.is_running():
            time.sleep(0.1)


def tune(model, lift_leg, out=None, generations=50, popsize=16):
    try:
        import cma
    except ImportError:
        print("pip install cma", file=sys.stderr)
        sys.exit(1)

    if out is None:
        out = f"best_balance_{lift_leg}.json"

    # 6 params: (hip_deg, knee_deg) for each of the 3 grounded legs
    # Start from default standing angles
    x0 = np.array([35, -80, 35, -80, 35, -80], dtype=float)

    es = cma.CMAEvolutionStrategy(x0, 10.0, {
        "popsize": popsize,
        "bounds": [[ 10, -110,  10, -110,  10, -110],
                   [ 60,  -40,  60,  -40,  60,  -40]],
        "maxiter": generations,
        "verbose": -9,
    })

    best_r = -1e9
    best_x = x0.copy()
    gen = 0
    grounded = sorted([n for n in LEGS if n != lift_leg])
    print(f"Tuning balance for lifting {lift_leg}")
    print(f"  grounded legs (param order): {grounded}")
    print(f"  x0 = {list(x0)}")

    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            r, _ = rollout(model, lift_leg, params=x, hold_time=5.0)
            fs.append(-r)
            if r > best_r:
                best_r = r
                best_x = x.copy()
        es.tell(xs, fs)
        gen += 1
        pop_mean = float(-np.mean(fs))
        print(f"gen {gen:3d}  best={best_r:+.3f}  mean={pop_mean:+.3f}",
              flush=True)

        result = {
            "lift_leg": lift_leg,
            "grounded_legs": grounded,
            "params": list(best_x),
            "params_labels": [f"{g}_hip" if i%2==0 else f"{g}_knee"
                              for i, g in enumerate(g for g in grounded
                                                    for _ in range(2))],
            "reward": best_r,
            "gen": gen,
        }
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        with open(f"tune_balance_{lift_leg}.jsonl", "a") as f:
            f.write(json.dumps({"gen": gen, "best": best_r,
                                "mean": pop_mean}) + "\n")

    # Print final angles in a useful format
    print(f"\ndone. best reward {best_r:.3f}")
    print(f"Grounded leg angles when lifting {lift_leg}:")
    for i, name in enumerate(grounded):
        h, k = best_x[i*2], best_x[i*2+1]
        print(f"  {name}: hip={h:+.1f}°  knee={k:+.1f}°")
    print(f"Saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", type=str, default=None,
                    help="Leg to lift: FL FR RL RR, or 'all'")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--replay", type=str, default=None)
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--popsize", type=int, default=16)
    args = ap.parse_args()

    model = build_model()

    if args.replay:
        with open(args.replay) as f:
            d = json.load(f)
        run_viewer(model, d["lift_leg"], np.array(d["params"]))
    elif args.tune:
        if not args.lift:
            print("--lift required with --tune"); sys.exit(1)
        legs = ["FL", "FR", "RL", "RR"] if args.lift.upper() == "ALL" else [args.lift.upper()]
        for leg in legs:
            tune(model, leg, generations=args.generations,
                 popsize=args.popsize)
    else:
        lift = args.lift.upper() if args.lift else None
        run_viewer(model, lift)


if __name__ == "__main__":
    main()
