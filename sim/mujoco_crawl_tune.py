#!/usr/bin/env python3
"""
Physics-driven crawl gait + CMA-ES tuning.

Modes:
  mjpython mujoco_crawl_tune.py              # run current params in viewer
  python   mujoco_crawl_tune.py --tune       # headless CMA-ES tuning
  mjpython mujoco_crawl_tune.py --replay best_params.json

Parameter vector (13 floats):
  0..11 : knee values for shift/plant phases in order
          [shift_FR_RL, shift_FR_FL, shift_FR_RR,
           plant_FR,    shift_RL_RR, shift_RL_FL,
           plant_RL,    shift_FL_RR, plant_FL,
           shift_RR_FL, shift_RR_RL, plant_RR]
  12    : phase_seconds
"""
import argparse, json, sys, time
import numpy as np
import mujoco

JOINTS = ["hip_fl","knee_fl","hip_fr","knee_fr","hip_rl","knee_rl","hip_rr","knee_rr"]

# ---------------------------------------------------------------- build model
def build_model():
    spec = mujoco.MjSpec.from_file("optimus_primal.urdf")

    # Scene
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.4,0.6,0.9], rgb2=[0.1,0.15,0.25],
                     width=512, height=512)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.25,0.26,0.27], rgb2=[0.32,0.33,0.34],
                     width=512, height=512)
    spec.add_material(name="grid", textures=["","grid"], texrepeat=[10,10], reflectance=0.1)
    spec.worldbody.add_light(pos=[0,0,3], dir=[0,0,-1], diffuse=[0.9,0.9,0.9])
    spec.worldbody.add_camera(name="chase", pos=[0.6, -0.8, 0.4],
                              xyaxes=[1, 0.75, 0, -0.2, 0.25, 1])
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[5,5,0.1], material="grid",
                            friction=[1.0, 0.05, 0.001])

    # Free joint on base_link so the body can fall / tip
    base = spec.body("base_link")
    base.add_freejoint()

    # Position actuators (one per joint)
    KP = 300.0
    KV = 15.0
    FMAX = 30.0  # huge — unrealistic but lets us see if it stands at all
    for j in JOINTS:
        a = spec.add_actuator(
            name=j + "_pos",
            target=j,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[KP, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0, -KP, -KV, 0, 0, 0, 0, 0, 0, 0],
            forcerange=[-FMAX, FMAX],
            ctrlrange=[-3.14, 3.14],
        )
    return spec.compile()

# ---------------------------------------------------------- phase generation
def phases_from_params(p):
    """Return list of (leg_targets_dict_in_degrees, duration_s)."""
    (s_fr_rl, s_fr_fl, s_fr_rr,
     pl_fr,
     s_rl_rr, s_rl_fl,
     pl_rl,
     s_fl_rr, pl_fl,
     s_rr_fl, s_rr_rl, pl_rr) = p[:12]
    dt = float(p[12])

    # each phase is (updates_dict, duration)
    # updates_dict: {"FL":(hip,knee), ...} — only listed legs change
    return [
        ({"FL":(35,-80),"FR":(35,-80),"RL":(35,-50),"RR":(35,-50)}, dt),  # start
        ({"RL":(35,s_fr_rl),"FL":(35,s_fr_fl),"RR":(35,s_fr_rr)}, dt),    # shift_FR
        ({"FR":(10,-110)}, dt),                                            # swing_FR
        ({"FR":(10,pl_fr)}, dt),                                           # plant_FR
        ({"FR":(10,-80),"RR":(35,s_rl_rr),"FL":(35,s_rl_fl)}, dt),        # shift_RL
        ({"RL":(10,-75)}, dt),                                             # swing_RL
        ({"RL":(10,pl_rl)}, dt),                                           # plant_RL
        ({"RR":(35,s_fl_rr),"FR":(10,-95),"RL":(10,-35)}, dt),            # shift_FL
        ({"FL":(10,-110)}, dt),                                            # swing_FL
        ({"FL":(10,pl_fl)}, dt),                                           # plant_FL
        ({"FL":(10,s_rr_fl),"RL":(10,s_rr_rl),"FR":(10,-65)}, dt),        # shift_RR
        ({"RR":(10,-75)}, dt),                                             # swing_RR
        ({"RR":(10,pl_rr)}, dt),                                           # plant_RR
    ]

DEFAULT_PARAMS = np.array([
    -65, -95, -40,   # shift_FR
    -65,             # plant_FR
    -65, -65,        # shift_RL
    -35,             # plant_RL
    -65, -65,        # shift_FL + plant_FL
    -95, -50, -35,   # shift_RR + plant_RR
    0.6,             # phase_seconds
], dtype=float)

# ---------------------------------------------------------------- rollout
def apply_phase(targets, current_deg, updates):
    current_deg.update(updates)
    for name,(h,k) in current_deg.items():
        targets[f"hip_{name.lower()}"]  = np.radians(h)
        targets[f"knee_{name.lower()}"] = np.radians(k)

def rollout(model, params, n_strides=5, viewer=None):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)   # initializes quat to (1,0,0,0)

    ctrl_idx = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j+"_pos") for j in JOINTS}
    qadr = {j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS}
    def set_targets(targets):
        for j, rad in targets.items():
            data.ctrl[ctrl_idx[j]] = rad

    phases = phases_from_params(params)
    targets = {j: 0.0 for j in JOINTS}
    current_deg = {}

    # Seed start stance into BOTH qpos and ctrl so the settle isn't a crash
    apply_phase(targets, current_deg, phases[0][0])
    for j, rad in targets.items():
        data.qpos[qadr[j]] = rad
    # Lift base so feet aren't clipping the floor at t=0
    data.qpos[2] = 0.25
    set_targets(targets)
    mujoco.mj_forward(model, data)
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(model.opt.timestep)
    if viewer is not None:
        print(f"  after settle: z={data.qpos[2]:.3f}  quat={data.qpos[3:7]}")

    # free joint: qpos[0..2]=xyz, qpos[3..6]=quat(w,x,y,z), then joints
    BASE_X, BASE_Y, BASE_Z = 0, 1, 2
    BASE_QW, BASE_QX, BASE_QY, BASE_QZ = 3, 4, 5, 6
    x0 = float(data.qpos[BASE_X])
    y0 = float(data.qpos[BASE_Y])
    z0 = float(data.qpos[BASE_Z])
    t0 = float(data.time)
    tilt_sum = 0.0
    tilt_n = 0
    terminated = False

    for _ in range(n_strides):
        for updates, duration in phases[1:]:
            apply_phase(targets, current_deg, updates)
            set_targets(targets)
            n_sub = max(1, int(duration / model.opt.timestep))
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(model.opt.timestep)
                # termination: body rolled/pitched too far or fell
                qw,qx,qy,qz = data.qpos[BASE_QW], data.qpos[BASE_QX], data.qpos[BASE_QY], data.qpos[BASE_QZ]
                # roll (x), pitch (y) from quaternion
                roll  = np.arctan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
                pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1))
                tilt_sum += abs(roll)+abs(pitch); tilt_n += 1
                if abs(roll) > 0.8 or abs(pitch) > 0.8 or data.qpos[BASE_Z] < 0.04:
                    terminated = True
                    if viewer is None:
                        break
            if terminated and viewer is None: break
        if terminated: break

    dx = float(data.qpos[BASE_X] - x0)
    dy = float(data.qpos[BASE_Y] - y0)
    dz = float(data.qpos[BASE_Z] - z0)
    elapsed = float(data.time - t0)
    mean_tilt = tilt_sum / max(1, tilt_n)
    # Cap forward credit at a realistic cm/stride so sim exploits don't pay off
    dx_capped = min(dx, 0.05 * n_strides)
    reward = (dx_capped
              - 3.0 * abs(dy)
              - 2.0 * mean_tilt
              - 10.0 * max(0, -dz)
              + 0.5 * elapsed
              - (50.0 if terminated else 0.0))
    return reward, dict(dx=dx, dy=dy, dz=dz, mean_tilt=mean_tilt,
                        elapsed=elapsed, terminated=terminated)

# ---------------------------------------------------------------- modes
def run_viewer(params):
    import mujoco.viewer
    model = build_model()
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(20):
            if not viewer.is_running(): break
            r, info = rollout(model, params, n_strides=3, viewer=viewer)
            print(f"[{i+1}/20] reward={r:+.3f} {info}")
            time.sleep(0.5)

def tune(out="best_params.json", generations=30, popsize=12, n_strides=5):
    try:
        import cma
    except ImportError:
        print("pip install cma", file=sys.stderr); sys.exit(1)
    model = build_model()
    x0 = DEFAULT_PARAMS.copy()
    sigma0 = np.array([8]*12 + [0.1])
    es = cma.CMAEvolutionStrategy(x0, 1.0,
        {"popsize": popsize,
         "CMA_stds": sigma0,
         "bounds": [[-130]*12+[0.2], [0]*12+[1.5]],
         "maxiter": generations,
         "verbose": -9})
    gen = 0
    best_r = -1e9; best_x = x0
    while not es.stop():
        xs = es.ask()
        fs = []
        for x in xs:
            r, _ = rollout(model, x, n_strides=n_strides)
            fs.append(-r)
            if r > best_r:
                best_r, best_x = r, x
        es.tell(xs, fs)
        gen += 1
        pop_mean = float(-np.mean(fs))
        print(f"gen {gen:3d}  best_reward={best_r:+.3f}  pop_mean={pop_mean:+.3f}", flush=True)
        with open(out, "w") as f:
            json.dump({"params": list(best_x), "reward": best_r, "gen": gen}, f, indent=2)
        with open("tune_history.jsonl", "a") as f:
            f.write(json.dumps({"gen": gen, "best": best_r, "mean": pop_mean}) + "\n")
    print(f"done. best reward {best_r:.3f} saved to {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--replay", type=str, default=None)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--strides", type=int, default=5)
    args = ap.parse_args()

    if args.tune:
        tune(generations=args.generations, popsize=args.popsize, n_strides=args.strides)
    elif args.replay:
        with open(args.replay) as f: d = json.load(f)
        run_viewer(np.array(d["params"]))
    else:
        run_viewer(DEFAULT_PARAMS)

if __name__ == "__main__":
    main()
