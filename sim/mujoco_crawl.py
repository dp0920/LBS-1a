#!/usr/bin/env python3
"""
Mirror of robot/stance.py 4-leg crawl in MuJoCo.
Joints driven kinematically via qpos; base is free so the body tips if unstable.

Phase function values (hip_off, knee_off in degrees) match stance.py raw leg()
convention: hip 35 = back stance, hip 10 = forward; knee -80 = leveled front,
more negative = more bent = foot up; less negative = foot drops.
"""
import time
import numpy as np
import mujoco
import mujoco.viewer

spec = mujoco.MjSpec.from_file("optimus_primal.urdf")
# Skybox + checker floor
spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                 builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                 rgb1=[0.4, 0.6, 0.9], rgb2=[0.1, 0.15, 0.25],
                 width=512, height=512)
spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                 builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                 rgb1=[0.25, 0.26, 0.27], rgb2=[0.32, 0.33, 0.34],
                 width=512, height=512)
spec.add_material(name="grid", textures=["", "grid"], texrepeat=[10, 10], reflectance=0.1)
spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.9, 0.9, 0.9])
spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[5, 5, 0.1], material="grid")
model = spec.compile()
data = mujoco.MjData(model)

JOINTS = ["hip_fl","knee_fl","hip_fr","knee_fr","hip_rl","knee_rl","hip_rr","knee_rr"]
QADR = {j: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS}

# target joint angles (radians) — updated by phase functions
target = {j: 0.0 for j in JOINTS}

def leg(name, hip_off, knee_off):
    """name in {FL,FR,RL,RR}; hip/knee in degrees, stance.py convention."""
    suf = name.lower()
    target[f"hip_{suf}"]  = np.radians(hip_off)
    target[f"knee_{suf}"] = np.radians(knee_off)

# ---- PHASE FUNCTIONS (copy of stance.py) ----
def crawl_start():
    leg("FL", 35, -80); leg("FR", 35, -80); leg("RL", 35, -50); leg("RR", 35, -50)

def shift_FR():
    leg("RL", 35, -65); leg("FL", 35, -95); leg("RR", 35, -40)
def swing_FR(): leg("FR", 10, -110)
def plant_FR(): leg("FR", 10, -65)

def shift_RL():
    leg("FR", 10, -80); leg("RR", 35, -65); leg("FL", 35, -65)
def swing_RL(): leg("RL", 10, -75)
def plant_RL(): leg("RL", 10, -35)

def shift_FL():
    leg("RR", 35, -65); leg("FR", 10, -95); leg("RL", 10, -35)
def swing_FL(): leg("FL", 10, -110)
def plant_FL(): leg("FL", 10, -65)

def shift_RR():
    leg("FL", 10, -95); leg("RL", 10, -50); leg("FR", 10, -65)
def swing_RR(): leg("RR", 10, -75)
def plant_RR(): leg("RR", 10, -35)

PHASES = [
    crawl_start,
    shift_FR, swing_FR, plant_FR,
    shift_RL, swing_RL, plant_RL,
    shift_FL, swing_FL, plant_FL,
    shift_RR, swing_RR, plant_RR,
]

PHASE_SECONDS = 0.6

def apply_targets():
    for j, a in target.items():
        data.qpos[QADR[j]] = a

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Settle into start stance
    crawl_start()
    for _ in range(500):
        apply_targets()
        mujoco.mj_step(model, data)
        viewer.sync()

    dt = model.opt.timestep
    steps_per_phase = max(1, int(PHASE_SECONDS / dt))

    x0 = data.qpos[0]
    stride_count = 0
    print("Running crawl stride loop. Close viewer to stop.")

    while viewer.is_running():
        for fn in PHASES:
            fn()
            print(f"  {fn.__name__}")
            for _ in range(steps_per_phase):
                apply_targets()
                mujoco.mj_step(model, data)
                viewer.sync()
                if not viewer.is_running():
                    break
            if not viewer.is_running():
                break
        stride_count += 1
        x = data.qpos[0]
        print(f"stride {stride_count}  dx={100*(x-x0):+.1f} cm  body_z={100*data.qpos[2]:.1f} cm")
