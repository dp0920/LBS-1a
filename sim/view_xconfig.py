#!/usr/bin/env python3
"""
Visual sanity check of the X-config URDF.

Opens the MuJoCo viewer with the optimus_primal_xconfig.urdf loaded and
holds the calibrated level-X stance the user captured kinesthetically.

Usage:
  mjpython view_xconfig.py                  # X-config URDF + level X pose
  mjpython view_xconfig.py --mammal         # plain URDF + symmetric stance
  mjpython view_xconfig.py --z=0.10         # drop closer to the ground
  mjpython view_xconfig.py --hip-mult=1.3   # widen the stance (more out-swing)
"""
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from sim_core import build_model, get_qadr, JOINTS

USE_MAMMAL = "--mammal" in sys.argv
INIT_Z = 0.12   # default drop height — was ~0.18 implicitly via URDF base
HIP_MULT = 1.0  # multiplier on hip values (>1 widens the support polygon)
for arg in sys.argv[1:]:
    if arg.startswith("--z="):
        INIT_Z = float(arg.split("=", 1)[1])
    elif arg.startswith("--hip-mult="):
        HIP_MULT = float(arg.split("=", 1)[1])

if USE_MAMMAL:
    urdf = "optimus_primal.urdf"
    pose = {"hip_fl": +35, "knee_fl": -80,
            "hip_fr": +35, "knee_fr": -80,
            "hip_rl": +35, "knee_rl": -80,
            "hip_rr": +35, "knee_rr": -80}
else:
    urdf = "optimus_primal_xconfig.urdf"
    # Symmetric X-config seed (averages of user's manual capture, with
    # per-side asymmetry removed since sim has no calibration drift).
    # Front legs:  hip ≈ +42, knee ≈ -59
    # Rear  legs:  hip ≈ +39, knee ≈ -72
    pose = {"hip_fl": +42, "knee_fl": -59,
            "hip_fr": +42, "knee_fr": -59,
            "hip_rl": +39, "knee_rl": -72,
            "hip_rr": +39, "knee_rr": -72}

# Apply hip-multiplier to widen the stance (push feet outward fore-aft)
for k in list(pose):
    if k.startswith("hip_"):
        pose[k] = pose[k] * HIP_MULT

model = build_model(urdf=urdf)
data = mujoco.MjData(model)
qadr = get_qadr(model)

# Lower the chassis closer to the ground before settling.
# qpos layout: free-joint at indices 0..6, then 8 actuated joints.
data.qpos[2] = INIT_Z

# Set joint angles
for j, deg in pose.items():
    data.qpos[qadr[j]] = np.radians(deg)
mujoco.mj_forward(model, data)

# Drive actuators to the same pose so the body settles under gravity
ctrl_idx = {j: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                  f"act_{j}") for j in JOINTS}
for j, deg in pose.items():
    data.ctrl[ctrl_idx[j]] = np.radians(deg)

print(f"URDF: {urdf}")
print(f"Pose: {pose}")
print("Launching viewer. Close window to exit.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
