"""
Shared MuJoCo plumbing for the Optimus Primal quadruped.

Pure sim building blocks — no opinions about control scheme, reward shaping,
or episode structure. Used by both the CMA-ES gait tuner (`mujoco_gait.py`)
and the RL gym env (`gym_env.py`).
"""
import os
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


def build_model(kp=2.5, kv=0.05, urdf="optimus_primal.urdf"):
    """Load the URDF and attach floor/lights/actuators. Returns a compiled
    MjModel. Uses the MjSpec declarative API (requires mujoco>=3.2).

    kp/kv are the position-actuator PD gains.  The defaults (kp=2.5,
    kv=0.05) match the original training setup but produce an underdamped
    response that PPO learns to exploit as a low-pass filter — making
    policies non-deployable to LX-16A hardware.  Higher values (e.g.
    kp=20, kv=1.0) force near-instant joint tracking so the action stream
    IS the joint trajectory and open-loop replay works.

    `urdf` selects which URDF to load. Use 'optimus_primal_xconfig.urdf'
    for the ANYmal X-config variant (rear knee axes flipped, rear knee
    joint range becomes [0, +120°] instead of [-120°, 0])."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, urdf)
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
        act.set_to_position(kp=kp, kv=kv)
    return spec.compile()


def get_qadr(model):
    """Map joint name → qpos address."""
    qadr = {}
    for j in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qadr[j] = model.jnt_qposadr[jid]
    return qadr


def get_ctrl_idx(model):
    """Map joint name → actuator control index."""
    ctrl = {}
    for j in JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                f"act_{j}")
        ctrl[j] = aid
    return ctrl


def get_base_body_id(model):
    """Return the MuJoCo body id for base_link (chassis)."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")


def body_rp(data):
    """Roll and pitch of the chassis (radians) from the free-joint quaternion."""
    qw, qx, qy, qz = data.qpos[3:7]
    roll = np.arctan2(2 * (qw * qx + qy * qz),
                      1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    return roll, pitch


def body_fwd_z(data, base_body_id):
    """World-z component of the body-x axis.
      +1 = nose pointing straight up
       0 = body level
      -1 = nose pointing straight down (forward flop)
    """
    mat = data.xmat[base_body_id].reshape(3, 3)
    return float(mat[2, 0])
