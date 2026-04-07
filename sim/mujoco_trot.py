import mujoco
import mujoco.viewer
import numpy as np

# ── Load model ─────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path("optimus_primal.urdf")
data = mujoco.MjData(model)

# ── Print joint map ─────────────────────────────────────────────────────────
print("\n── Joint map ──────────────────────────────")
for i in range(model.njnt):
    print(f"  [{i}] {model.joint(i).name}")
print(f"Actuators: {model.nu}")
print("───────────────────────────────────────────\n")

# ── Standing pose angles (from real servo diffs) ───────────────────────────
HIP_STAND  =  np.radians(35)
KNEE_STAND = -np.radians(80)

def apply_pose(data, pose: dict):
    """Drive joints directly via qpos (no actuators needed)."""
    for name, angle in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle

STAND = {
    "hip_fl": HIP_STAND,  "knee_fl": KNEE_STAND,
    "hip_fr": HIP_STAND,  "knee_fr": KNEE_STAND,
    "hip_rl": HIP_STAND,  "knee_rl": KNEE_STAND,
    "hip_rr": HIP_STAND,  "knee_rr": KNEE_STAND,
}

# ── Trot gait ───────────────────────────────────────────────────────────────
def trot_pose(t, freq=0.8, hip_amp=0.25, knee_lift=0.35):
    phase = 2 * np.pi * freq * t
    pose = {}
    pairs = [
        ("hip_fl", "knee_fl", 0),
        ("hip_rr", "knee_rr", 0),
        ("hip_fr", "knee_fr", np.pi),
        ("hip_rl", "knee_rl", np.pi),
    ]
    for hip_name, knee_name, offset in pairs:
        p = phase + offset
        pose[hip_name]  = HIP_STAND + hip_amp * np.sin(p)
        pose[knee_name] = KNEE_STAND - knee_lift * max(0, np.sin(p))
    return pose

# ── Main loop ───────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Settling into standing pose (500 steps)...")
    for _ in range(500):
        apply_pose(data, STAND)
        mujoco.mj_step(model, data)
        viewer.sync()

    print("Starting trot gait — watch the viewer!")
    sim_time = 0.0
    dt = model.opt.timestep
    step = 0

    while viewer.is_running():
        apply_pose(data, trot_pose(sim_time))
        mujoco.mj_step(model, data)
        viewer.sync()

        sim_time += dt
        step += 1

        if step % int(3.0 / dt) == 0:
            x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
            print(f"t={sim_time:.1f}s  x={x*100:.1f}cm  body_z={z*100:.1f}cm")
