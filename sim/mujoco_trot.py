import time
import mujoco
import mujoco.viewer
import numpy as np

# ── Load model via MjSpec so we can add floor + free joint ─────────────────
spec = mujoco.MjSpec.from_file("optimus_primal.urdf")

# Scene: skybox + checker floor + light
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
spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.9, 0.9, 0.9])
spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[5, 5, 0.1], material="grid",
                        friction=[1.0, 0.05, 0.001])

# Free joint so body responds to gravity
base = spec.body("base_link")
base.add_freejoint()

model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# ── Joint map ──────────────────────────────────────────────────────────────
JOINTS = ["hip_fl", "knee_fl", "hip_fr", "knee_fr",
          "hip_rl", "knee_rl", "hip_rr", "knee_rr"]
qadr = {}
for j in JOINTS:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    qadr[j] = model.jnt_qposadr[jid]

print("\n── Joint map ──────────────────────────────")
for j in JOINTS:
    print(f"  {j:12s}  qposadr={qadr[j]}")
print(f"  nq={model.nq}  (7 free + 8 joints = 15)")
print("───────────────────────────────────────────\n")

# ── Standing pose angles (from real servo diffs) ─────────────────────────
HIP_STAND = np.radians(35)
KNEE_STAND = -np.radians(80)

STAND = {
    "hip_fl": HIP_STAND,  "knee_fl": KNEE_STAND,
    "hip_fr": HIP_STAND,  "knee_fr": KNEE_STAND,
    "hip_rl": HIP_STAND,  "knee_rl": KNEE_STAND,
    "hip_rr": HIP_STAND,  "knee_rr": KNEE_STAND,
}

def apply_pose(pose: dict):
    """Drive joints via qpos (kinematic — joints are pinned to targets)."""
    for name, angle in pose.items():
        data.qpos[qadr[name]] = angle

# ── Trot gait ──────────────────────────────────────────────────────────────
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
        pose[hip_name] = HIP_STAND + hip_amp * np.sin(p)
        pose[knee_name] = KNEE_STAND - knee_lift * max(0, np.sin(p))
    return pose

# ── Initialize: set pose + drop from height ───────────────────────────────
apply_pose(STAND)
data.qpos[2] = 0.3       # base z — drop from 30cm
data.qpos[3] = 1.0       # quaternion w (identity orientation)
mujoco.mj_forward(model, data)

dt = model.opt.timestep

# ── Main loop ──────────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Settling (1s drop)...")
    wall_start = time.time()
    for _ in range(int(1.0 / dt)):
        apply_pose(STAND)
        mujoco.mj_step(model, data)
        viewer.sync()
        # Real-time pacing
        elapsed = time.time() - wall_start
        sim_time = data.time
        if sim_time > elapsed:
            time.sleep(sim_time - elapsed)

    z = data.qpos[2]
    quat = data.qpos[3:7]
    print(f"After settle: z={z:.3f}m  quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")

    print("Starting trot gait — watch the viewer!")
    sim_time = 0.0
    step = 0
    wall_start = time.time()

    while viewer.is_running():
        apply_pose(trot_pose(sim_time))
        mujoco.mj_step(model, data)
        viewer.sync()

        sim_time += dt
        step += 1

        # Real-time pacing
        wall_elapsed = time.time() - wall_start
        if sim_time > wall_elapsed:
            time.sleep(sim_time - wall_elapsed)

        if step % int(3.0 / dt) == 0:
            x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
            print(f"t={sim_time:.1f}s  x={x*100:.1f}cm  y={y*100:.1f}cm  body_z={z*100:.1f}cm")
