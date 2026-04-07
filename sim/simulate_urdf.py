"""
simulate_urdf.py — Optimus Primal PyBullet URDF loader
=======================================================
Place this file alongside your URDF and meshes folder:

  optimus_primal/
    simulate_urdf.py        ← this file
    optimus_primal.urdf
    meshes/
      base.stl
      upper_link.stl
      lower_link.stl

Run:
  python simulate_urdf.py

Dependencies:
  pip install pybullet
"""

import pybullet as p
import pybullet_data
import math
import time

# ── Joint name → index map (populated after load) ──────────────────────────
JOINT_MAP = {}

# ── Standing pose angles (radians, derived from servo diffs) ────────────────
# Hip  ~+35 deg forward from vertical  → +0.611 rad
# Knee ~-80 deg (folds back toward body) → -1.396 rad
#
# Left side servos push positive, right side push negative in real life.
# In URDF all joints share the same axis direction, so we mirror manually.
HIP_STAND  =  math.radians(35)   #  0.611 rad
KNEE_STAND = -math.radians(80)   # -1.396 rad

STAND_POSE = {
    "hip_fl":  HIP_STAND,
    "knee_fl": KNEE_STAND,
    "hip_fr":  HIP_STAND,
    "knee_fr": KNEE_STAND,
    "hip_rl":  HIP_STAND,
    "knee_rl": KNEE_STAND,
    "hip_rr":  HIP_STAND,
    "knee_rr": KNEE_STAND,
}


def build_joint_map(robot_id):
    """Populate JOINT_MAP with name → joint index."""
    n = p.getNumJoints(robot_id)
    print("\n── Joint map ──────────────────────────────")
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode()
        jtype = info[2]
        type_str = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}.get(jtype, str(jtype))
        print(f"  [{i:2d}] {name:<20s}  ({type_str})")
        JOINT_MAP[name] = i
    print("───────────────────────────────────────────\n")


def set_pose(robot_id, pose: dict, force: float = 2.5):
    """Drive named joints to target angles (radians) using position control."""
    for name, angle in pose.items():
        if name in JOINT_MAP:
            p.setJointMotorControl2(
                robot_id,
                JOINT_MAP[name],
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=force,
                maxVelocity=5.0,
            )


def get_body_height(robot_id) -> float:
    """Return the Z height of base_link in world frame (metres)."""
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    return pos[2]


def main():
    # ── Connect & configure ────────────────────────────────────────────────
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # Optional: nicer camera angle
    p.resetDebugVisualizerCamera(
        cameraDistance=0.6,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.1],
    )

    # ── Load ground plane ──────────────────────────────────────────────────
    plane_id = p.loadURDF("plane.urdf")

    # ── Load robot ─────────────────────────────────────────────────────────
    # Spawn high enough that legs don't clip the ground on load
    start_z = 0.30
    robot_id = p.loadURDF(
        "/workspace/optimus_primal.urdf",
        basePosition=[0, 0, start_z],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    build_joint_map(robot_id)

    # ── Settle into standing pose ──────────────────────────────────────────
    print("Settling into standing pose (500 steps)...")
    for _ in range(500):
        set_pose(robot_id, STAND_POSE)
        p.stepSimulation()

    height = get_body_height(robot_id)
    print(f"Body height after settling: {height*1000:.1f} mm")

    # ── Main loop ─────────────────────────────────────────────────────────
    print("\nSimulation running. Close the PyBullet window to exit.\n")
    step = 0
    try:
        while True:
            set_pose(robot_id, STAND_POSE)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            step += 1

            # Print body height every 5 seconds
            if step % (240 * 5) == 0:
                h = get_body_height(robot_id)
                print(f"t={step/240:.0f}s  body_z={h*1000:.1f} mm")

    except p.error:
        print("PyBullet window closed.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()