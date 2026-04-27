#!/usr/bin/env python3
"""
Manually pose the robot, save servo angles as JSON.

Workflow:
  1. Robot moves to a symmetric crouched stance.
  2. For each leg in turn (FL, FR, RL, RR):
     - Torque is DISABLED on that leg's two servos (hip + knee).
     - You physically pose the leg into a "lifted-but-balanced" position.
     - Press Enter.
     - Script reads ALL 8 servo angles and saves them as the lift_<LEG> pose.
     - Torque re-enables, leg returns to stance for the next iteration.
  3. Final JSON written to balance_poses.json.

The saved JSON has one pose per leg ("lift_FL", "lift_FR", ...) plus the
"stance" pose, with raw servo IDs as keys. A companion script can
replay these to verify the manual balance configuration.

Usage:
  python3 pose_capture.py                     # default 'balance_poses.json'
  python3 pose_capture.py --out my_poses.json
  python3 pose_capture.py --hip=35 --knee=-75 # tweak symmetric stance
"""
import json
import sys
import time


# Servo IDs for each leg (hip_id, knee_id)
LEG_SERVOS = {
    "FL": (3, 7), "FR": (4, 8),
    "RL": (1, 5), "RR": (2, 6),
}


def main():
    out_path = "balance_poses.json"
    hip = 35
    knee = -75
    for arg in sys.argv[1:]:
        if arg.startswith("--out="):
            out_path = arg.split("=", 1)[1]
        elif arg.startswith("--hip="):
            hip = int(arg.split("=", 1)[1])
        elif arg.startswith("--knee="):
            knee = int(arg.split("=", 1)[1])

    sys.argv = [sys.argv[0]]   # avoid gait_controller's __main__ block

    import gait_controller as gc
    from gait_controller import leg_abs
    from pylx16a.lx16a import LX16A

    # Disable trim for the symmetric stance (post-CoM-redistribution we
    # want true symmetric pose, not the old front-deeper compensation).
    original_trim = dict(gc.KNEE_TRIM)
    for k in gc.KNEE_TRIM:
        gc.KNEE_TRIM[k] = 0

    def symmetric_stance():
        for leg in ["FL", "FR", "RL", "RR"]:
            leg_abs(leg, hip, knee)
        time.sleep(0.8)

    def read_all_angles():
        """Return {servo_id: angle_degrees} for all 8 servos."""
        out = {}
        for leg, (h, k) in LEG_SERVOS.items():
            out[h] = LX16A(h).get_physical_angle()
            out[k] = LX16A(k).get_physical_angle()
        return out

    def disable_leg_torque(leg):
        h, k = LEG_SERVOS[leg]
        LX16A(h).disable_torque()
        LX16A(k).disable_torque()

    def enable_leg_torque(leg):
        h, k = LEG_SERVOS[leg]
        LX16A(h).enable_torque()
        LX16A(k).enable_torque()

    print(f"\n=== Pose capture ===")
    print(f"  symmetric stance:  hip={hip}°  knee={knee}°")
    print(f"  output file:       {out_path}")
    print()
    print("Settling into symmetric stance...")
    symmetric_stance()
    time.sleep(1.2)

    poses = {"stance": read_all_angles()}
    print(f"  stance angles: {poses['stance']}")

    input("\nReady. Press Enter to start the capture loop...")

    for leg in ["FL", "FR", "RL", "RR"]:
        print(f"\n--- Capturing lift_{leg} ---")
        print(f"  Disabling torque on {leg} (hip {LEG_SERVOS[leg][0]}, knee {LEG_SERVOS[leg][1]})...")
        disable_leg_torque(leg)
        print(f"  Pose the {leg} leg manually now (lift it into a stable position).")
        print(f"  The other 3 legs are still holding their stance.")
        input(f"  When you're happy with the {leg} pose, press Enter...")

        angles = read_all_angles()
        poses[f"lift_{leg}"] = angles
        print(f"  Captured {leg} angles: {angles}")

        print(f"  Re-engaging {leg} torque + returning to stance...")
        enable_leg_torque(leg)
        time.sleep(0.3)
        # Return JUST this leg to stance before doing the next one,
        # so the rest of the body doesn't shift.
        leg_abs(leg, hip, knee)
        time.sleep(1.0)

    # Restore trim
    for k, v in original_trim.items():
        gc.KNEE_TRIM[k] = v

    print(f"\nReturning all legs to stance...")
    symmetric_stance()

    with open(out_path, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"\nSaved {len(poses)} poses to {out_path}.")
    print(f"Pose names: {list(poses.keys())}")


if __name__ == "__main__":
    main()
