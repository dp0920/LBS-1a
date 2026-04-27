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
  python3 pose_capture.py --manual-stance     # disable ALL torque first,
                                               # let you level body manually,
                                               # then capture stance pose.
                                               # Use when commanded
                                               # symmetric pose is tilted
                                               # (per-servo calib drift).
  python3 pose_capture.py --pose-all-legs     # during each lift capture,
                                               # disable torque on ALL 4 legs
                                               # (not just the target). Lets
                                               # you actively shift support
                                               # legs to compensate for CoM
                                               # bias. Required when one
                                               # diagonal can't support a
                                               # static 3-leg tripod.
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
    manual_stance = False
    pose_all_legs = False
    for arg in sys.argv[1:]:
        if arg.startswith("--out="):
            out_path = arg.split("=", 1)[1]
        elif arg.startswith("--hip="):
            hip = int(arg.split("=", 1)[1])
        elif arg.startswith("--knee="):
            knee = int(arg.split("=", 1)[1])
        elif arg == "--manual-stance":
            manual_stance = True
        elif arg == "--pose-all-legs":
            pose_all_legs = True

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
    print(f"  manual stance:     {manual_stance}")
    print(f"  output file:       {out_path}")
    print()
    print("Settling into symmetric stance...")
    symmetric_stance()
    time.sleep(1.2)

    if manual_stance:
        print("\n--- Capturing stance pose manually ---")
        print("  Disabling torque on ALL 8 servos. Body will go floppy —")
        print("  support it, level it physically, then press Enter.")
        for leg in ["FL", "FR", "RL", "RR"]:
            disable_leg_torque(leg)
        input("  Level the body, then press Enter to capture stance...")
        stance_angles = read_all_angles()
        print(f"  Captured stance angles: {stance_angles}")
        # Re-enable torque AT THE CAPTURED ANGLES so the robot holds.
        # pylx16a doesn't have a "hold here" call, so we send move() to
        # the just-read positions before re-enabling torque, then enable.
        # Simpler: enable_torque, then move() to the captured angles.
        for leg in ["FL", "FR", "RL", "RR"]:
            enable_leg_torque(leg)
        time.sleep(0.2)
        # Move servos to the captured angles via direct LX16A.move.
        # This avoids leg_abs's neutral+offset arithmetic.
        for sid, angle in stance_angles.items():
            LX16A(sid).move(angle, time=400)
        time.sleep(0.6)
        poses = {"stance": stance_angles}
    else:
        poses = {"stance": read_all_angles()}
        print(f"  stance angles: {poses['stance']}")

    input("\nReady. Press Enter to start the capture loop...")

    for leg in ["FL", "FR", "RL", "RR"]:
        print(f"\n--- Capturing lift_{leg} ---")
        if pose_all_legs:
            print(f"  Disabling torque on ALL 4 legs (--pose-all-legs).")
            print(f"  You can now reposition every leg — shift the support")
            print(f"  legs to compensate for CoM bias, then lift {leg}.")
            for L in ["FL", "FR", "RL", "RR"]:
                disable_leg_torque(L)
        else:
            print(f"  Disabling torque on {leg} (hip {LEG_SERVOS[leg][0]}, "
                  f"knee {LEG_SERVOS[leg][1]})...")
            disable_leg_torque(leg)
            print(f"  Pose the {leg} leg manually now (lift it into a stable "
                  f"position).")
            print(f"  The other 3 legs are still holding their stance.")
        input(f"  When the robot is balanced with {leg} lifted, press Enter...")

        angles = read_all_angles()
        poses[f"lift_{leg}"] = angles
        print(f"  Captured angles: {angles}")

        print(f"  Re-engaging torque + returning to stance...")
        if pose_all_legs:
            # Re-enable all 4 legs at their captured-lift angles, then
            # move each back to the captured stance.
            for L in ["FL", "FR", "RL", "RR"]:
                enable_leg_torque(L)
            time.sleep(0.3)
            # Move every servo back to the stance configuration.
            for sid, ang in poses["stance"].items():
                LX16A(sid).move(ang, time=600)
        else:
            enable_leg_torque(leg)
            time.sleep(0.3)
            if manual_stance and "stance" in poses:
                h, k = LEG_SERVOS[leg]
                LX16A(h).move(poses["stance"][h], time=600)
                LX16A(k).move(poses["stance"][k], time=600)
            else:
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
