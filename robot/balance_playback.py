#!/usr/bin/env python3
"""
Playback of captured balance poses (from pose_capture.py).

Loads balance_poses.json and runs through the captured stance + per-leg
lift poses, demonstrating that the robot can stand and lift each leg
without tipping. Pure replay — no inverse kinematics, no calibration
math: the script directly commands the servo angles you physically
verified during capture.

Usage:
  python3 balance_playback.py                        # default 'balance_poses.json'
  python3 balance_playback.py --in=my_poses.json
  python3 balance_playback.py --hold=2.5             # 2.5s hold per pose
  python3 balance_playback.py --transition=800       # 800 ms move time
  python3 balance_playback.py --legs=FL,FR,RL,RR     # subset / reorder
  python3 balance_playback.py --loops=3              # repeat the full sequence
  python3 balance_playback.py --sequence=stance,lean_fwd,lift_RL,lean_fwd,stance
                                                      # arbitrary pose order;
                                                      # bypasses default
                                                      # stance->lift->stance
                                                      # cycle. Useful for
                                                      # routing through
                                                      # intermediate CoM-shift
                                                      # poses (e.g. lean_fwd
                                                      # before a rear lift).
"""
import json
import sys
import time


def main():
    in_path = "balance_poses.json"
    hold_time = 2.0
    transition_ms = 600
    legs = ["FL", "FR", "RL", "RR"]
    loops = 1
    sequence = None

    for arg in sys.argv[1:]:
        if arg.startswith("--in="):
            in_path = arg.split("=", 1)[1]
        elif arg.startswith("--hold="):
            hold_time = float(arg.split("=", 1)[1])
        elif arg.startswith("--transition="):
            transition_ms = int(arg.split("=", 1)[1])
        elif arg.startswith("--legs="):
            legs = [s.strip().upper()
                    for s in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("--loops="):
            loops = int(arg.split("=", 1)[1])
        elif arg.startswith("--sequence="):
            sequence = [s.strip()
                        for s in arg.split("=", 1)[1].split(",")]

    sys.argv = [sys.argv[0]]   # avoid gait_controller's __main__ block

    from pylx16a.lx16a import LX16A
    import gait_controller   # auto-initializes the LX-16A bus

    with open(in_path) as f:
        raw = json.load(f)
    # Normalize servo IDs (string in JSON) to int for indexing.
    poses = {name: {int(sid): ang for sid, ang in angles.items()}
             for name, angles in raw.items()}

    print(f"\n=== Balance playback ===")
    print(f"  loaded:        {in_path}")
    print(f"  poses:         {list(poses.keys())}")
    print(f"  hold per pose: {hold_time}s")
    print(f"  transition:    {transition_ms}ms")
    print(f"  loops:         {loops}")
    if sequence is not None:
        print(f"  sequence:      {sequence}")
    else:
        print(f"  legs:          {legs}")
    print()

    if "stance" not in poses:
        print(f"ERROR: 'stance' pose not found in {in_path}")
        sys.exit(1)

    def move_to_pose(name):
        if name not in poses:
            print(f"  WARN: '{name}' not in JSON, skipping")
            return False
        for sid, angle in poses[name].items():
            LX16A(sid).move(angle, time=transition_ms)
        return True

    print("Moving to stance...")
    move_to_pose("stance")
    time.sleep(transition_ms / 1000.0 + 0.3)

    input("Robot in stance. Press Enter to start the playback...")

    for loop_i in range(loops):
        if loops > 1:
            print(f"\n=== Loop {loop_i + 1}/{loops} ===")
        if sequence is not None:
            for pose_name in sequence:
                print(f"\n  Moving to '{pose_name}'...")
                if not move_to_pose(pose_name):
                    continue
                time.sleep(transition_ms / 1000.0 + 0.2)
                print(f"  Holding for {hold_time}s...")
                time.sleep(hold_time)
        else:
            for leg in legs:
                print(f"\n  Lifting {leg} (pose 'lift_{leg}')...")
                if not move_to_pose(f"lift_{leg}"):
                    continue
                time.sleep(transition_ms / 1000.0 + 0.2)
                print(f"  Holding for {hold_time}s...")
                time.sleep(hold_time)
                print(f"  Returning to stance...")
                move_to_pose("stance")
                time.sleep(transition_ms / 1000.0 + 0.3)

    print("\nReturning to stance.")
    move_to_pose("stance")
    time.sleep(transition_ms / 1000.0 + 0.3)
    print("Done.")


if __name__ == "__main__":
    main()
