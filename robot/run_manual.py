#!/usr/bin/env python3
"""
Run the hand-tuned weight-transfer gait (full_stride) on hardware.

This is the simplest deployable controller for the Optimus Primal — bypasses
the JSON gait playback and the keyboard-input mode. Just imports
`crawl_stance` and `full_stride` from gait_controller and calls them in
sequence. Useful as a sanity-check that the robot is mechanically and
electrically working before trying the CMA / RL gaits.

Usage:
  python3 run_manual.py              # default: 3 strides
  python3 run_manual.py --n=5        # 5 strides
  python3 run_manual.py --no-pause   # skip the "place tape, press Enter" prompt

The full_stride routine, defined in gait_controller.py, performs the
explicit weight-transfer pattern that real quadrupeds use:
  1. Raise the two adjacent legs + drop the diagonal leg's height
     -> shifts weight off the target leg.
  2. Lift, swing forward, plant the target leg.
  3. Mirror for the opposite diagonal pair.
This is friction-tolerant (no slip-push to propel forward), which is
why hand gaits often outperform sim-optimized gaits on real hardware.
"""
import sys
import time


def main():
    cycles = 3
    pause = True
    for arg in sys.argv[1:]:
        if arg.startswith("--n="):
            cycles = int(arg.split("=", 1)[1])
        elif arg == "--no-pause":
            pause = False

    # Strip our argv before importing gait_controller so its __main__
    # block doesn't run (it auto-runs on import unfortunately).
    sys.argv = [sys.argv[0]]

    from gait_controller import crawl_stance, full_stride

    print(f"Settling into crawl_stance...")
    crawl_stance()
    time.sleep(1.0)

    if pause:
        input("Robot in start pose. Place tape, then press Enter to walk...")

    for i in range(cycles):
        print(f"\n=== Stride {i + 1}/{cycles} ===")
        full_stride()
        time.sleep(0.3)

    print("\nDone. Returning to crawl_stance.")
    crawl_stance()


if __name__ == "__main__":
    main()
