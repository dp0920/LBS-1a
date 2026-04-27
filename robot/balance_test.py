#!/usr/bin/env python3
"""
Balance test from crawl_stance: sequential leg lifts.

Uses the actual crawl_stance pose (front legs deep crouch, rear legs
shallow) — same starting pose the gait routines use. Then lifts each
leg in place (no hip swing) so the only thing that changes is the
weight distribution onto the remaining 3-leg support tripod.

If the robot stays upright through all 4 lifts, the CoM is well inside
every support tripod — confirms balance. If it tips during a specific
leg's lift, the CoM is biased toward the OPPOSITE corner from that leg
(e.g., tipping during FL lift => CoM biased to FL).

Usage:
  python3 balance_test.py                 # default: 35° lift, 1.5s hold
  python3 balance_test.py --lift=45       # bigger lift (more demanding)
  python3 balance_test.py --hold=3.0      # 3s hold per leg
  python3 balance_test.py --no-pause      # skip the Enter prompt
"""
import sys
import time


# crawl_stance knee offsets per leg (from gait_controller.crawl_stance):
STANCE_KNEE = {
    "FL": -100,  "FR": -100,   # front legs deep crouch
    "RL":  -50,  "RR":  -50,   # rear legs shallow
}


def main():
    lift_amount = 35
    hold_time = 1.5
    pause = True
    for arg in sys.argv[1:]:
        if arg.startswith("--lift="):
            lift_amount = int(arg.split("=", 1)[1])
        elif arg.startswith("--hold="):
            hold_time = float(arg.split("=", 1)[1])
        elif arg == "--no-pause":
            pause = False

    # Strip argv before importing so gait_controller's __main__ block
    # doesn't trigger.
    sys.argv = [sys.argv[0]]

    from gait_controller import crawl_stance, leg_abs

    print(f"\n=== Balance test from crawl_stance ===")
    print(f"  lift amount:    +{lift_amount}° (knee straighter)")
    print(f"  hold per leg:   {hold_time}s")
    print()
    print("Settling into crawl_stance...")
    crawl_stance()
    time.sleep(1.0)

    if pause:
        input("Robot in crawl_stance. Press Enter to start leg-lift sequence...")

    for leg in ["FL", "FR", "RL", "RR"]:
        knee = STANCE_KNEE[leg]
        print(f"\n  Lifting {leg} (stance knee={knee}°, lift to {knee + lift_amount}°)...")
        leg_abs(leg, 35, knee + lift_amount)   # lift in place (knee straighter)
        time.sleep(hold_time)
        print(f"  Planting {leg}...")
        leg_abs(leg, 35, knee)                  # back to stance
        time.sleep(0.6)

    print("\nReturning to crawl_stance.")
    crawl_stance()
    print("Done.")


if __name__ == "__main__":
    main()
