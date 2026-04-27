#!/usr/bin/env python3
"""
Balance test: symmetric stance + sequential leg lifts.

After the CoM redistribution (Pi + buck repositioning), the robot should
be roughly centered. This script demonstrates that by:
  1. Putting all 4 legs at the SAME hip + knee offset (symmetric stance)
  2. Lifting each leg in turn (FL → FR → RL → RR), holding for inspection,
     then planting it back
  3. Returning to symmetric stance at the end

If the robot stays upright through all 4 lifts, the CoM is well-centered
within the support tripod for each lift. If it tips during one specific
leg's lift, the CoM is biased toward the OPPOSITE corner from that leg
(e.g., tipping during FL lift = CoM is biased to FL).

Usage:
  python3 balance_test.py                 # default: 30° lifts, 1.5s hold
  python3 balance_test.py --lift=45       # bigger lifts (more demanding)
  python3 balance_test.py --hold=3.0      # 3s hold per leg (longer to inspect)
  python3 balance_test.py --hip=20 --knee=20    # tweak symmetric stance offsets
"""
import sys
import time


def main():
    lift_amount = 30
    hold_time = 1.5
    hip_offset = 20
    knee_offset = 20

    for arg in sys.argv[1:]:
        if arg.startswith("--lift="):
            lift_amount = int(arg.split("=", 1)[1])
        elif arg.startswith("--hold="):
            hold_time = float(arg.split("=", 1)[1])
        elif arg.startswith("--hip="):
            hip_offset = int(arg.split("=", 1)[1])
        elif arg.startswith("--knee="):
            knee_offset = int(arg.split("=", 1)[1])

    # stance.py auto-runs apply_stance() on import. Keep argv clean so
    # we don't accidentally trip any flag-parsing it might add later.
    sys.argv = [sys.argv[0]]

    import stance

    # Override the standing offsets and re-apply for a clean symmetric pose.
    # Using module attributes so apply_stance() picks them up.
    stance.hip_offset = hip_offset
    stance.knee_offset = knee_offset

    print(f"\n=== Balance test ===")
    print(f"  symmetric stance:  hip={hip_offset}°  knee={knee_offset}°")
    print(f"  lift amount:       {lift_amount}°")
    print(f"  hold per leg:      {hold_time}s")
    print()
    print("Settling into symmetric stance...")
    stance.apply_stance(duration=800)
    time.sleep(1.0)

    input("Robot in symmetric stance. Press Enter to start leg-lift sequence...")

    for leg in ["FL", "FR", "RL", "RR"]:
        print(f"\n  Lifting {leg}...")
        stance.lift_leg(leg, lift_amount=lift_amount)
        time.sleep(hold_time)
        print(f"  Planting {leg}...")
        stance.plant_leg(leg)
        time.sleep(0.6)

    print("\nReturning to symmetric stance.")
    stance.apply_stance(duration=800)
    print("Done.")


if __name__ == "__main__":
    main()
