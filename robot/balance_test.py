#!/usr/bin/env python3
"""
Balance test from a SYMMETRIC crawl stance: sequential leg lifts.

The default crawl_stance in gait_controller.py is intentionally
asymmetric (front knees deep crouch -100°, rear knees shallow -50°)
to compensate for an older rear-heavy CoM. Now that the CoM has been
redistributed via the recent build changes, this test uses a SYMMETRIC
crouch — same knee depth on all 4 legs, hip=35° on all 4 — so the
robot's body is level (not tilted forward) at rest. Lifting each leg
then measures whether the CoM is genuinely centered.

If the robot stays level through all 4 lifts, balance is confirmed:
no further front/rear bias compensation is needed. If it tips during
a specific leg's lift, the CoM is biased toward the OPPOSITE corner.

Usage:
  python3 balance_test.py                  # symmetric crouch knee=-75, NO trim
  python3 balance_test.py --use-trim       # apply gait_controller's KNEE_TRIM
                                            # (compensates for OLD rear-heavy CoM)
  python3 balance_test.py --knee=-80       # deeper symmetric crouch
  python3 balance_test.py --knee=-60       # shallower (more upright)
  python3 balance_test.py --hip=30         # tighter hip stance
  python3 balance_test.py --lift=45        # bigger lift (more demanding)
  python3 balance_test.py --hold=3.0       # 3s hold per leg
  python3 balance_test.py --no-pause       # skip the Enter prompt

KNEE_TRIM context:
  gait_controller.KNEE_TRIM = {"FL": -20, "FR": -20, "RL": 0, "RR": 0}
  was tuned for the OLD rear-heavy CoM, biasing front knees deeper to
  keep the body level. After the recent CoM redistribution, this trim
  causes a forward tilt — so this script defaults to NOT applying it.
  Pass --use-trim to fall back to the original behavior.
"""
import sys
import time


def main():
    hip = 35
    knee = -75       # symmetric depth — between front -100 and rear -50
    lift_amount = 35
    hold_time = 1.5
    pause = True
    use_trim = False  # default off; old trim compensates for old rear-heavy CoM
    for arg in sys.argv[1:]:
        if arg.startswith("--hip="):
            hip = int(arg.split("=", 1)[1])
        elif arg.startswith("--knee="):
            knee = int(arg.split("=", 1)[1])
        elif arg.startswith("--lift="):
            lift_amount = int(arg.split("=", 1)[1])
        elif arg.startswith("--hold="):
            hold_time = float(arg.split("=", 1)[1])
        elif arg == "--no-pause":
            pause = False
        elif arg == "--use-trim":
            use_trim = True
        elif arg == "--no-trim":
            use_trim = False

    # Strip argv before importing so gait_controller's __main__ block
    # doesn't trigger.
    sys.argv = [sys.argv[0]]

    import gait_controller as gc
    from gait_controller import leg_abs

    # KNEE_TRIM is a module-level dict applied inside leg_abs(). To skip
    # the trim non-destructively, save the original and zero it out for
    # the duration of this test, then restore at the end.
    original_trim = dict(gc.KNEE_TRIM)
    if not use_trim:
        for k in gc.KNEE_TRIM:
            gc.KNEE_TRIM[k] = 0
        trim_status = "DISABLED (zeroed for this test)"
    else:
        trim_status = f"applied: {original_trim}"

    def symmetric_stance():
        """All 4 legs at the same hip + knee — body sits level."""
        for leg in ["FL", "FR", "RL", "RR"]:
            leg_abs(leg, hip, knee)
        time.sleep(0.8)

    print(f"\n=== Balance test from SYMMETRIC stance ===")
    print(f"  stance:         hip={hip}°  knee={knee}° (all 4 legs)")
    print(f"  KNEE_TRIM:      {trim_status}")
    print(f"  lift amount:    +{lift_amount}° (knee straighter)")
    print(f"  hold per leg:   {hold_time}s")
    print()
    print("Settling into symmetric stance...")
    symmetric_stance()
    time.sleep(1.0)

    if pause:
        input("Robot in symmetric stance — should be level. "
              "Press Enter to start leg-lift sequence...")

    for leg in ["FL", "FR", "RL", "RR"]:
        target_knee = knee + lift_amount   # knee straighter -> foot lifts
        print(f"\n  Lifting {leg} (knee {knee}° -> {target_knee}°)...")
        leg_abs(leg, hip, target_knee)
        time.sleep(hold_time)
        print(f"  Planting {leg}...")
        leg_abs(leg, hip, knee)
        time.sleep(0.6)

    print("\nReturning to symmetric stance.")
    symmetric_stance()

    # Restore original trim so any subsequent call into gait_controller
    # (e.g. running run_manual.py in the same Python session) sees the
    # canonical values.
    for k, v in original_trim.items():
        gc.KNEE_TRIM[k] = v
    print("Done.")


if __name__ == "__main__":
    main()
