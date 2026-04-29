#!/usr/bin/env python3
"""
Step-response calibration for LX-16A servos.

For each of the 8 servos:
  1. Read current position.
  2. Command a +20° step (or whatever --step is set to) at the servo's
     fastest internal slew (move(target, time=0)).
  3. Sample physical angle every ~20 ms for --duration ms.
  4. Return servo to its original position.

Output: motor_step_response.json — list of per-servo dicts with the
raw time-series. Run on the Mac:
    python3 fit_pd_from_step_response.py motor_step_response.json
to get the recommended MuJoCo kp/kv.

Usage:
  python3 motor_step_response.py
  python3 motor_step_response.py --step 25 --duration 400
  python3 motor_step_response.py --step -20    # downward step
  python3 motor_step_response.py --servos 5,7  # only test FL/RL knees

Notes:
- Run with the robot in a stable starting pose (e.g. X-config from
  stance.py). Each step nudges one joint, briefly disturbs balance,
  then returns to start. Don't run on a bare desk if the robot might
  fall — pose it on the floor or in a jig.
- Reading and commanding LX-16A over the half-duplex bus introduces
  ~5-10 ms of round-trip latency per call; sampling rate is limited
  to ~50 Hz.
"""
import argparse
import json
import sys
import time

# Allow running from anywhere — gait_controller import sets up the bus
sys.argv_backup = sys.argv[:]
sys.argv = [sys.argv[0]]   # avoid gait_controller's __main__ block
import gait_controller    # noqa: F401  — required for bus init
from pylx16a.lx16a import LX16A
sys.argv = sys.argv_backup

ALL_SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8]


# Per-servo soft limits — relative to current position. Mechanical
# safety: don't command beyond ±SAFE_TRAVEL_DEG of current.
SAFE_TRAVEL_DEG = 35.0


def _capture(servo, t0, duration_ms, sample_dt_ms):
    """Sample servo angle from t0 until duration_ms."""
    samples_t, samples_a = [], []
    next_t = 0.0
    while True:
        now_ms = (time.perf_counter() - t0) * 1000.0
        if now_ms > duration_ms:
            break
        if now_ms >= next_t:
            samples_t.append(now_ms)
            samples_a.append(servo.get_physical_angle())
            next_t += sample_dt_ms
    return samples_t, samples_a


def step_response(servo_id, step_deg, duration_ms, sample_dt_ms):
    """Run a step-response test: step UP by step_deg, then DOWN back.

    Always relative to the servo's CURRENT angle. Clamped so the step
    never moves more than SAFE_TRAVEL_DEG from the start position.
    """
    s = LX16A(servo_id)
    time.sleep(0.3)
    start = s.get_physical_angle()

    # Cap requested step so we stay inside SAFE_TRAVEL_DEG.
    step_deg = float(max(-SAFE_TRAVEL_DEG, min(SAFE_TRAVEL_DEG, step_deg)))
    target_up = start + step_deg
    # Hard physical limits as a final guard.
    target_up = max(5.0, min(235.0, target_up))
    actual_step = target_up - start
    if abs(actual_step) < 1.0:
        return {"servo_id": servo_id, "error": "near limit, step too small"}

    # ----- Forward step (start → start + step) -----
    t0 = time.perf_counter()
    s.move(target_up, time=0)
    t_up, a_up = _capture(s, t0, duration_ms, sample_dt_ms)

    # Settle at target before reverse step
    time.sleep(0.3)

    # ----- Reverse step (start + step → start) -----
    t0 = time.perf_counter()
    s.move(start, time=0)
    t_dn, a_dn = _capture(s, t0, duration_ms, sample_dt_ms)

    time.sleep(0.4)

    return {
        "servo_id": servo_id,
        "start_deg": start,
        "target_deg": target_up,
        "commanded_step_deg": actual_step,
        "duration_ms": duration_ms,
        "sample_dt_ms": sample_dt_ms,
        "up": {"t_ms": t_up, "angle_deg": a_up},
        "down": {"t_ms": t_dn, "angle_deg": a_dn},
    }


def quick_summary(d):
    """Print rise stats for both step directions."""
    out = {}
    for direction, key in (("up", "up"), ("down", "down")):
        sub = d.get(key)
        if not sub or not sub["t_ms"]:
            out[direction] = None
            continue
        a0 = d["start_deg"] if direction == "up" else d["target_deg"]
        a_arr = sub["angle_deg"]
        t_arr = sub["t_ms"]
        target_swing = (d["target_deg"] - d["start_deg"]
                        if direction == "up" else
                        d["start_deg"] - d["target_deg"])
        t50 = t90 = None
        for t, a in zip(t_arr, a_arr):
            prog = (a - a0) / target_swing if target_swing else 0
            if t50 is None and prog >= 0.5:
                t50 = t
            if t90 is None and prog >= 0.9:
                t90 = t
                break
        out[direction] = {"t50_ms": t50, "t90_ms": t90,
                          "n_samples": len(t_arr)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=10.0,
                    help="Step magnitude in degrees (positive or negative). "
                         "Default 10°. Capped to ±35° from current "
                         "position regardless.")
    ap.add_argument("--duration", type=int, default=300,
                    help="Capture window in ms. Default 300.")
    ap.add_argument("--sample-dt", type=int, default=20,
                    help="Sampling interval in ms. Default 20 (~50 Hz). "
                         "Lower won't help — bus latency caps real rate.")
    ap.add_argument("--servos", type=str, default=None,
                    help="Comma-separated list of servo IDs to test "
                         "(default: all 8).")
    ap.add_argument("--out", default="motor_step_response.json")
    args = ap.parse_args()

    sid_list = ([int(s) for s in args.servos.split(",")]
                if args.servos else ALL_SERVO_IDS)

    print(f"\n=== LX-16A step-response calibration ===")
    print(f"  step: {args.step:+.1f}°   duration: {args.duration} ms   "
          f"sample_dt: {args.sample_dt} ms")
    print(f"  servos: {sid_list}")

    results = []
    for sid in sid_list:
        print(f"\nServo {sid}:")
        try:
            data = step_response(sid, args.step,
                                 args.duration, args.sample_dt)
            if "error" in data:
                print(f"  SKIPPED: {data['error']}")
                results.append(data)
                continue
            results.append(data)
            print(f"  start={data['start_deg']:.1f}°  "
                  f"target={data['target_deg']:.1f}°  "
                  f"step={data['commanded_step_deg']:+.1f}°")
            sm = quick_summary(data)
            for direction in ("up", "down"):
                s = sm.get(direction)
                if s:
                    print(f"  {direction:>4s}: t50={s['t50_ms']} ms  "
                          f"t90={s['t90_ms']} ms  n={s['n_samples']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"servo_id": sid, "error": str(e)})

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} servo traces to {args.out}")
    print(f"\nNext step on the Mac:")
    print(f"  scp -O admin@<pi>:~/LBS-1a/robot/{args.out} sim/")
    print(f"  python3 sim/fit_pd_from_step_response.py {args.out}")


if __name__ == "__main__":
    main()
