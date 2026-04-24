#!/usr/bin/env python3
"""
Rank trained gaits by rollout distance (or speed) in simulation.

Runs a full MuJoCo rollout for every best_gait*.json found under a directory
and prints them sorted. Distance is not stored in the JSON — it's measured
here by replaying the gait.

Usage:
  python rank_gaits.py                          # rank everything under results/
  python rank_gaits.py results/20260422_005700  # rank one run
  python rank_gaits.py --sort speed --top 10    # top 10 by m/s
  python rank_gaits.py --cycles 15              # longer rollouts
"""
import argparse
import glob
import json
import os
import numpy as np

from mujoco_gait import build_model, decode_params
from record_rollout import record_rollout


def rollout_distance(model, path, cycles):
    d = json.load(open(path))
    poses, pt = decode_params(np.array(d["params"]))
    cfg = d.get("config", {})
    interp = cfg.get("interp", "linear")
    rec = record_rollout(model, poses, pt, n_cycles=cycles, interp=interp)
    dist = float(rec["x"][-1]) if len(rec["x"]) else 0.0
    t = float(rec["time"][-1]) if len(rec["time"]) else 0.0
    speed = dist / t if t > 0 else 0.0
    reward = float(d.get("reward", 0.0))
    tilt = cfg.get("fall_tilt_deg", float("nan"))
    sigma = cfg.get("sigma_init", float("nan"))
    return dist, t, speed, reward, interp, tilt, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results",
                    help="Directory to search (recursively) for best_gait*.json")
    ap.add_argument("--cycles", type=int, default=10,
                    help="Rollout cycles per gait (default: 10)")
    ap.add_argument("--sort", choices=["distance", "speed", "reward"],
                    default="distance", help="Rank by this metric")
    ap.add_argument("--top", type=int, default=10,
                    help="Show top N (default: 10; use 0 for all)")
    args = ap.parse_args()

    pattern = os.path.join(args.path, "**", "best_gait*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"No best_gait*.json found under {args.path}")
        return

    print(f"Rolling out {len(files)} gaits ({args.cycles} cycles each)...\n")
    hdr = (f"{'dist(m)':>8} {'time(s)':>8} {'m/s':>7} {'reward':>9} "
           f"{'interp':>12} {'tilt':>6} {'sigma':>6}  file")
    print(hdr)
    print("-" * len(hdr))

    model = build_model()
    rows = []
    for f in files:
        dist, t, speed, reward, interp, tilt, sigma = rollout_distance(
            model, f, args.cycles)
        rows.append((dist, t, speed, reward, interp, tilt, sigma, f))
        print(f"{dist:8.2f} {t:8.1f} {speed:7.3f} {reward:+9.2f} "
              f"{interp:>12} {tilt:6.2f} {sigma:6.2f}  {f}")

    sort_idx = {"distance": 0, "speed": 2, "reward": 3}[args.sort]
    rows.sort(key=lambda r: -r[sort_idx])

    n = args.top if args.top > 0 else len(rows)
    print(f"\n=== Top {min(n, len(rows))} by {args.sort} ===")
    print(f"{'rank':>4} " + hdr)
    for i, (dist, t, speed, reward, interp, tilt, sigma, f) in enumerate(rows[:n], 1):
        print(f"{i:4d} {dist:8.2f} {t:8.1f} {speed:7.3f} {reward:+9.2f} "
              f"{interp:>12} {tilt:6.2f} {sigma:6.2f}  {f}")


if __name__ == "__main__":
    main()
