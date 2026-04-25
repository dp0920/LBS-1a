#!/usr/bin/env python3
"""
Random-uniform coefficient sweep over the main RL reward bonuses.

Each config samples log-uniformly:
  velocity_bonus       ∈ [0.1, 10]
  extension_bonus      ∈ [0.1, 10]
  stride_bonus         ∈ [0.1, 10]
  weight_transfer_bonus ∈ [0.1, 10]
  gait_reward_scale    ∈ [0.01, 1]   (smaller range — gait score is already ±4)

Trains each config for 1 M timesteps, runs 30-episode stats, ranks all
configs by combined score (mean walking distance × survival rate). The
top 3 should be re-evaluated at 50 episodes for tighter error bars.

Usage:
  python random_sweep.py                     # default 12 configs, seed 42
  python random_sweep.py --n-configs 20      # bigger sweep
  python random_sweep.py --timesteps 500000  # quicker (less converged)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np


# Fall-tilt and ctrl-repeat to use for stats (must match training defaults).
EVAL_FALL_TILT = 30.0
EVAL_CTRL_REPEAT = 8


def parse_stats(stats_stdout):
    """Pull (mean_dist, max_dist, min_dist, survived) out of ppo_stats output."""
    mean_d = max_d = min_d = surv = None
    for line in stats_stdout.splitlines():
        m = re.search(r"dist\s+min/mean/max:\s+([\+\-\d.]+)\s*/\s*"
                      r"([\+\-\d.]+)\s*/\s*([\+\-\d.]+)", line)
        if m:
            min_d, mean_d, max_d = float(m.group(1)), float(m.group(2)), float(m.group(3))
        m = re.search(r"survived full episode:\s*(\d+)/\d+", line)
        if m:
            surv = int(m.group(1))
    return mean_d, max_d, min_d, surv


def train_and_eval(idx, n_configs, cfg, timesteps, eval_episodes, out_prefix):
    """Train one config, run stats, return result dict."""
    tag = f"rand_{idx:02d}"
    out_path = f"{out_prefix}_{tag}.zip"
    print(f"\n=== {idx + 1}/{n_configs}  {tag}  ===")
    print(f"    vel={cfg['vel']:6.3f}  ext={cfg['ext']:6.3f}  "
          f"stride={cfg['stride']:6.3f}  wt={cfg['wt']:6.3f}  "
          f"gait={cfg['gait']:6.3f}", flush=True)
    t0 = time.time()
    train_cmd = [
        sys.executable, "train_ppo.py",
        "--timesteps", str(timesteps),
        "--n-envs", "4",
        "--out", out_path,
        "--tb-name", tag,
        "--velocity-shape", "trig",
        "--velocity-bonus", f"{cfg['vel']:.4f}",
        "--lr-schedule", "constant",
        "--ent-coef", "0.0",
        "--gait-reward-scale", f"{cfg['gait']:.4f}",
        "--stride-bonus", f"{cfg['stride']:.4f}",
        "--weight-transfer-bonus", f"{cfg['wt']:.4f}",
    ]
    proc = subprocess.run(train_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    train failed: {proc.stderr[-500:]}")
        return None
    stats_cmd = [
        sys.executable, "ppo_stats.py",
        "--policy", out_path,
        "--episodes", str(eval_episodes),
        "--fall-tilt", str(EVAL_FALL_TILT),
        "--ctrl-repeat", str(EVAL_CTRL_REPEAT),
    ]
    proc = subprocess.run(stats_cmd, capture_output=True, text=True)
    mean_d, max_d, min_d, surv = parse_stats(proc.stdout)
    elapsed = time.time() - t0
    if mean_d is None or surv is None:
        print(f"    stats parse failed for {tag}")
        return None
    print(f"    -> mean={mean_d:5.1f}m  max={max_d:5.1f}m  "
          f"surv={surv}/{eval_episodes}  ({elapsed:.0f}s)", flush=True)
    return {
        "tag": tag,
        "out_path": out_path,
        "config": cfg,
        "mean_dist": mean_d,
        "max_dist": max_d,
        "min_dist": min_d,
        "survived": surv,
        "n_episodes": eval_episodes,
        "elapsed_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-configs", type=int, default=12)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="ppo_rand")
    ap.add_argument("--results-json", default="random_sweep_results.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Random-uniform coefficient sweep ({args.n_configs} configs, "
          f"{args.timesteps:,} timesteps each)")
    print(f"Sampling log-uniform: vel/ext/stride/wt ∈ [0.1, 10], "
          f"gait ∈ [0.01, 1]")
    print(f"Seed: {args.seed}\n")

    results = []
    for i in range(args.n_configs):
        cfg = {
            "vel":    float(10 ** rng.uniform(-1, 1)),
            "ext":    float(10 ** rng.uniform(-1, 1)),
            "stride": float(10 ** rng.uniform(-1, 1)),
            "wt":     float(10 ** rng.uniform(-1, 1)),
            "gait":   float(10 ** rng.uniform(-2, 0)),
        }
        res = train_and_eval(i, args.n_configs, cfg,
                             args.timesteps, args.eval_episodes,
                             args.out_prefix)
        if res is not None:
            results.append(res)
            # Save partial results after each run so a crash doesn't lose them
            with open(args.results_json, "w") as f:
                json.dump(results, f, indent=2)

    # Final ranking
    if not results:
        print("No successful runs.")
        return
    for r in results:
        r["combined"] = r["mean_dist"] * (r["survived"] / r["n_episodes"])
    ranked = sorted(results, key=lambda r: -r["combined"])

    print("\n\n=== Ranked by combined score (mean_dist × survival_rate) ===")
    print(f"{'#':>3s} {'tag':>9s} {'mean':>6s} {'max':>6s} "
          f"{'surv':>6s} {'comb':>6s}    "
          f"{'vel':>5s} {'ext':>5s} {'stride':>6s} {'wt':>5s} {'gait':>5s}")
    print("-" * 95)
    for i, r in enumerate(ranked):
        c = r["config"]
        print(f"{i+1:>3d} {r['tag']:>9s} {r['mean_dist']:>6.1f} "
              f"{r['max_dist']:>6.1f} {r['survived']:>3d}/{r['n_episodes']:<2d} "
              f"{r['combined']:>6.1f}    "
              f"{c['vel']:>5.2f} {c['ext']:>5.2f} {c['stride']:>6.2f} "
              f"{c['wt']:>5.2f} {c['gait']:>5.2f}")

    print(f"\nSaved {args.results_json}.")
    print(f"Top 3 candidates worth re-running at 3M / 50-episode evaluation:")
    for r in ranked[:3]:
        print(f"  - {r['tag']}: {r['out_path']}")


if __name__ == "__main__":
    main()
