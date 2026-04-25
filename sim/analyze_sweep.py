#!/usr/bin/env python3
"""
Analyze pooled random_sweep results.

Loads one or more random_sweep JSON files, computes:
  - Top N configurations by combined score (mean_dist × survival_rate)
  - Per-coefficient correlation with mean_dist, survival
  - Marginal trend for each coefficient (mean performance binned by coef value)

Usage:
  python analyze_sweep.py random_sweep_results.json random_sweep_results_2.json
  python analyze_sweep.py *.json --top 10
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np


COEFS = ["vel", "ext", "stride", "wt", "gait"]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for r in data:
            # Each entry: tag, out_path, config{vel,ext,stride,wt,gait},
            # mean_dist, max_dist, min_dist, survived, n_episodes
            r["combined"] = r["mean_dist"] * (r["survived"] / r["n_episodes"])
            r["source"] = p
            rows.append(r)
    return rows


def print_top(rows, k):
    ranked = sorted(rows, key=lambda r: -r["combined"])[:k]
    print(f"\n=== Top {k} by combined score (mean_dist × survival_rate) ===")
    print(f"{'#':>3s} {'tag':>10s} {'mean':>6s} {'max':>6s} "
          f"{'surv':>7s} {'comb':>6s}    "
          f"{'vel':>5s} {'ext':>5s} {'stride':>6s} {'wt':>5s} {'gait':>5s}")
    print("-" * 95)
    for i, r in enumerate(ranked):
        c = r["config"]
        print(f"{i+1:>3d} {r['tag']:>10s} {r['mean_dist']:>6.1f} "
              f"{r['max_dist']:>6.1f} {r['survived']:>3d}/{r['n_episodes']:<3d} "
              f"{r['combined']:>6.1f}    "
              f"{c['vel']:>5.2f} {c['ext']:>5.2f} {c['stride']:>6.2f} "
              f"{c['wt']:>5.2f} {c['gait']:>5.2f}")


def correlation_table(rows):
    print("\n=== Per-coefficient correlation with outcome metrics ===")
    print(f"{'coef':>8s}  {'corr(mean)':>10s}  {'corr(surv)':>10s}  "
          f"{'corr(combined)':>14s}")
    print("-" * 50)
    means = np.array([r["mean_dist"] for r in rows])
    survs = np.array([r["survived"] / r["n_episodes"] for r in rows])
    combs = np.array([r["combined"] for r in rows])
    for coef in COEFS:
        vals = np.array([r["config"][coef] for r in rows])
        # Use log-coef since they're log-uniform sampled
        log_vals = np.log10(vals)
        c_mean = np.corrcoef(log_vals, means)[0, 1]
        c_surv = np.corrcoef(log_vals, survs)[0, 1]
        c_comb = np.corrcoef(log_vals, combs)[0, 1]
        print(f"{coef:>8s}  {c_mean:>+10.3f}  {c_surv:>+10.3f}  "
              f"{c_comb:>+14.3f}")


def marginal_trends(rows, n_bins=4):
    """For each coefficient, bin samples and show mean outcome per bin."""
    print(f"\n=== Marginal trends (binning each coef into {n_bins} buckets) ===")
    print("Each row is one coefficient. Within row, buckets show "
          "mean_dist by coef-value range.")
    for coef in COEFS:
        vals = np.array([r["config"][coef] for r in rows])
        means = np.array([r["mean_dist"] for r in rows])
        survs = np.array([r["survived"] / r["n_episodes"] for r in rows])
        combs = np.array([r["combined"] for r in rows])
        # Log-spaced bins
        log_vals = np.log10(vals)
        edges = np.quantile(log_vals, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9  # nudge so the max is included in last bin
        which = np.digitize(log_vals, edges) - 1
        which = np.clip(which, 0, n_bins - 1)

        print(f"\n  {coef}:")
        for b in range(n_bins):
            mask = which == b
            n = int(mask.sum())
            if n == 0:
                continue
            lo = 10 ** edges[b]
            hi = 10 ** edges[b + 1]
            print(f"    [{lo:.2f}, {hi:.2f}]  n={n:2d}  "
                  f"mean_dist={means[mask].mean():5.1f}m  "
                  f"surv={survs[mask].mean()*100:4.0f}%  "
                  f"combined={combs[mask].mean():5.1f}")


def reference_table(rows):
    """Append known reference points for context."""
    refs = [
        {"label": "v20 (3M, stride=5, gait on)",
         "mean_dist": 63.92, "survived": 33, "n_episodes": 50,
         "config": {"vel": 5.0, "ext": 3.0, "stride": 5.0, "wt": 2.0,
                    "gait": 0.25}},
        {"label": "v21 (3M, stride=5, gait off)",
         "mean_dist": 50.39, "survived": 36, "n_episodes": 50,
         "config": {"vel": 5.0, "ext": 3.0, "stride": 5.0, "wt": 2.0,
                    "gait": 0.0}},
        {"label": "v12_3M_fixed (stride=1.5)",
         "mean_dist": 49.26, "survived": 24, "n_episodes": 50,
         "config": {"vel": 5.0, "ext": 3.0, "stride": 1.5, "wt": 2.0,
                    "gait": 0.25}},
    ]
    print("\n=== Reference policies (3M training, 50-episode eval) ===")
    print(f"{'label':>32s}  {'mean':>5s}  {'surv':>6s}  "
          f"{'combined':>8s}")
    print("-" * 60)
    for r in refs:
        comb = r["mean_dist"] * (r["survived"] / r["n_episodes"])
        print(f"{r['label']:>32s}  {r['mean_dist']:>5.1f}  "
              f"{r['survived']:>2d}/{r['n_episodes']:<2d}  {comb:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="random_sweep JSON files")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rows = load(args.paths)
    print(f"Loaded {len(rows)} configurations from {len(args.paths)} files.")
    if len(rows) == 0:
        print("No data."); sys.exit(0)

    print_top(rows, args.top)
    reference_table(rows)
    correlation_table(rows)
    marginal_trends(rows, n_bins=min(4, max(2, len(rows) // 10)))

    print(f"\nTotal samples analyzed: {len(rows)}")


if __name__ == "__main__":
    main()
