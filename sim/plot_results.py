#!/usr/bin/env python3
"""
Generate convergence comparison plots from training logs.

Usage:
  python plot_results.py                     # plot everything in results/
  python plot_results.py --by-seed           # one plot per seed (algos compared)
  python plot_results.py --by-algo           # one plot per algo (seeds compared)
"""
import argparse
import json
import os
import glob
import matplotlib.pyplot as plt


def load_log(path):
    """Load a tune_gait*.jsonl file."""
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def find_logs(results_dir="results"):
    """Find all log files and parse their algo/seed from directory name."""
    logs = {}
    for jsonl in sorted(glob.glob(f"{results_dir}/*/tune_gait*.jsonl")):
        dirname = os.path.basename(os.path.dirname(jsonl))
        # dirname is like "cma_gait", "random_stand", "de_random"
        parts = dirname.split("_", 1)
        if len(parts) == 2:
            algo, init = parts
        else:
            algo, init = dirname, "unknown"
        logs[dirname] = {"path": jsonl, "algo": algo, "init": init}
    return logs


def plot_all(logs, output="convergence_all.png"):
    """Single plot with all runs overlaid."""
    plt.figure(figsize=(12, 6))
    for name, info in sorted(logs.items()):
        entries = load_log(info["path"])
        gens = [e["gen"] for e in entries]
        bests = [e["best"] for e in entries]
        plt.plot(gens, bests, label=name)
    plt.xlabel("Generation")
    plt.ylabel("Best Reward")
    plt.title("Gait Optimization — All Runs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def plot_by_group(logs, group_key, compare_key, output_prefix):
    """One plot per group_key value, comparing across compare_key."""
    groups = {}
    for name, info in logs.items():
        g = info[group_key]
        if g not in groups:
            groups[g] = []
        groups[g].append((name, info))

    for group_name, items in sorted(groups.items()):
        plt.figure(figsize=(10, 5))
        for name, info in sorted(items):
            entries = load_log(info["path"])
            gens = [e["gen"] for e in entries]
            bests = [e["best"] for e in entries]
            label = info[compare_key]
            plt.plot(gens, bests, label=label)
        plt.xlabel("Generation")
        plt.ylabel("Best Reward")
        plt.title(f"Convergence — {group_key}={group_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = f"{output_prefix}_{group_name}.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-seed", action="store_true",
                    help="One plot per seed, comparing algorithms")
    ap.add_argument("--by-algo", action="store_true",
                    help="One plot per algorithm, comparing seeds")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    logs = find_logs(args.results_dir)
    if not logs:
        print(f"No logs found in {args.results_dir}/")
        return

    print(f"Found {len(logs)} runs: {', '.join(sorted(logs.keys()))}")

    # Always generate the combined plot
    plot_all(logs)

    if args.by_seed:
        # Group by init seed, compare algorithms
        plot_by_group(logs, "init", "algo", "convergence_seed")

    if args.by_algo:
        # Group by algorithm, compare seeds
        plot_by_group(logs, "algo", "init", "convergence_algo")

    if not args.by_seed and not args.by_algo:
        # Default: generate both
        plot_by_group(logs, "init", "algo", "convergence_seed")
        plot_by_group(logs, "algo", "init", "convergence_algo")


if __name__ == "__main__":
    main()
