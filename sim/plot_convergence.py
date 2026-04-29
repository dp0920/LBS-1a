#!/usr/bin/env python3
"""
Generate convergence plots from CMA tune_gait*.jsonl logs.

Saves PNGs into sim/ for use in slides.tex.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    with open(path) as f:
        rows = [json.loads(L) for L in f]
    gens = np.array([r["gen"] for r in rows])
    best = np.array([r["best"] for r in rows])
    means = np.array([r.get("mean", np.nan) for r in rows])
    # Running-max of best (CMA can dip across restarts)
    return gens, np.maximum.accumulate(best), means


def main():
    runs = [
        ("results/20260422_005700/cma_gait_2500/tune_gait.jsonl",
         "Mammalian (gen 0–2500)", "C0"),
        ("tune_gait.jsonl",
         "X-config (gen 0–500)",   "C3"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for path, label, color in runs:
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        gens, best, means = load_log(path)
        # Plot best-so-far
        ax.plot(np.arange(len(best)), best, color=color, lw=2, label=label)
    ax.set_xlabel("Generation (cumulative across restarts)")
    ax.set_ylabel("Best reward")
    ax.set_title("CMA convergence — mammalian vs ANYmal X-config")
    ax.grid(True, alpha=0.3)
    ax.axhline(820.7, color="C0", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(686.2, color="C3", ls="--", lw=0.8, alpha=0.7)
    ax.text(2400, 825, "+820 (mammalian best)", fontsize=8, color="C0",
            ha="right", va="bottom")
    ax.text(500, 691, "+686 (X-config best)", fontsize=8, color="C3",
            ha="right", va="bottom")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = "convergence_xconfig.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
