#!/usr/bin/env python3
"""
Plot PPO learning curves from TensorBoard logs.

Saves ppo_learning_curves.png for use in slides.
"""
import os
import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorboard.backend.event_processing import event_accumulator


def load_scalar(tb_dir, tag="rollout/ep_rew_mean"):
    ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None, None
    evs = ea.Scalars(tag)
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])


def main():
    # Hand-pick representative TB dirs spanning the PPO journey
    runs = [
        ("ppo_tb/v20_1",                  "v20  (post-fix champion)",    "C0"),
        ("ppo_tb/v25_scratch_b_1",        "v25  scratch (seed b)",       "C1"),
        ("ppo_tb/v26_body_smooth_b_1",    "v26  body-smooth",            "C2"),
        ("ppo_tb/v30_bodysmooth_dr_1",    "v30  friction-DR",            "C3"),
        ("ppo_tb/v33_xconfig_lowfric_a_1_1",
                                          "v33  X-config + lowfric",     "C4"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for tb, label, color in runs:
        if not os.path.exists(tb):
            print(f"missing: {tb}")
            continue
        steps, vals = load_scalar(tb)
        if steps is None or len(steps) < 2:
            print(f"no data: {tb}")
            continue
        ax.plot(steps / 1e6, vals, color=color, lw=1.8, label=label)
    ax.set_xlabel("Training timesteps (millions)")
    ax.set_ylabel("Episode reward mean")
    ax.set_title("PPO learning curves — selected runs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = "ppo_learning_curves.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
