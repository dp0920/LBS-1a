#!/usr/bin/env python3
"""
Visualize the motor angle choreography from trained gaits.

Usage:
  python plot_gaits.py best_gait.json                     # single gait
  python plot_gaits.py results/*/best_gait*.json           # all results
  python plot_gaits.py --compare best_gait.json best_gait_random.json
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

PHASE_ORDER = [
    "start",
    "shift_FR", "swing_FR", "plant_FR",
    "shift_RL", "swing_RL", "plant_RL",
    "shift_FL", "swing_FL", "plant_FL",
    "shift_RR", "swing_RR", "plant_RR",
]

JOINT_LABELS = [
    "FL hip", "FL knee",
    "FR hip", "FR knee",
    "RL hip", "RL knee",
    "RR hip", "RR knee",
]

LEG_COLORS = {
    "FL": "#e41a1c",  # red
    "FR": "#377eb8",  # blue
    "RL": "#4daf4a",  # green
    "RR": "#ff7f00",  # orange
}


def decode_gait(path):
    """Load a best_gait JSON → (13×8 angle array, phase_time, reward, name)."""
    with open(path) as f:
        d = json.load(f)
    params = d["params"]
    n_phases = len(PHASE_ORDER)
    angles = np.array(params[:n_phases * 8]).reshape(n_phases, 8)
    phase_time = params[n_phases * 8]
    reward = d.get("reward", 0)
    name = os.path.splitext(os.path.basename(path))[0]
    # Try to get a nicer name from parent directory
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent and parent != "sim" and parent != ".":
        name = parent
    return angles, phase_time, reward, name


def plot_heatmap(angles, phase_time, reward, name, output=None):
    """Heatmap: phases (rows) × joints (columns), color = angle."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate color scale for hips (positive) and knees (negative)
    im = ax.imshow(angles, aspect="auto", cmap="RdYlBu_r",
                   interpolation="nearest")

    ax.set_xticks(range(8))
    ax.set_xticklabels(JOINT_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(PHASE_ORDER)))
    ax.set_yticklabels(PHASE_ORDER)

    # Annotate each cell with the angle value
    for i in range(len(PHASE_ORDER)):
        for j in range(8):
            val = angles[i, j]
            color = "white" if abs(val) > 60 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Angle (degrees)")
    ax.set_title(f"{name}  |  reward={reward:.1f}  |  phase_time={phase_time:.3f}s")
    plt.tight_layout()

    if output is None:
        output = f"gait_heatmap_{name}.png"
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close()


def plot_joint_trajectories(angles, phase_time, reward, name, output=None):
    """Line plot: one line per joint across all 13 phases."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    phases_x = range(len(PHASE_ORDER))

    # Top plot: hip angles
    ax = axes[0]
    for leg, color in LEG_COLORS.items():
        col_idx = JOINT_LABELS.index(f"{leg} hip")
        ax.plot(phases_x, angles[:, col_idx], "o-", color=color,
                label=f"{leg} hip", linewidth=2, markersize=4)
    ax.set_ylabel("Hip angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{name}  |  reward={reward:.1f}")

    # Bottom plot: knee angles
    ax = axes[1]
    for leg, color in LEG_COLORS.items():
        col_idx = JOINT_LABELS.index(f"{leg} knee")
        ax.plot(phases_x, angles[:, col_idx], "s--", color=color,
                label=f"{leg} knee", linewidth=2, markersize=4)
    ax.set_ylabel("Knee angle (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(phases_x)
    ax.set_xticklabels(PHASE_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Phase")

    # Shade the phase groups (shift/swing/plant per leg)
    for a in axes:
        for i in range(1, 13, 3):
            a.axvspan(i - 0.5, i + 2.5, alpha=0.06, color="gray")

    plt.tight_layout()
    if output is None:
        output = f"gait_trajectory_{name}.png"
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close()


def plot_per_leg(angles, phase_time, reward, name, output=None):
    """One subplot per leg showing hip + knee coordination."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
    legs = ["FL", "FR", "RL", "RR"]
    phases_x = range(len(PHASE_ORDER))

    for ax, leg in zip(axes.flat, legs):
        hip_idx = JOINT_LABELS.index(f"{leg} hip")
        knee_idx = JOINT_LABELS.index(f"{leg} knee")
        color = LEG_COLORS[leg]

        ax.plot(phases_x, angles[:, hip_idx], "o-", color=color,
                label="Hip", linewidth=2)
        ax.plot(phases_x, angles[:, knee_idx], "s--", color=color,
                label="Knee", linewidth=2, alpha=0.7)

        # Highlight this leg's own shift/swing/plant phases
        leg_phases = [f"shift_{leg}", f"swing_{leg}", f"plant_{leg}"]
        for pname in leg_phases:
            if pname in PHASE_ORDER:
                idx = PHASE_ORDER.index(pname)
                ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.15, color=color)

        ax.set_title(f"{leg}", fontsize=12, fontweight="bold", color=color)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[1]:
        ax.set_xticks(phases_x)
        ax.set_xticklabels(PHASE_ORDER, rotation=45, ha="right", fontsize=7)

    fig.suptitle(f"{name}  |  reward={reward:.1f}  |  phase_time={phase_time:.3f}s",
                 fontsize=12)
    plt.tight_layout()

    if output is None:
        output = f"gait_legs_{name}.png"
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close()


def plot_comparison_heatmaps(gaits, output="gait_comparison.png"):
    """Side-by-side heatmaps for multiple gaits."""
    n = len(gaits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7))
    if n == 1:
        axes = [axes]

    vmin = min(a.min() for a, _, _, _ in gaits)
    vmax = max(a.max() for a, _, _, _ in gaits)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ax, (angles, pt, reward, name) in zip(axes, gaits):
        im = ax.imshow(angles, aspect="auto", cmap="RdYlBu_r",
                       norm=norm, interpolation="nearest")
        ax.set_xticks(range(8))
        ax.set_xticklabels(JOINT_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(PHASE_ORDER)))
        ax.set_yticklabels(PHASE_ORDER, fontsize=7)
        ax.set_title(f"{name}\nreward={reward:.1f}", fontsize=9)

        for i in range(len(PHASE_ORDER)):
            for j in range(8):
                val = angles[i, j]
                color = "white" if abs(val) > 60 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=axes, label="Angle (degrees)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="best_gait*.json files")
    ap.add_argument("--compare", action="store_true",
                    help="Side-by-side heatmap comparison")
    args = ap.parse_args()

    gaits = []
    for path in args.files:
        try:
            angles, pt, reward, name = decode_gait(path)
            gaits.append((angles, pt, reward, name))
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not gaits:
        print("No valid gait files found.")
        return

    if args.compare and len(gaits) > 1:
        plot_comparison_heatmaps(gaits)

    for angles, pt, reward, name in gaits:
        plot_heatmap(angles, pt, reward, name)
        plot_joint_trajectories(angles, pt, reward, name)
        plot_per_leg(angles, pt, reward, name)


if __name__ == "__main__":
    main()
