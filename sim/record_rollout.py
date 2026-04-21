#!/usr/bin/env python3
"""
Record a full rollout from a trained gait and generate analysis plots.

Captures per-timestep: body position, orientation, joint angles (actual vs
commanded), nose-down angle, pitch rate. Saves to .npz and generates plots.

Usage:
  python record_rollout.py best_gait.json
  python record_rollout.py best_gait.json --cycles 10
  python record_rollout.py --compare best_gait.json best_gait_random.json
"""
import argparse
import json
import os
import numpy as np
import mujoco

# Import shared infrastructure from mujoco_gait
from mujoco_gait import (
    build_model, JOINTS, PHASE_ORDER, decode_params,
    get_qadr, get_ctrl_idx, lerp_pose, X0
)


def record_rollout(model, poses, phase_time, n_cycles=5):
    """Run a rollout and record per-step telemetry."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qadr = get_qadr(model)
    ctrl_idx = get_ctrl_idx(model)
    dt = model.opt.timestep

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     "base_link")

    def set_ctrl(pose):
        for name, angle in pose.items():
            data.ctrl[ctrl_idx[name]] = angle

    def set_qpos(pose):
        for name, angle in pose.items():
            data.qpos[qadr[name]] = angle

    def get_rpy():
        qw, qx, qy, qz = data.qpos[3:7]
        roll = np.arctan2(2 * (qw * qx + qy * qz),
                          1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
        yaw = np.arctan2(2 * (qw * qz + qx * qy),
                         1 - 2 * (qy * qy + qz * qz))
        return roll, pitch, yaw

    def get_nose_down():
        mat = data.xmat[base_body_id].reshape(3, 3)
        return max(0.0, -float(mat[2, 0]))

    # Recording arrays
    rec = {
        "time": [], "x": [], "y": [], "z": [],
        "roll": [], "pitch": [], "yaw": [],
        "nose_down": [], "pitch_rate": [],
        "phase": [], "phase_name": [],
    }
    # Joint angles (actual from qpos) and commands (ctrl)
    for j in JOINTS:
        rec[f"q_{j}"] = []
        rec[f"cmd_{j}"] = []

    step = [0]

    def sample(phase_name=""):
        roll, pitch, yaw = get_rpy()
        rec["time"].append(step[0] * dt)
        rec["x"].append(float(data.qpos[0]))
        rec["y"].append(float(data.qpos[1]))
        rec["z"].append(float(data.qpos[2]))
        rec["roll"].append(np.degrees(roll))
        rec["pitch"].append(np.degrees(pitch))
        rec["yaw"].append(np.degrees(yaw))
        rec["nose_down"].append(np.degrees(np.arcsin(min(1.0, get_nose_down()))))
        rec["pitch_rate"].append(float(data.qvel[4]))
        rec["phase_name"].append(phase_name)
        for j in JOINTS:
            rec[f"q_{j}"].append(np.degrees(float(data.qpos[qadr[j]])))
            rec[f"cmd_{j}"].append(np.degrees(float(data.ctrl[ctrl_idx[j]])))

    def run_phase(from_pose, to_pose, duration, phase_name=""):
        steps = max(1, int(duration / dt))
        for i in range(steps):
            t = i / steps
            set_ctrl(lerp_pose(from_pose, to_pose, t))
            mujoco.mj_step(model, data)
            step[0] += 1
            if step[0] % 5 == 0:  # sample every 5th step to keep data manageable
                sample(phase_name)
            # Check for fall
            if data.qpos[2] < 0.08:
                sample(phase_name)
                return False
            roll, pitch = get_rpy()[:2]
            if abs(roll) > np.radians(30) or abs(pitch) > np.radians(30):
                sample(phase_name)
                return False
        set_ctrl(to_pose)
        mujoco.mj_step(model, data)
        step[0] += 1
        sample(phase_name)
        return True

    # Init
    start_pose = poses["start"]
    set_qpos(start_pose)
    data.qpos[2] = 0.16
    data.qpos[3] = 1.0
    set_ctrl(start_pose)
    mujoco.mj_forward(model, data)

    # Settle
    for i in range(int(2.0 / dt)):
        set_ctrl(start_pose)
        mujoco.mj_step(model, data)
        step[0] += 1
    sample("settle")

    stride_phases = PHASE_ORDER[1:]
    alive = True

    for cycle in range(n_cycles):
        prev_name = "start"
        for phase_name in stride_phases:
            if not run_phase(poses[prev_name], poses[phase_name],
                             phase_time, phase_name):
                alive = False
                break
            prev_name = phase_name
        if not alive:
            break
        # Recenter
        if cycle < n_cycles - 1:
            if not run_phase(poses[stride_phases[-1]], poses["start"],
                             phase_time, "recenter"):
                break

    # Convert to numpy arrays
    result = {}
    for k, v in rec.items():
        if k == "phase_name":
            result[k] = v  # keep as list of strings
        else:
            result[k] = np.array(v)

    return result


def plot_rollout(rec, name, output_prefix=None):
    """Generate analysis plots from recorded rollout data."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if output_prefix is None:
        output_prefix = f"rollout_{name}"

    t = rec["time"]

    # === Plot 1: Body trajectory & height ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, rec["x"], "b-", linewidth=1.5, label="x (forward)")
    axes[0].set_ylabel("X position (m)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{name} — Body Trajectory")

    axes[1].plot(t, rec["z"], "g-", linewidth=1.5)
    axes[1].axhline(y=0.08, color="r", linestyle="--", alpha=0.5, label="fall threshold")
    axes[1].set_ylabel("Z height (m)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, rec["y"], "m-", linewidth=1.5, label="y (lateral)")
    axes[2].set_ylabel("Y position (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{output_prefix}_trajectory.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

    # === Plot 2: Body orientation ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(t, rec["pitch"], "r-", linewidth=1, label="pitch")
    axes[0].axhline(y=30, color="r", linestyle="--", alpha=0.3)
    axes[0].axhline(y=-30, color="r", linestyle="--", alpha=0.3)
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{name} — Body Orientation")

    axes[1].plot(t, rec["roll"], "b-", linewidth=1, label="roll")
    axes[1].axhline(y=30, color="r", linestyle="--", alpha=0.3)
    axes[1].axhline(y=-30, color="r", linestyle="--", alpha=0.3)
    axes[1].set_ylabel("Roll (deg)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, rec["nose_down"], "orange", linewidth=1, label="nose-down")
    axes[2].set_ylabel("Nose-down (deg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{output_prefix}_orientation.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

    # === Plot 3: Joint angles (actual) per leg ===
    leg_joints = {
        "FL": ("hip_fl", "knee_fl"),
        "FR": ("hip_fr", "knee_fr"),
        "RL": ("hip_rl", "knee_rl"),
        "RR": ("hip_rr", "knee_rr"),
    }
    colors = {"FL": "#e41a1c", "FR": "#377eb8", "RL": "#4daf4a", "RR": "#ff7f00"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for ax, (leg, (hip, knee)) in zip(axes.flat, leg_joints.items()):
        c = colors[leg]
        ax.plot(t, rec[f"q_{hip}"], "-", color=c, linewidth=1, label="hip actual")
        ax.plot(t, rec[f"cmd_{hip}"], "--", color=c, linewidth=0.8,
                alpha=0.5, label="hip cmd")
        ax.plot(t, rec[f"q_{knee}"], "-", color=c, linewidth=1,
                alpha=0.6, label="knee actual")
        ax.plot(t, rec[f"cmd_{knee}"], ":", color=c, linewidth=0.8,
                alpha=0.4, label="knee cmd")
        ax.set_title(leg, fontweight="bold", color=c)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Angle (deg)")

    for ax in axes[1]:
        ax.set_xlabel("Time (s)")

    fig.suptitle(f"{name} — Joint Angles (actual vs commanded)", fontsize=12)
    plt.tight_layout()
    out = f"{output_prefix}_joints.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()

    # === Plot 4: Distance over time ===
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, rec["x"], "b-", linewidth=2)
    ax.fill_between(t, 0, rec["x"], alpha=0.1, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward distance (m)")
    ax.set_title(f"{name} — Distance Traveled")
    ax.grid(True, alpha=0.3)
    final_x = rec["x"][-1] if len(rec["x"]) > 0 else 0
    final_t = t[-1] if len(t) > 0 else 0
    ax.annotate(f"{final_x:.3f} m in {final_t:.1f}s",
                xy=(final_t, final_x), fontsize=11,
                ha="right", va="bottom")
    plt.tight_layout()
    out = f"{output_prefix}_distance.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_comparison(recordings, output="rollout_comparison.png"):
    """Overlay multiple gaits' x-distance and pitch on one plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for name, rec in recordings:
        t = rec["time"]
        axes[0].plot(t, rec["x"], linewidth=1.5, label=name)
        axes[1].plot(t, rec["pitch"], linewidth=1, label=name, alpha=0.8)

    axes[0].set_ylabel("Forward distance (m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Gait Comparison — Distance & Stability")

    axes[1].set_ylabel("Pitch (deg)")
    axes[1].axhline(y=30, color="r", linestyle="--", alpha=0.3)
    axes[1].axhline(y=-30, color="r", linestyle="--", alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="best_gait*.json files to record")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--compare", action="store_true",
                    help="Overlay distance/pitch comparison")
    ap.add_argument("--save-npz", action="store_true",
                    help="Save raw data as .npz files")
    args = ap.parse_args()

    model = build_model()
    recordings = []

    for path in args.files:
        with open(path) as f:
            d = json.load(f)
        poses, pt = decode_params(np.array(d["params"]))
        reward = d.get("reward", 0)

        name = os.path.splitext(os.path.basename(path))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if parent and parent not in ("sim", "."):
            name = parent

        print(f"\nRecording {name} (reward={reward:.1f}, "
              f"phase_time={pt:.3f}s, {args.cycles} cycles)...")
        rec = record_rollout(model, poses, pt, n_cycles=args.cycles)

        final_x = rec["x"][-1] if len(rec["x"]) > 0 else 0
        final_t = rec["time"][-1] if len(rec["time"]) > 0 else 0
        print(f"  -> {final_x:.3f}m in {final_t:.1f}s  "
              f"({len(rec['time'])} samples)")

        if args.save_npz:
            npz_path = f"rollout_{name}.npz"
            np.savez(npz_path, **{k: v for k, v in rec.items()
                                  if k != "phase_name"})
            print(f"  -> Saved {npz_path}")

        plot_rollout(rec, name)
        recordings.append((name, rec))

    if args.compare and len(recordings) > 1:
        plot_comparison(recordings)


if __name__ == "__main__":
    main()
