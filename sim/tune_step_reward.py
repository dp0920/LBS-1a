#!/usr/bin/env python3
"""
Diagnostic: run the RL env under different action strategies and plot the
per-step reward (and its components). The goal is to sanity-check that
`RewardAccumulator.step_reward()` returns sensible magnitudes — if penalties
dominate forward reward at typical gait speeds, PPO will learn to stand
still or quick-fail instead of walking.

Usage:
  python tune_step_reward.py                    # random, standing, gait
  python tune_step_reward.py --gait path.json   # replay a best_gait.json

Outputs:
  step_reward_diagnostic.png    per-step reward traces
  plus printed summary stats for each strategy
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from gym_env import OptimusPrimalEnv, ACTION_LOW, ACTION_HIGH


def run_strategy(env, action_fn, n_steps=400, label=""):
    """Run `n_steps` in the env using action_fn(step, obs) → action.
    Returns dict of per-step traces."""
    obs, _ = env.reset()
    rewards, xs, zs, pitches, falls = [], [], [], [], []
    for i in range(n_steps):
        a = action_fn(i, obs)
        obs, r, term, trunc, info = env.step(a)
        rewards.append(r)
        xs.append(info["x"])
        zs.append(float(env.data.qpos[2]))
        pitches.append(np.degrees(float(obs[18])))  # pitch
        falls.append(info["fell"])
        if term or trunc:
            break
    n = len(rewards)
    print(f"\n=== {label} ({n} steps) ===")
    print(f"  reward: mean={np.mean(rewards):+.3f}  sum={np.sum(rewards):+.2f}  "
          f"min={np.min(rewards):+.3f}  max={np.max(rewards):+.3f}")
    print(f"  fell: {any(falls)}  final_x: {xs[-1]:+.3f}m  "
          f"final_z: {zs[-1]:.3f}m")
    return {"reward": np.array(rewards), "x": np.array(xs),
            "z": np.array(zs), "pitch": np.array(pitches),
            "label": label, "fell": any(falls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", type=str, default=None,
                    help="Optional best_gait.json to replay through the env")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    env = OptimusPrimalEnv(max_steps=args.steps + 10)

    # Strategy 1: all-zeros (not in-bounds, will get clipped) — baseline
    def zero_action(i, obs):
        return np.zeros(8, dtype=np.float32)

    # Strategy 2: standing pose held constant
    STAND = np.array([
        np.radians(35), np.radians(-80),   # FL
        np.radians(35), np.radians(-80),   # FR
        np.radians(35), np.radians(-50),   # RL
        np.radians(35), np.radians(-50),   # RR
    ], dtype=np.float32)

    def stand_action(i, obs):
        return STAND

    # Strategy 3: pure random in bounds
    def random_action(i, obs):
        return np.random.uniform(ACTION_LOW, ACTION_HIGH).astype(np.float32)

    # Strategy 4: small sine wiggle around stance (oscillate hip)
    def wiggle_action(i, obs):
        a = STAND.copy()
        a[0] += 0.2 * np.sin(2 * np.pi * i / 40)   # hip_fl
        a[4] += 0.2 * np.sin(2 * np.pi * i / 40 + np.pi)   # hip_rl
        return np.clip(a, ACTION_LOW, ACTION_HIGH).astype(np.float32)

    strategies = [
        (stand_action, "stand (action held at standing pose)"),
        (wiggle_action, "wiggle (sine oscillation around stance)"),
        (random_action, "random (uniform in bounds)"),
    ]

    # Optional: replay a CMA gait through the env by sampling phase
    # interpolation at the env control rate.
    if args.gait:
        from mujoco_gait import decode_params, lerp_pose, PHASE_ORDER
        d = json.load(open(args.gait))
        poses, phase_time = decode_params(np.array(d["params"]))
        interp = d.get("config", {}).get("interp", "linear")
        stride = PHASE_ORDER[1:]
        env_step_dt = env.dt * env.ctrl_repeat

        def gait_action(i, obs):
            # Which cycle and phase are we in?
            t_abs = i * env_step_dt
            phase_idx = int(t_abs / phase_time) % len(stride)
            phase_t = (t_abs / phase_time) % 1.0
            from_name = "start" if phase_idx == 0 else stride[phase_idx - 1]
            to_name = stride[phase_idx]
            pose = lerp_pose(poses[from_name], poses[to_name], phase_t, interp)
            return np.array([
                pose["hip_fl"], pose["knee_fl"],
                pose["hip_fr"], pose["knee_fr"],
                pose["hip_rl"], pose["knee_rl"],
                pose["hip_rr"], pose["knee_rr"],
            ], dtype=np.float32)

        strategies.append(
            (gait_action, f"gait (replay {args.gait})"))

    traces = []
    for fn, label in strategies:
        traces.append(run_strategy(env, fn, n_steps=args.steps, label=label))

    env.close()

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for tr in traces:
        t = np.arange(len(tr["reward"])) * (env.dt * env.ctrl_repeat)
        axes[0].plot(t, tr["reward"], label=tr["label"], linewidth=1)
        axes[1].plot(t, tr["x"], label=tr["label"], linewidth=1)
        axes[2].plot(t, tr["pitch"], label=tr["label"], linewidth=1)

    axes[0].set_ylabel("Per-step reward")
    axes[0].axhline(0, color="k", linewidth=0.5, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("RL step_reward diagnostic")

    axes[1].set_ylabel("Forward x (m)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Pitch (deg)")
    axes[2].axhline(20, color="r", linestyle="--", alpha=0.3)
    axes[2].axhline(-20, color="r", linestyle="--", alpha=0.3)
    axes[2].set_xlabel("Sim time (s)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = "step_reward_diagnostic.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
