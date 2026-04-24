#!/usr/bin/env python3
"""
Correlation + PCA analysis of per-step reward components.

Answers: "Which reward terms are redundant, and which are independent
signals?" Roll out a trained policy, collect per-step contributions of
every reward term, then compute the correlation matrix and its
eigendecomposition.

Interpretation:
  - |corr(A, B)| near 1 → A and B co-vary (redundant, drop one or merge).
  - Top eigenvectors = the "base vectors" of the reward space.
  - Small eigenvalues = directions with little variance (those components
    probably aren't doing anything useful — consider dropping).

Usage:
  python reward_pca.py --policy ppo_v12.zip --episodes 20
"""
import argparse
import numpy as np

from gym_env import OptimusPrimalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def collect_components(policy_path, n_episodes, max_steps, fall_tilt,
                       ctrl_repeat, deterministic,
                       velocity_shape, velocity_bonus,
                       gait_reward_scale, stride_bonus,
                       weight_transfer_bonus):
    env = OptimusPrimalEnv(max_steps=max_steps, fall_tilt_deg=fall_tilt,
                           ctrl_repeat=ctrl_repeat,
                           velocity_shape=velocity_shape,
                           velocity_bonus=velocity_bonus,
                           gait_reward_scale=gait_reward_scale,
                           stride_bonus=stride_bonus,
                           weight_transfer_bonus=weight_transfer_bonus)
    vn_path = policy_path.replace(".zip", "_vecnormalize.pkl")
    dv = VecNormalize.load(vn_path, DummyVecEnv([lambda: env]))
    dv.training = False
    dv.norm_reward = False
    model = PPO.load(policy_path, env=dv)

    rows = []
    keys = None
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            obs_n = dv.normalize_obs(obs.reshape(1, -1))[0]
            action, _ = model.predict(obs_n, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(action)
            comps = info["reward_components"]
            if keys is None:
                keys = list(comps.keys())
            rows.append([comps[k] for k in keys])
            if term or trunc:
                break
    return np.array(rows, dtype=np.float64), keys


def summarize(X, keys):
    """Print means, stds, correlation matrix, and top eigenvectors."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    print("\n=== Per-term mean / std contribution per step ===")
    for k, m, s in zip(keys, mean, std):
        print(f"  {k:18s} mean={m:+8.3f}   std={s:8.3f}")

    # Correlation matrix — use numpy with handling for zero-std columns
    # (a term that's always 0 has undefined correlation).
    nonzero = std > 1e-9
    if not np.all(nonzero):
        zero_keys = [k for k, nz in zip(keys, nonzero) if not nz]
        print(f"\n  Warning: always-zero columns excluded from "
              f"correlation/PCA: {zero_keys}")
    keys_live = [k for k, nz in zip(keys, nonzero) if nz]
    X_live = X[:, nonzero]
    C = np.corrcoef(X_live, rowvar=False)

    print(f"\n=== Correlation matrix ({len(keys_live)}×{len(keys_live)}) ===")
    print(f"  {'':20s} " + "  ".join(f"{k[:6]:>6s}" for k in keys_live))
    for i, k in enumerate(keys_live):
        vals = "  ".join(f"{C[i, j]:+.2f}" for j in range(len(keys_live)))
        print(f"  {k:20s} {vals}")

    # Pairs with high correlation (redundancy flags)
    print("\n=== Highly correlated pairs (|corr| > 0.7) ===")
    flagged = []
    for i in range(len(keys_live)):
        for j in range(i + 1, len(keys_live)):
            if abs(C[i, j]) > 0.7:
                flagged.append((keys_live[i], keys_live[j], C[i, j]))
    if not flagged:
        print("  (none — every term is contributing an independent signal)")
    for a, b, c in sorted(flagged, key=lambda t: -abs(t[2])):
        print(f"  {a:18s} <-> {b:18s}   corr = {c:+.3f}")

    # Eigendecomposition of correlation matrix
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    print("\n=== Principal components of the reward space ===")
    total_var = eigvals.sum()
    for i in range(len(keys_live)):
        pct = 100 * eigvals[i] / total_var
        bar = "█" * int(round(pct / 2))
        print(f"  PC{i+1:2d}  eigvalue={eigvals[i]:6.3f}  "
              f"variance={pct:5.1f}% {bar}")
        # Show largest loadings
        loadings = eigvecs[:, i]
        lo_order = np.argsort(np.abs(loadings))[::-1]
        for idx in lo_order[:4]:
            if abs(loadings[idx]) > 0.15:
                print(f"          {keys_live[idx]:18s} "
                      f"loading = {loadings[idx]:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--fall-tilt", type=float, default=30.0)
    ap.add_argument("--ctrl-repeat", type=int, default=8)
    ap.add_argument("--deterministic", action="store_true")
    # Pass the training-time reward config through so the reported
    # per-step components match what the policy actually experienced.
    ap.add_argument("--velocity-shape", type=str, default="trig",
                    choices=["linear", "quadratic", "cubic", "trig"])
    ap.add_argument("--velocity-bonus", type=float, default=5.0)
    ap.add_argument("--gait-reward-scale", type=float, default=0.25)
    ap.add_argument("--stride-bonus", type=float, default=1.5)
    ap.add_argument("--weight-transfer-bonus", type=float, default=2.0)
    args = ap.parse_args()

    print(f"Collecting {args.episodes} episodes from {args.policy}...")
    X, keys = collect_components(args.policy, args.episodes, args.max_steps,
                                 args.fall_tilt, args.ctrl_repeat,
                                 args.deterministic,
                                 args.velocity_shape, args.velocity_bonus,
                                 args.gait_reward_scale, args.stride_bonus,
                                 args.weight_transfer_bonus)
    print(f"  Got {X.shape[0]} timesteps across {X.shape[1]} components.\n")
    summarize(X, keys)


if __name__ == "__main__":
    main()
