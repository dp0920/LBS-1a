#!/usr/bin/env python3
"""
Run N episodes of a trained PPO policy (no viewer) and report the
distribution of episode length / reward / distance.

Useful when a single replay in the viewer fell over — this tells you
whether it was bad luck or the policy is genuinely broken.

Usage:
  python ppo_stats.py                            # default: 20 eps of ppo_policy.zip
  python ppo_stats.py --policy ppo_policy_v2.zip --episodes 50
  python ppo_stats.py --deterministic            # mean-action replay
"""
import argparse
import numpy as np

from gym_env import OptimusPrimalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="ppo_policy.zip")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--fall-tilt", type=float, default=20.0,
                    help="Fall-termination tilt threshold in degrees. "
                         "Default 20.0 matches training; try 30–35 if you're "
                         "seeing false FELL calls on a policy that visually "
                         "isn't falling.")
    ap.add_argument("--ctrl-repeat", type=int, default=8,
                    help="Physics substeps per env step (v5 was trained at 4, "
                         "v6/v7 at 8). Must match training — mismatch breaks "
                         "the policy's timing.")
    ap.add_argument("--friction", type=float, default=None,
                    help="Fix the floor tangential friction coefficient for "
                         "every episode (overrides any randomization). Use to "
                         "evaluate a friction-DR policy at a specific operating "
                         "point — e.g. 1.0 for sim-default, ~0.4 for real-world.")
    args = ap.parse_args()

    fric_range = (args.friction, args.friction) if args.friction is not None else None
    env = OptimusPrimalEnv(max_steps=args.max_steps,
                           fall_tilt_deg=args.fall_tilt,
                           ctrl_repeat=args.ctrl_repeat,
                           friction_range=fric_range)
    vn_path = args.policy.replace(".zip", "_vecnormalize.pkl")
    dv = VecNormalize.load(vn_path, DummyVecEnv([lambda: env]))
    dv.training = False
    dv.norm_reward = False
    model = PPO.load(args.policy, env=dv)

    lens, rewards, final_x = [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        tot = 0.0
        for i in range(args.max_steps):
            obs_n = dv.normalize_obs(obs.reshape(1, -1))[0]
            a, _ = model.predict(obs_n, deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(a)
            tot += r
            if term or trunc:
                break
        lens.append(i + 1)
        rewards.append(tot)
        final_x.append(info["x"])
        status = "FELL" if info["fell"] else "ok"
        print(f"ep {ep:2d}: len={i+1:4d}  rew={tot:+8.1f}  "
              f"x={info['x']:+.2f}m  [{status}]")

    print()
    print(f"Across {args.episodes} episodes "
          f"({'deterministic' if args.deterministic else 'stochastic'}):")
    print(f"  len   min/mean/max: {min(lens)} / {np.mean(lens):.0f} / {max(lens)}")
    print(f"  rew   min/mean/max: {min(rewards):+.0f} / {np.mean(rewards):+.0f} / {max(rewards):+.0f}")
    print(f"  dist  min/mean/max: {min(final_x):+.2f} / {np.mean(final_x):+.2f} / {max(final_x):+.2f} m")
    n_survived = sum(1 for L in lens if L >= args.max_steps - 1)
    print(f"  survived full episode: {n_survived}/{args.episodes}")


if __name__ == "__main__":
    main()
