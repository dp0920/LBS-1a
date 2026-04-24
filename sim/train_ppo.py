#!/usr/bin/env python3
"""
Train a PPO policy on OptimusPrimalEnv.

Usage:
  python train_ppo.py                       # quick smoke (50k steps, 4 envs)
  python train_ppo.py --timesteps 1000000   # longer run
  python train_ppo.py --n-envs 8            # more parallel envs
  python train_ppo.py --replay ppo_policy.zip  # visualize a trained policy

Monitor:
  tensorboard --logdir ./ppo_tb
"""
import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from gym_env import OptimusPrimalEnv


def make_env(seed=0, fall_tilt_deg=20.0, tilt_scale=1.0,
             velocity_shape="quadratic", velocity_bonus=5.0):
    def _init():
        env = OptimusPrimalEnv(
            fall_tilt_deg=fall_tilt_deg,
            tilt_scale=tilt_scale,
            velocity_shape=velocity_shape,
            velocity_bonus=velocity_bonus,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train(timesteps, n_envs, out_path, log_dir, fall_tilt_deg, tilt_scale,
          tb_name=None, velocity_shape="quadratic", velocity_bonus=5.0):
    envs = SubprocVecEnv([make_env(seed=i,
                                   fall_tilt_deg=fall_tilt_deg,
                                   tilt_scale=tilt_scale,
                                   velocity_shape=velocity_shape,
                                   velocity_bonus=velocity_bonus)
                          for i in range(n_envs)])
    # Normalize obs + reward so learning signal isn't dominated by scale.
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        envs,
        verbose=1,
        tensorboard_log=log_dir,
        # Reasonable defaults for continuous control; tweak later.
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    learn_kwargs = {"total_timesteps": timesteps, "progress_bar": False}
    if tb_name:
        learn_kwargs["tb_log_name"] = tb_name
    model.learn(**learn_kwargs)
    model.save(out_path)
    # Save the vec-normalize stats alongside the policy.
    envs.save(out_path.replace(".zip", "_vecnormalize.pkl"))
    envs.close()
    print(f"Saved {out_path} and _vecnormalize.pkl")


def replay(policy_path, n_steps=2000, fall_tilt_deg=20.0, tilt_scale=1.0,
           args=None):
    """Load a trained policy and run it in the viewer."""
    import mujoco.viewer
    import mujoco
    import time as _time

    env = OptimusPrimalEnv(
        max_steps=n_steps,
        fall_tilt_deg=fall_tilt_deg,
        tilt_scale=tilt_scale,
    )
    # Reload vec-normalize stats if present.
    from stable_baselines3.common.vec_env import DummyVecEnv
    vn_path = policy_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vn_path):
        dummy = DummyVecEnv([lambda: env])
        dummy = VecNormalize.load(vn_path, dummy)
        dummy.training = False
        dummy.norm_reward = False
        model = PPO.load(policy_path, env=dummy)
        normed_env = dummy
    else:
        model = PPO.load(policy_path)
        normed_env = None

    obs, _ = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        total_r = 0.0
        for i in range(n_steps):
            if normed_env is not None:
                obs_in = normed_env.normalize_obs(obs.reshape(1, -1))[0]
            else:
                obs_in = obs
            action, _ = model.predict(obs_in,
                                      deterministic=args.deterministic)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            viewer.sync()
            _time.sleep(env.dt * env.ctrl_repeat)
            if term or trunc or not viewer.is_running():
                break
        print(f"Replay done. steps={i+1}  total_reward={total_r:+.2f}  "
              f"x={info['x']:+.3f}m  fell={info['fell']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=50_000,
                    help="Total PPO timesteps. Default 50k (smoke test); "
                         "real run ~1M+.")
    ap.add_argument("--n-envs", type=int, default=4,
                    help="Parallel envs for SubprocVecEnv")
    ap.add_argument("--out", type=str, default="ppo_policy.zip")
    ap.add_argument("--log-dir", type=str, default="./ppo_tb")
    ap.add_argument("--fall-tilt", type=float, default=20.0)
    ap.add_argument("--tilt-scale", type=float, default=1.0)
    ap.add_argument("--replay", type=str, default=None,
                    help="Path to a saved policy to visualize (skips train)")
    ap.add_argument("--deterministic", action="store_true",
                    help="Use mean action in replay (default: stochastic, "
                         "which matches training-time behavior)")
    ap.add_argument("--tb-name", type=str, default=None,
                    help="TensorBoard sub-run name (e.g. 'v3'). Default: "
                         "auto-increments as PPO_N")
    ap.add_argument("--velocity-shape", type=str, default="quadratic",
                    choices=["linear", "quadratic", "cubic", "trig"],
                    help="Shape of the velocity bonus term. See "
                         "OptimusPrimalEnv.velocity_shape docstring.")
    ap.add_argument("--velocity-bonus", type=float, default=5.0,
                    help="Scalar coefficient on the velocity bonus term "
                         "(default 5.0)")
    args = ap.parse_args()

    if args.replay:
        replay(args.replay,
               fall_tilt_deg=args.fall_tilt,
               tilt_scale=args.tilt_scale,
               args=args)
    else:
        train(args.timesteps, args.n_envs, args.out, args.log_dir,
              args.fall_tilt, args.tilt_scale,
              tb_name=args.tb_name,
              velocity_shape=args.velocity_shape,
              velocity_bonus=args.velocity_bonus)


if __name__ == "__main__":
    main()
