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
             velocity_shape="quadratic", velocity_bonus=5.0,
             gait_reward_scale=0.25, stride_bonus=10.0,
             randomize_init=False, dynamic_posture_target=False,
             weight_transfer_bonus=0.0, extension_bonus=3.0,
             start_pose_json=None,
             body_smoothness_penalty=0.0, foot_drift_penalty=0.0,
             fall_penalty=20.0, survival_bonus=0.0):
    def _init():
        env = OptimusPrimalEnv(
            fall_tilt_deg=fall_tilt_deg,
            tilt_scale=tilt_scale,
            velocity_shape=velocity_shape,
            velocity_bonus=velocity_bonus,
            gait_reward_scale=gait_reward_scale,
            stride_bonus=stride_bonus,
            randomize_init=randomize_init,
            dynamic_posture_target=dynamic_posture_target,
            weight_transfer_bonus=weight_transfer_bonus,
            extension_bonus=extension_bonus,
            start_pose_json=start_pose_json,
            body_smoothness_penalty=body_smoothness_penalty,
            foot_drift_penalty=foot_drift_penalty,
            fall_penalty=fall_penalty,
            survival_bonus=survival_bonus,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def linear_lr_schedule(initial=3e-4, final=1e-5):
    """Linear decay from `initial` to `final` over the training run.

    SB3 passes `progress_remaining` ∈ [1.0, 0.0] (1 at start, 0 at end),
    so we lerp accordingly. Late-training refinement with a smaller LR
    reduces policy oscillation after the value function has settled.
    """
    def schedule(progress_remaining):
        return final + progress_remaining * (initial - final)
    return schedule


def train(timesteps, n_envs, out_path, log_dir, fall_tilt_deg, tilt_scale,
          tb_name=None, velocity_shape="quadratic", velocity_bonus=5.0,
          lr_schedule="linear", ent_coef=0.0,
          gait_reward_scale=0.25, stride_bonus=10.0,
          randomize_init=False, dynamic_posture_target=False,
          weight_transfer_bonus=0.0, extension_bonus=3.0,
          init_from=None, start_pose_json=None,
          body_smoothness_penalty=0.0, foot_drift_penalty=0.0,
          fall_penalty=20.0, survival_bonus=0.0):
    envs = SubprocVecEnv([make_env(seed=i,
                                   fall_tilt_deg=fall_tilt_deg,
                                   tilt_scale=tilt_scale,
                                   velocity_shape=velocity_shape,
                                   velocity_bonus=velocity_bonus,
                                   gait_reward_scale=gait_reward_scale,
                                   stride_bonus=stride_bonus,
                                   randomize_init=randomize_init,
                                   dynamic_posture_target=dynamic_posture_target,
                                   weight_transfer_bonus=weight_transfer_bonus,
                                   extension_bonus=extension_bonus,
                                   start_pose_json=start_pose_json,
                                   body_smoothness_penalty=body_smoothness_penalty,
                                   foot_drift_penalty=foot_drift_penalty,
                                   fall_penalty=fall_penalty,
                                   survival_bonus=survival_bonus)
                          for i in range(n_envs)])
    # Normalize obs + reward so learning signal isn't dominated by scale.
    if init_from is not None:
        # Warm-start: load VecNormalize stats from BC pretrain so the policy
        # sees obs in the same scale it was trained on. Stats keep updating
        # during PPO learning (training=True is the default).
        vn_path = init_from.replace(".zip", "_vecnormalize.pkl")
        envs = VecNormalize.load(vn_path, envs)
    else:
        envs = VecNormalize(envs, norm_obs=True, norm_reward=True,
                             clip_obs=10.0)

    if lr_schedule == "linear":
        lr = linear_lr_schedule(3e-4, 1e-5)
    else:
        lr = 3e-4

    if init_from is not None:
        # Load BC-pretrained policy + value weights, then continue training.
        model = PPO.load(init_from, env=envs,
                         tensorboard_log=log_dir,
                         learning_rate=lr,
                         ent_coef=ent_coef)
    else:
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
            learning_rate=lr,
            ent_coef=ent_coef,
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
    ap.add_argument("--lr-schedule", type=str, default="linear",
                    choices=["linear", "constant"],
                    help="linear: decay 3e-4 → 1e-5 over training. "
                         "constant: 3e-4 throughout.")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="PPO entropy coefficient. Default 0.0 (SB3 default). "
                         "Try 0.01 if policy is collapsing to deterministic "
                         "before exploring the observation space.")
    ap.add_argument("--gait-reward-scale", type=float, default=0.25,
                    help="Scales the forced FR→RL→FL→RR contact-pattern "
                         "reward. 0.0 disables it entirely (lets PPO find "
                         "its own gait). Default 0.25.")
    ap.add_argument("--stride-bonus", type=float, default=10.0,
                    help="Bonus per plant event, times world-frame forward "
                         "foot displacement during swing. Encourages fewer, "
                         "larger strides (v8).")
    ap.add_argument("--randomize-init", action="store_true",
                    help="Per-episode random initial body-z + ±5° joint "
                         "jitter (v9 domain randomization).")
    ap.add_argument("--dynamic-posture-target", action="store_true",
                    help="Posture band targets the per-episode starting z "
                         "instead of a fixed 0.15 m — policy learns to "
                         "maintain whatever height it was placed at (v9).")
    ap.add_argument("--weight-transfer-bonus", type=float, default=0.0,
                    help="Bonus for target swing leg having low ground-"
                         "contact force (v10). Rewards 'shift weight off "
                         "this leg before lifting' pre-stride behavior. "
                         "Try 2.0 to start.")
    ap.add_argument("--extension-bonus", type=float, default=3.0,
                    help="Bonus on action distance from joint-bound midpoint "
                         "(v5). Default 3.0 was the v5 unlock; ablate by "
                         "setting to 0.")
    ap.add_argument("--init-from", type=str, default=None,
                    help="Path to a saved PPO .zip to warm-start from "
                         "(e.g. a BC-pretrained policy). Loads policy+value "
                         "weights and the saved VecNormalize stats, then "
                         "continues training with normal PPO learning.")
    ap.add_argument("--start-from-gait", type=str, default=None,
                    help="Path to a CMA gait JSON. Use that gait's 'start' "
                         "phase pose as the per-episode reset stance instead "
                         "of the hardcoded symmetric squat.")
    ap.add_argument("--body-smoothness-penalty", type=float, default=0.0,
                    help="Penalty coef on |Δroll| + |Δpitch| step-to-step "
                         "(v26). Targets 'rocking' gaits. Try ~10-50.")
    ap.add_argument("--foot-drift-penalty", type=float, default=0.0,
                    help="Penalty coef on Σ|Δfoot_y| across all 4 feet "
                         "step-to-step (v26). Targets 'waddly' gaits. "
                         "Try ~50-200.")
    ap.add_argument("--fall-penalty", type=float, default=20.0,
                    help="One-shot magnitude on episode-ending fall (v27). "
                         "Default 20.0; bump to 200+ to make falls "
                         "catastrophic.")
    ap.add_argument("--survival-bonus", type=float, default=0.0,
                    help="One-shot bonus if episode reaches max_steps "
                         "without falling (v27). Try 500+ for survival-"
                         "priority training.")
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
              velocity_bonus=args.velocity_bonus,
              lr_schedule=args.lr_schedule,
              ent_coef=args.ent_coef,
              gait_reward_scale=args.gait_reward_scale,
              stride_bonus=args.stride_bonus,
              randomize_init=args.randomize_init,
              dynamic_posture_target=args.dynamic_posture_target,
              weight_transfer_bonus=args.weight_transfer_bonus,
              extension_bonus=args.extension_bonus,
              init_from=args.init_from,
              start_pose_json=args.start_from_gait,
              body_smoothness_penalty=args.body_smoothness_penalty,
              foot_drift_penalty=args.foot_drift_penalty,
              fall_penalty=args.fall_penalty,
              survival_bonus=args.survival_bonus)


if __name__ == "__main__":
    main()
