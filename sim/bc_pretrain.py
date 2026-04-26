#!/usr/bin/env python3
"""
Behavioral cloning pre-training: warm-start a PPO policy by imitating a
trained CMA gait.

Workflow:
  1. Roll out the CMA gait in the RL env (with init-pose noise for diversity),
     recording (observation, CMA-action) pairs at each control step.
  2. Train PPO's actor network to predict the CMA action given the observation,
     via MSE supervised learning.
  3. Save the resulting PPO model so it can be loaded by train_ppo.py and
     fine-tuned with normal RL learning.

Usage:
  python bc_pretrain.py best_cma.json --out ppo_bc_init.zip
  python bc_pretrain.py best.json --episodes 100 --epochs 100 --noise-deg 3.0
"""
import argparse
import json
import os

import numpy as np
import torch
import mujoco

from gym_env import OptimusPrimalEnv, ACTION_LOW, ACTION_HIGH
from sim_core import JOINTS, get_qadr, get_ctrl_idx
from mujoco_gait import PHASE_ORDER, decode_params, lerp_pose
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def collect_demos(gait_json, n_episodes, noise_deg, noise_z, ctrl_repeat,
                  rng):
    """Roll out CMA gait in the RL env, collect (obs, action) pairs."""
    with open(gait_json) as f:
        d = json.load(f)
    params = np.array(d["params"])
    interp = d.get("config", {}).get("interp", "linear")
    poses, phase_time = decode_params(params)
    phase_names = PHASE_ORDER
    n_phases = len(phase_names) - 1   # excluding 'start' as a transition source
    init_noise_rad = float(np.radians(noise_deg))

    # Build the env we'll be training the policy in (so obs format matches).
    env = OptimusPrimalEnv(max_steps=4000, fall_tilt_deg=30.0,
                           ctrl_repeat=ctrl_repeat,
                           velocity_shape="trig", velocity_bonus=5.0,
                           gait_reward_scale=0.25, stride_bonus=5.0,
                           weight_transfer_bonus=2.0)
    qadr = get_qadr(env.model)
    ctrl_idx = get_ctrl_idx(env.model)

    obs_list, act_list = [], []
    distances = []
    for ep in range(n_episodes):
        # Manual reset (skip env.reset's default stance pose since we want
        # the CMA start pose). Then settle the robot under that pose.
        mujoco.mj_resetData(env.model, env.data)
        start = dict(poses["start"])
        for j in start:
            start[j] += rng.uniform(-init_noise_rad, init_noise_rad)
        for name, ang in start.items():
            env.data.qpos[qadr[name]] = ang
            env.data.ctrl[ctrl_idx[name]] = ang
        env.data.qpos[2] = 0.16 + rng.uniform(-noise_z, noise_z)
        env.data.qpos[3] = 1.0   # qw (upright)
        mujoco.mj_forward(env.model, env.data)
        # Settle so feet contact the ground naturally.
        for _ in range(int(0.5 / env.dt)):
            mujoco.mj_step(env.model, env.data)
        # Re-init env's bookkeeping to match the new state.
        from reward import RewardAccumulator
        env.acc = RewardAccumulator(env.dt, fall_tilt_deg=env.fall_tilt_deg,
                                     tilt_scale=env.tilt_scale)
        env._step_count = 0
        env._last_x = float(env.data.qpos[0])
        env._prev_action = None
        env._foot_x_at_lift = {leg: None for leg in env.foot_body_ids}
        env._foot_lifted = {leg: False for leg in env.foot_body_ids}
        env._z_at_reset = float(env.data.qpos[2])
        obs = env._get_obs()

        # Roll out N cycles. Within each cycle, move from phase i to phase i+1.
        steps_per_phase = max(1, int(phase_time / (env.dt * ctrl_repeat)))
        prev_pose = poses["start"]
        ep_steps = 0
        for cycle in range(15):
            for phase_idx in range(1, len(phase_names)):
                target_pose = poses[phase_names[phase_idx]]
                for sub in range(steps_per_phase):
                    t = sub / steps_per_phase
                    cur = lerp_pose(prev_pose, target_pose, t, interp)
                    # Build action vector in JOINTS order.
                    action = np.array([cur[j] for j in JOINTS],
                                      dtype=np.float32)
                    action = np.clip(action, ACTION_LOW, ACTION_HIGH)
                    # Record (obs, action) — observation comes from
                    # the env's standard _get_obs().
                    obs_list.append(obs.copy())
                    act_list.append(action.copy())
                    obs, _, term, trunc, _ = env.step(action)
                    ep_steps += 1
                    if term or trunc:
                        break
                if term or trunc:
                    break
                prev_pose = target_pose
            # End of cycle — reset prev to "start" before the next cycle.
            prev_pose = poses["start"]
            if term or trunc:
                break
        distances.append(float(env.data.qpos[0]))
        print(f"  episode {ep:2d}: {ep_steps} steps, "
              f"final x={env.data.qpos[0]:+.2f}m")

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)
    return obs_arr, act_arr, env, np.array(distances)


def bc_train(env, obs_arr, act_arr, epochs, batch_size, lr, model_kwargs):
    """Build a PPO model, train its actor via MSE supervised learning."""
    # Wrap env in VecNormalize so the policy's input layer expects normalized
    # obs — exactly what train_ppo.py does. We seed VecNormalize's running
    # stats from the demonstration data so subsequent .learn() calls don't
    # break compat.
    dv = DummyVecEnv([lambda: env])
    dv = VecNormalize(dv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # Manually set obs_rms from demo data (so VecNormalize matches the
    # statistics the policy was trained against).
    dv.obs_rms.mean = obs_arr.mean(axis=0)
    dv.obs_rms.var = obs_arr.var(axis=0) + 1e-8
    dv.obs_rms.count = len(obs_arr)

    model = PPO("MlpPolicy", dv,
                policy_kwargs=dict(net_arch=[128, 128]),
                verbose=0, **model_kwargs)

    # Normalize obs for training (PPO's policy expects normalized inputs).
    obs_norm = (obs_arr - dv.obs_rms.mean) / np.sqrt(dv.obs_rms.var)
    obs_norm = np.clip(obs_norm, -10, 10)

    device = model.policy.device
    obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=device)
    act_t = torch.tensor(act_arr, dtype=torch.float32, device=device)
    # PPO's MlpPolicy outputs distribution over actions. For BC we want to
    # match the *mean* action. Access policy.predict via gradient-friendly
    # forward pass.

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)

    n = len(obs_t)
    print(f"\nBC training: {n} demo pairs, {epochs} epochs, batch {batch_size}")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ob = obs_t[idx]
            ac_target = act_t[idx]
            # Get the policy's predicted mean action for these obs.
            # ActorCriticPolicy.forward returns (actions, values, log_probs)
            # but the *mean* is via policy.get_distribution(ob).distribution.mean.
            dist = model.policy.get_distribution(ob)
            ac_mean = dist.distribution.mean   # for DiagGaussianDistribution
            loss = ((ac_mean - ac_target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}: "
                  f"mean MSE = {total_loss / n_batches:.5f}")

    return model, dv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gait_json")
    ap.add_argument("--out", required=True,
                    help="Output PPO .zip path (will also save _vecnormalize.pkl)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noise-deg", type=float, default=2.0)
    ap.add_argument("--noise-z", type=float, default=0.005)
    ap.add_argument("--ctrl-repeat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"=== Collecting CMA demos ({args.episodes} episodes) ===")
    obs_arr, act_arr, env, dists = collect_demos(
        args.gait_json, args.episodes, args.noise_deg, args.noise_z,
        args.ctrl_repeat, rng,
    )
    print(f"\nCollected {len(obs_arr)} (obs, action) pairs.")
    print(f"Demo distance stats: mean={dists.mean():.2f}m  "
          f"min={dists.min():.2f}  max={dists.max():.2f}")

    print(f"\n=== Behavioral cloning ===")
    model, dv = bc_train(env, obs_arr, act_arr,
                          args.epochs, args.batch_size, args.lr,
                          model_kwargs={})

    model.save(args.out)
    dv.save(args.out.replace(".zip", "_vecnormalize.pkl"))
    print(f"\nSaved {args.out}  +  _vecnormalize.pkl")
    print(f"Next step: load this in train_ppo.py for PPO fine-tuning.")


if __name__ == "__main__":
    main()
