#!/usr/bin/env python3
"""
Diagnostic: why the trained PPO policies don't deploy to LX-16A hardware.

Three measurements:

  (1) Bang-bang rate of the action stream — what fraction of joint
      commands sit exactly at ACTION_LOW or ACTION_HIGH.

  (2) Joint qpos stream — the trajectory the body actually follows
      under MuJoCo's PD-controlled position actuators (kp=2.5,
      kv=0.05). Smoother than the action stream, but still has
      large per-step deltas because the PD is heavily underdamped.

  (3) Open-loop replay of the qpos trace as the env's target
      sequence. If walking lives in the joint trajectory, replay
      should still walk. If walking lives in the bang-bang action
      stream interacting with the PD dynamics, replay collapses.

Headline result (v30_bodysmooth_dr, mu=0.5, deterministic):
  closed-loop trained policy:        +65.77 m
  open-loop replay of qpos trace:    +0.57 m
  fraction of actions at LOW/HIGH:   ~88%

Interpretation: the policy exploited the PD's underdamped response
as a low-pass filter, learning a control regime that has no analog
on the LX-16A trajectory generator. Neither the action stream nor
the joint-target stream is deployable open-loop.

For hardware deployment, fall back to CMA gaits (best_gait.json),
whose 0.6s phase times are slow enough that PD response doesn't
matter. PPO results stand as a sim-only methodological achievement;
the friction-DR sweep numbers are real in sim but don't transfer.

Usage:
  python3 ppo_actuator_gap.py --policy models/ablations/ppo_v30_bodysmooth_dr.zip
  python3 ppo_actuator_gap.py --policy ... --friction 0.5
"""
import argparse

import numpy as np

from gym_env import OptimusPrimalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


JOINTS = ["hip_fl", "knee_fl", "hip_fr", "knee_fr",
          "hip_rl", "knee_rl", "hip_rr", "knee_rr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=2000)
    args = ap.parse_args()

    # Closed-loop rollout: record actions and joint qpos.
    env = OptimusPrimalEnv(max_steps=args.max_steps, fall_tilt_deg=30.0,
                           ctrl_repeat=8,
                           friction_range=(args.friction, args.friction))
    vn_path = args.policy.replace(".zip", "_vecnormalize.pkl")
    dv = VecNormalize.load(vn_path, DummyVecEnv([lambda: env]))
    dv.training = False
    dv.norm_reward = False
    model = PPO.load(args.policy, env=dv)

    obs, _ = env.reset()
    actions, qpos_seq = [], []
    for i in range(args.max_steps):
        on = dv.normalize_obs(obs.reshape(1, -1))[0]
        a, _ = model.predict(on, deterministic=True)
        actions.append(a.copy())
        obs, _, term, trunc, info = env.step(a)
        qpos_seq.append(np.array([env.data.qpos[env.qadr[j]] for j in JOINTS]))
        if term or trunc:
            break
    actions = np.array(actions)
    qpos_seq = np.array(qpos_seq)
    closed_x = float(info["x"])

    print(f"=== closed-loop rollout ({args.policy}) ===")
    print(f"  steps: {len(actions)}   final_x: {closed_x:+.2f} m   "
          f"fell: {info['fell']}")

    # Smoothness stats
    a_diff = np.abs(np.diff(actions, axis=0))
    q_diff = np.abs(np.diff(qpos_seq, axis=0))
    print(f"\n  action stream  mean|Δ|/step={a_diff.mean():.4f} rad  "
          f"P95={np.percentile(a_diff, 95):.4f}  max={a_diff.max():.4f}")
    print(f"  qpos   stream  mean|Δ|/step={q_diff.mean():.4f} rad  "
          f"P95={np.percentile(q_diff, 95):.4f}  max={q_diff.max():.4f}")

    sat = ((np.isclose(actions, env.action_space.low, atol=1e-3)) |
           (np.isclose(actions, env.action_space.high, atol=1e-3)))
    print(f"  action samples at LOW or HIGH: {sat.mean() * 100:.1f}%")

    # Open-loop replay of qpos trace
    env2 = OptimusPrimalEnv(max_steps=args.max_steps, fall_tilt_deg=30.0,
                            ctrl_repeat=8,
                            friction_range=(args.friction, args.friction))
    env2.reset()
    for i in range(len(qpos_seq)):
        _, _, term, trunc, info2 = env2.step(qpos_seq[i].astype(np.float32))
        if term or trunc:
            break
    open_x = float(info2["x"])

    print(f"\n=== open-loop qpos-trace replay ===")
    print(f"  final_x: {open_x:+.2f} m   steps: {i + 1}   fell: {info2['fell']}")
    print(f"\nratio: open-loop / closed-loop = {open_x / closed_x * 100:.1f}%")
    if open_x / max(closed_x, 1e-3) < 0.10:
        print("VERDICT: walking lives in the bang-bang action stream, not the "
              "joint trajectory. Not deployable open-loop.")


if __name__ == "__main__":
    main()
