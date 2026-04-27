#!/usr/bin/env python3
"""
Export a PPO policy as an open-loop joint-target trace for hardware replay.

Why open-loop:
  Empirically (sim-to-sim test), the trained env is fully deterministic —
  identical actions are produced every episode. The "closed loop" exists
  only in principle; in practice the policy's behavior is fixed once the
  env is deterministic. So we can record one deterministic rollout and
  stream those joint targets to the LX-16A servos on the Pi without
  needing to read any state back. This matches the existing CMA deploy
  pattern (best_gait.json + replay) and avoids putting torch + sb3 on
  the Pi.

What gets saved (JSON):
  {
    "policy":  <model name>,
    "ctrl_hz":  62.5,
    "step_ms":  16,
    "n_steps":  N,
    "friction": <mu used during trace gen>,
    "joint_order": ["FL_hip","FL_knee","FR_hip","FR_knee",
                    "RL_hip","RL_knee","RR_hip","RR_knee"],
    "trace": [[hip_FL_deg, knee_FL_deg, ...]  × n_steps]   # all in degrees
  }

Joint values are in DEGREES, in the same convention `leg_abs()` expects
on the robot side (hip ∈ [5, 55], knee ∈ [-100, -25]). KNEE_TRIM should
be ZEROED at replay time — it was tuned for an old CoM and would tilt
the policy's pose.

Usage:
  python3 export_ppo_trace.py --policy models/ablations/ppo_v30_bodysmooth_dr.zip
  python3 export_ppo_trace.py --policy ... --friction 0.4 --out trace.json
  python3 export_ppo_trace.py --policy ... --max-steps 1000   # shorter trace
"""
import argparse
import json
import os

import numpy as np

from gym_env import OptimusPrimalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: derive from --policy)")
    ap.add_argument("--friction", type=float, default=0.5,
                    help="Floor mu during trace generation. Pick the value "
                         "you expect on the deploy surface (carpet ~0.4, "
                         "rubber mat ~0.7). Default 0.5.")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--fall-tilt", type=float, default=30.0)
    ap.add_argument("--ctrl-repeat", type=int, default=8)
    args = ap.parse_args()

    out_path = args.out or (
        os.path.splitext(args.policy)[0] + f"_trace_mu{args.friction}.json"
    )

    env = OptimusPrimalEnv(max_steps=args.max_steps,
                           fall_tilt_deg=args.fall_tilt,
                           ctrl_repeat=args.ctrl_repeat,
                           friction_range=(args.friction, args.friction))
    vn_path = args.policy.replace(".zip", "_vecnormalize.pkl")
    dv = VecNormalize.load(vn_path, DummyVecEnv([lambda: env]))
    dv.training = False
    dv.norm_reward = False
    model = PPO.load(args.policy, env=dv)

    obs, _ = env.reset()
    actions = []
    for i in range(args.max_steps):
        on = dv.normalize_obs(obs.reshape(1, -1))[0]
        a, _ = model.predict(on, deterministic=True)
        actions.append(a.copy())
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    n = len(actions)
    final_x = float(info["x"])
    fell = bool(info["fell"])

    # Convert to degrees in the JOINT order used by gym_env
    # ("hip_fl, knee_fl, hip_fr, knee_fr, hip_rl, knee_rl, hip_rr, knee_rr").
    trace_deg = [list(map(float, np.degrees(a))) for a in actions]

    step_ms = env.dt * env.ctrl_repeat * 1000.0
    out = {
        "policy": os.path.basename(args.policy).replace(".zip", ""),
        "ctrl_hz": float(1.0 / (env.dt * env.ctrl_repeat)),
        "step_ms": float(step_ms),
        "n_steps": n,
        "friction": args.friction,
        "joint_order": ["FL_hip", "FL_knee", "FR_hip", "FR_knee",
                        "RL_hip", "RL_knee", "RR_hip", "RR_knee"],
        "trace": trace_deg,
        "sim_final_x_m": final_x,
        "sim_fell": fell,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Exported {n} steps ({n*step_ms/1000:.1f}s of motion) to {out_path}")
    print(f"  sim final x: {final_x:+.2f} m  fell={fell}")
    print(f"  step interval: {step_ms:.1f} ms ({1000/step_ms:.1f} Hz)")
    print(f"  initial pose (deg): "
          f"{[f'{v:+.1f}' for v in trace_deg[0]]}")


if __name__ == "__main__":
    main()
