#!/usr/bin/env python3
"""
50-episode survival/distance evaluation for a CMA-trained gait.

CMA gaits are deterministic (open-loop replay of `best_gait.json`'s 13-phase
trajectory), so a vanilla rollout always produces the same distance. To
compare to RL policies on the same "stochastic 50-episode" axis, this
evaluator adds small initial-pose noise to each rollout — the chaotic
divergence makes each playback slightly different and gives a meaningful
distribution.

Usage:
  python cma_stats.py best_gait.json
  python cma_stats.py results/<ts>/cma_stand/linear/t29.85_s15_trial11/1000/best_gait_stand.json --episodes 50
  python cma_stats.py PATH --cycles 20            # longer rollout
  python cma_stats.py PATH --noise-deg 2.0        # ±2° joint init noise
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

from sim_core import build_model, JOINTS, get_qadr, get_ctrl_idx, body_rp, body_fwd_z
from mujoco_gait import PHASE_ORDER, decode_params, lerp_pose


def play_one(model, params, n_cycles, interp, init_noise_rad, init_z_noise,
             fall_tilt_rad, fall_z, np_random):
    """One rollout. Returns (distance, time, fell, n_phases_completed)."""
    poses, phase_time = decode_params(params)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qadr = get_qadr(model)
    ctrl_idx = get_ctrl_idx(model)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     "base_link")
    dt = model.opt.timestep

    # Set start pose with optional noise.
    start = dict(poses["start"])
    for name in start:
        start[name] += np_random.uniform(-init_noise_rad, init_noise_rad)
    for name, ang in start.items():
        data.qpos[qadr[name]] = ang
        data.ctrl[ctrl_idx[name]] = ang
    data.qpos[2] = 0.16 + np_random.uniform(-init_z_noise, init_z_noise)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    # Settle
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(model, data)

    fall_nose_down = float(np.sin(fall_tilt_rad))

    def is_fallen():
        if data.qpos[2] < fall_z:
            return True
        roll, pitch = body_rp(data)
        if abs(roll) > fall_tilt_rad or abs(pitch) > fall_tilt_rad:
            return True
        if -body_fwd_z(data, base_body_id) > fall_nose_down:
            return True
        return False

    def run_phase(from_pose, to_pose, duration):
        steps = max(1, int(duration / dt))
        for i in range(steps):
            t = i / steps
            cur = lerp_pose(from_pose, to_pose, t, interp)
            for name, ang in cur.items():
                data.ctrl[ctrl_idx[name]] = ang
            mujoco.mj_step(model, data)
            if is_fallen():
                return False
        for name, ang in to_pose.items():
            data.ctrl[ctrl_idx[name]] = ang
        mujoco.mj_step(model, data)
        return True

    phase_names = PHASE_ORDER[1:]
    n_phases = 0
    alive = True

    for cycle in range(n_cycles):
        prev = "start"
        for phase_name in phase_names:
            if not run_phase(poses[prev], poses[phase_name], phase_time):
                alive = False
                break
            prev = phase_name
            n_phases += 1
        if not alive:
            break
        if cycle < n_cycles - 1:
            if not run_phase(poses[phase_names[-1]], poses["start"], phase_time):
                alive = False
                break

    distance = float(data.qpos[0])
    elapsed = (n_phases + 1) * phase_time + 0.5  # +settle
    return distance, elapsed, not alive, n_phases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gait_json")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--noise-deg", type=float, default=2.0,
                    help="Initial joint-angle noise (±deg, uniform)")
    ap.add_argument("--noise-z", type=float, default=0.005,
                    help="Initial body-z noise (±m, uniform)")
    ap.add_argument("--fall-tilt", type=float, default=30.0,
                    help="Tilt threshold for fall detection (degrees), to "
                         "match ppo_stats's fall-tilt convention.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.gait_json) as f:
        d = json.load(f)
    params = np.array(d["params"])
    interp = d.get("config", {}).get("interp", "linear")
    print(f"Gait: {args.gait_json}")
    print(f"  reward (training): {d.get('reward', '?'):+.1f}")
    print(f"  interp: {interp}")
    print(f"  cycles per episode: {args.cycles}")
    print(f"  noise: ±{args.noise_deg}° joints, ±{args.noise_z*1000:.0f}mm z")

    rng = np.random.default_rng(args.seed)
    model = build_model()
    init_noise_rad = float(np.radians(args.noise_deg))

    dists, elapsed, fell_count, phases = [], [], 0, []
    for ep in range(args.episodes):
        dist, t, fell, np_done = play_one(
            model, params, args.cycles, interp,
            init_noise_rad, args.noise_z,
            float(np.radians(args.fall_tilt)),
            0.08, rng,
        )
        dists.append(dist)
        elapsed.append(t)
        phases.append(np_done)
        if fell:
            fell_count += 1
        status = "FELL" if fell else "ok"
        print(f"  ep {ep:2d}: dist={dist:+6.2f}m  t={t:5.1f}s  "
              f"phases={np_done:3d}  [{status}]")

    dists = np.array(dists)
    elapsed_arr = np.array(elapsed)
    speeds = dists / elapsed_arr
    survived = args.episodes - fell_count

    print()
    print(f"Across {args.episodes} episodes (init-pose noise):")
    print(f"  dist  min/mean/max:  {dists.min():+.2f} / {dists.mean():+.2f} / "
          f"{dists.max():+.2f} m")
    print(f"  speed min/mean/max:  {speeds.min():+.3f} / {speeds.mean():+.3f} / "
          f"{speeds.max():+.3f} m/s")
    print(f"  phases  mean/max:    {np.mean(phases):.1f} / {max(phases)}  "
          f"(of {args.cycles * (len(PHASE_ORDER) - 1)} possible)")
    print(f"  survived full eval:  {survived}/{args.episodes}  "
          f"({survived/args.episodes*100:.0f}%)")
    combined = float(dists.mean()) * (survived / args.episodes)
    print(f"  combined (mean × surv_rate): {combined:.2f}")


if __name__ == "__main__":
    main()
