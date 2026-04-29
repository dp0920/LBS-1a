#!/usr/bin/env python3
"""
Given LX-16A step-response data captured on the Pi, find the MuJoCo
position-actuator kp/kv that produces the closest matching response.

Workflow:
  1. On the Pi: python3 motor_step_response.py
  2. SCP motor_step_response.json from Pi → sim/
  3. python3 fit_pd_from_step_response.py motor_step_response.json

For each servo's recorded trajectory, this script:
  - Builds a single-DOF MuJoCo model of the corresponding joint
  - Sweeps kp ∈ [1, 100] (log-spaced) and zeta ∈ [0.5, 2.0]
    where kv = 2 * zeta * sqrt(kp * inertia)
  - Picks the (kp, kv) with the lowest mean-squared error vs. the
    real servo trajectory
  - Reports per-servo and average (kp, kv)

Run this on the Mac (sim/ side has MuJoCo). The output recommendations
go into sim_core.py's build_model defaults.
"""
import argparse
import json
import sys

import numpy as np
import mujoco

from sim_core import build_model, JOINTS, get_qadr, get_ctrl_idx


# Map LX-16A servo IDs → URDF joint names (from gait_controller.py LEGS_CRAWL)
SERVO_TO_JOINT = {
    1: "hip_rl",  2: "hip_rr",
    3: "hip_fl",  4: "hip_fr",
    5: "knee_rl", 6: "knee_rr",
    7: "knee_fl", 8: "knee_fr",
}


def simulate_step(joint_name, kp, kv, step_rad, duration_s, dt=0.002):
    """Run the URDF + actuator with a step input on `joint_name`.

    Returns (t_ms, joint_angle_rad) sampled at the same dt as MuJoCo.
    Robot is held in air (free joint locked at neutral) so we measure
    the actuator response without body-coupling.
    """
    model = build_model(kp=kp, kv=kv)
    # Pin the chassis high in the air so it doesn't fall during the test.
    model.opt.gravity[:] = (0.0, 0.0, 0.0)
    data = mujoco.MjData(model)
    qadr = get_qadr(model)
    cidx = get_ctrl_idx(model)

    # Pre-set: joint at 0, command at 0.
    for j in JOINTS:
        data.qpos[qadr[j]] = 0.0
    mujoco.mj_forward(model, data)

    # Settle for 100 ms with control = 0
    for _ in range(50):
        mujoco.mj_step(model, data)

    # Apply step
    data.ctrl[cidx[joint_name]] = step_rad

    n_steps = int(duration_s / dt)
    t_arr = np.zeros(n_steps)
    a_arr = np.zeros(n_steps)
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        t_arr[i] = i * dt * 1000.0
        a_arr[i] = data.qpos[qadr[joint_name]]
    return t_arr, a_arr


def fit_one(servo_data):
    """Find best (kp, kv) for one servo trace."""
    sid = servo_data["servo_id"]
    if "error" in servo_data:
        return None
    joint = SERVO_TO_JOINT[sid]

    # Combine the up and down step responses, each referenced to its
    # own starting position so both are comparable to a sim run that
    # starts at angle 0.
    up = servo_data["up"]
    dn = servo_data["down"]
    a0_up = servo_data["start_deg"]
    a0_dn = servo_data["target_deg"]
    step_up_rad = np.radians(servo_data["commanded_step_deg"])
    step_dn_rad = -step_up_rad

    t_up = np.array(up["t_ms"])
    a_up_rad = np.radians(np.array(up["angle_deg"]) - a0_up)
    t_dn = np.array(dn["t_ms"])
    a_dn_rad = np.radians(np.array(dn["angle_deg"]) - a0_dn)

    # Concatenate both responses for joint fitting; sim runs each
    # direction separately and we score against both.
    t_real_ms = np.concatenate([t_up, t_dn])
    duration_s = (max(t_up[-1], t_dn[-1]) + 50.0) / 1000.0

    best = (1e9, None, None)   # (rmse, kp, kv)
    for kp in np.geomspace(0.05, 100.0, 18):
        # zeta ∈ [0.3, 3.0]: under, critical, very-over-damped
        for zeta in np.linspace(0.3, 3.0, 10):
            # Approximate inertia for a leg link as 1.0 (URDF's mass scaling
            # is implicit; we just want the ratio kp:kv that gives the right
            # response shape regardless of absolute torque scale).
            kv = 2.0 * zeta * np.sqrt(kp * 1.0)
            try:
                t_sim_u, a_sim_u = simulate_step(joint, kp, kv,
                                                  step_up_rad, duration_s)
                t_sim_d, a_sim_d = simulate_step(joint, kp, kv,
                                                  step_dn_rad, duration_s)
            except Exception:
                continue
            # Score on both up and down responses
            au_at_real = np.interp(t_up, t_sim_u, a_sim_u)
            ad_at_real = np.interp(t_dn, t_sim_d, a_sim_d)
            rmse = float(np.sqrt(0.5 * (
                np.mean((au_at_real - a_up_rad) ** 2) +
                np.mean((ad_at_real - a_dn_rad) ** 2))))
            if rmse < best[0]:
                best = (rmse, kp, kv)

    return {
        "servo_id": sid,
        "joint": joint,
        "kp": best[1],
        "kv": best[2],
        "rmse_rad": best[0],
        "rmse_deg": float(np.degrees(best[0])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_json")
    ap.add_argument("--out", default=None,
                    help="Write per-servo fits to this JSON. Default: "
                         "<data_json>_fit.json")
    args = ap.parse_args()

    with open(args.data_json) as f:
        servo_data = json.load(f)

    fits = []
    print(f"\n{'servo':>5s} {'joint':>10s} {'kp':>8s} {'kv':>8s} "
          f"{'rmse_deg':>10s}")
    print("-" * 50)
    for d in servo_data:
        fit = fit_one(d)
        if fit is None:
            print(f"{d['servo_id']:>5d}    (skipped — error in capture)")
            continue
        fits.append(fit)
        print(f"{fit['servo_id']:>5d} {fit['joint']:>10s} "
              f"{fit['kp']:>8.2f} {fit['kv']:>8.2f} "
              f"{fit['rmse_deg']:>10.2f}")

    if fits:
        kp_arr = np.array([f["kp"] for f in fits])
        kv_arr = np.array([f["kv"] for f in fits])
        print("-" * 50)
        print(f"{'mean':>5s} {'-':>10s} {kp_arr.mean():>8.2f} "
              f"{kv_arr.mean():>8.2f}")
        print(f"{'med':>5s}  {'-':>10s} {np.median(kp_arr):>8.2f} "
              f"{np.median(kv_arr):>8.2f}")
        print()
        print("Recommended sim_core.build_model defaults:")
        print(f"  kp ≈ {np.median(kp_arr):.2f}")
        print(f"  kv ≈ {np.median(kv_arr):.2f}")

    out = args.out or args.data_json.replace(".json", "_fit.json")
    with open(out, "w") as f:
        json.dump({"per_servo": fits,
                   "median_kp": float(np.median(kp_arr)) if fits else None,
                   "median_kv": float(np.median(kv_arr)) if fits else None},
                  f, indent=2)
    print(f"\nSaved fits to {out}")


if __name__ == "__main__":
    main()
