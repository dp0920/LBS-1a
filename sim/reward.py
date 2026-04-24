"""
Reward shaping and fall detection for the Optimus Primal quadruped.

The `RewardAccumulator` samples body attitude on every physics step, enforces
hard termination when the robot flops/falls, and produces the final scalar
reward + info dict at the end of a rollout.

Design:
  - CMA-ES / sweep rollouts call `finalize()` once at the end to get the
    cumulative reward the optimizer minimizes.
  - RL (gym env) uses `step_reward()` per physics step for dense feedback,
    and `is_fallen()` for episode termination. `finalize()` is still available
    if the env wants a summary at episode end.

The coefficients here encode the "v4" reward: forward distance + speed
bonus + stability/flop/pitch-rate penalties (see METHODS.md §4).
"""
import numpy as np

from sim_core import body_rp, body_fwd_z


class RewardAccumulator:
    """Per-rollout attitude tracker + reward calculator.

    Usage (CMA-ES style):
        acc = RewardAccumulator(dt, fall_tilt_deg=20.0, tilt_scale=1.0)
        for step in rollout:
            mj_step(...)
            acc.sample_step(data, base_body_id)
            if acc.is_fallen(data, base_body_id):
                break
        reward, info = acc.finalize(earned_distance, phases_completed,
                                    total_phases, mean_z, final_z, min_z)

    Usage (RL style):
        acc = RewardAccumulator(dt, ...)
        # per env.step():
        acc.sample_step(data, base_body_id)
        r_step = acc.step_reward(dx)
        done = acc.is_fallen(data, base_body_id)
    """

    FALL_Z = 0.08   # body-z at/below this → fallen

    def __init__(self, dt, fall_tilt_deg=20.0, tilt_scale=1.0):
        self.dt = dt
        self.fall_tilt_rad = np.radians(max(0.1, fall_tilt_deg))
        # nose-down kill threshold scales with tilt budget.
        # sin(tilt_deg) ≈ world-z projection cutoff for body-x axis.
        self.fall_nose_down = float(np.sin(self.fall_tilt_rad))
        self.tilt_scale = tilt_scale

        # Per-step accumulators (body frame).
        self._pitch_sq_sum = 0.0
        self._roll_sq_sum = 0.0
        self._pitch_abs_sum = 0.0
        self._roll_abs_sum = 0.0
        self._nose_down_sum = 0.0
        self._nose_down_sq_sum = 0.0
        self._pitch_rate_sq_sum = 0.0
        self._max_pitch = 0.0
        self._max_roll = 0.0
        self._max_nose_down = 0.0
        self._max_pitch_rate = 0.0
        self._n_steps = 0

    # --- Fall detection --------------------------------------------------

    def is_fallen(self, data, base_body_id):
        """True if the body has fallen below `FALL_Z`, tilted past the
        configured threshold, or pitched nose-down past the kill cutoff."""
        if data.qpos[2] < self.FALL_Z:
            return True
        roll, pitch = body_rp(data)
        if abs(roll) > self.fall_tilt_rad or abs(pitch) > self.fall_tilt_rad:
            return True
        if -body_fwd_z(data, base_body_id) > self.fall_nose_down:
            return True
        return False

    # --- Per-step sampling ----------------------------------------------

    def sample_step(self, data, base_body_id):
        """Accumulate attitude metrics for this physics step. Must be called
        AFTER `mj_step(model, data)` on every step you want in the average."""
        roll, pitch = body_rp(data)
        self._pitch_sq_sum += pitch * pitch
        self._roll_sq_sum += roll * roll
        self._pitch_abs_sum += abs(pitch)
        self._roll_abs_sum += abs(roll)
        if abs(pitch) > self._max_pitch:
            self._max_pitch = abs(pitch)
        if abs(roll) > self._max_roll:
            self._max_roll = abs(roll)

        nd = max(0.0, -body_fwd_z(data, base_body_id))
        self._nose_down_sum += nd
        self._nose_down_sq_sum += nd * nd
        if nd > self._max_nose_down:
            self._max_nose_down = nd

        pr = float(data.qvel[4])    # body-frame pitch-rate
        self._pitch_rate_sq_sum += pr * pr
        if abs(pr) > self._max_pitch_rate:
            self._max_pitch_rate = abs(pr)

        self._n_steps += 1

    # --- RL: dense per-step reward --------------------------------------

    def step_reward(self, dx):
        """Instantaneous reward for one physics step. Used by the RL env.

        dx: forward displacement (meters) since the last step.
        Returns a scalar ~ per-step contribution to the episode reward.

        The formula is intentionally compatible with `finalize()`: summing
        step_reward over the whole rollout yields the forward + speed terms
        of `finalize()`, modulo the constant step_bonus and height_bonus.
        """
        # Forward progress (100 per meter, matching `finalize()`).
        fwd = 100.0 * max(0.0, dx) - 200.0 * max(0.0, -dx)

        # Per-step speed bonus: (dx/dt) × 1000 × dt = 1000 × dx, but we only
        # want to pay it when moving forward, and averaged — so scale to an
        # instantaneous m/s-equivalent bump.
        speed_bonus = 1000.0 * max(0.0, dx)

        # Stability penalties derived from the most recent sample.
        roll = self._roll_sq_sum_last() if self._n_steps else 0.0
        pitch = self._pitch_sq_sum_last() if self._n_steps else 0.0
        nd2 = self._nd_sq_last() if self._n_steps else 0.0
        pr2 = self._pr_sq_last() if self._n_steps else 0.0

        stab = self.tilt_scale * (
            60.0 * pitch + 80.0 * roll
            + 800.0 * nd2
            + 8.0 * pr2
        )
        return fwd + speed_bonus - stab

    # Helpers to get the last-sampled squared values (before accumulation is
    # folded into the running sum; we can't recover that, so just use the
    # running mean — close enough for dense RL rewards).
    def _pitch_sq_sum_last(self):
        return self._pitch_sq_sum / self._n_steps

    def _roll_sq_sum_last(self):
        return self._roll_sq_sum / self._n_steps

    def _nd_sq_last(self):
        return self._nose_down_sq_sum / self._n_steps

    def _pr_sq_last(self):
        return self._pitch_rate_sq_sum / self._n_steps

    # --- Finalize --------------------------------------------------------

    def finalize(self, earned_distance, phases_completed, total_phases,
                 mean_z=0.0, final_z=0.0, min_z=0.0):
        """Compute the end-of-rollout reward and info dict.
        This is the 'v4' reward from METHODS.md §4.
        """
        n = max(1, self._n_steps)
        mean_pitch_sq = self._pitch_sq_sum / n
        mean_roll_sq = self._roll_sq_sum / n
        mean_pitch_abs = self._pitch_abs_sum / n
        mean_roll_abs = self._roll_abs_sum / n
        mean_nose_down = self._nose_down_sum / n
        mean_nose_down_sq = self._nose_down_sq_sum / n
        mean_pitch_rate_sq = self._pitch_rate_sq_sum / n

        elapsed_time = max(self.dt, self._n_steps * self.dt)
        speed = max(0.0, earned_distance) / elapsed_time

        step_bonus = 0.5 * phases_completed
        forward = max(0.0, earned_distance) * 100.0 + speed * 1000.0
        backward_penalty = max(0.0, -earned_distance) * 200.0
        height_bonus = 5.0 * mean_z if phases_completed > 0 else 0.0

        pitch_penalty = self.tilt_scale * 60.0 * mean_pitch_sq
        roll_penalty = self.tilt_scale * 80.0 * mean_roll_sq
        peak_penalty = self.tilt_scale * (16.0 * self._max_pitch ** 2
                                          + 16.0 * self._max_roll ** 2)
        flop_penalty = self.tilt_scale * (800.0 * mean_nose_down_sq
                                          + 250.0 * (self._max_nose_down ** 2))
        pitch_rate_penalty = self.tilt_scale * (
            8.0 * mean_pitch_rate_sq + 3.0 * (self._max_pitch_rate ** 2))

        reward = (step_bonus + forward - backward_penalty + height_bonus
                  - pitch_penalty - roll_penalty - peak_penalty
                  - flop_penalty - pitch_rate_penalty)

        survival = phases_completed / max(1, total_phases)
        info = {
            "earned_dist_m": earned_distance,
            "elapsed_s": elapsed_time,
            "speed_mps": speed,
            "mean_pitch_deg": np.degrees(mean_pitch_abs),
            "mean_roll_deg": np.degrees(mean_roll_abs),
            "max_pitch_deg": np.degrees(self._max_pitch),
            "max_roll_deg": np.degrees(self._max_roll),
            "mean_nose_down_deg": np.degrees(
                np.arcsin(min(1.0, mean_nose_down))),
            "max_nose_down_deg": np.degrees(
                np.arcsin(min(1.0, self._max_nose_down))),
            "max_pitch_rate": self._max_pitch_rate,
            "mean_z": mean_z,
            "final_z": final_z,
            "min_z": min_z,
            "phases": f"{phases_completed}/{total_phases}",
            "survival": f"{survival:.0%}",
        }
        return reward, info
