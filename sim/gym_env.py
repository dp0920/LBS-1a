"""
Gymnasium environment for the Optimus Primal quadruped.

Wraps `sim_core` (MuJoCo model) + `reward` (reward primitives) as a
standard `gym.Env`. The policy sees joint positions + body orientation +
velocities and outputs target joint angles directly — no phase concept,
no hand-coded gait structure.

Compatible with Stable-Baselines3 PPO via `SubprocVecEnv`.

Status: skeleton. Not trained yet. See TODO notes for what still needs
implementing / tuning before first training run.
"""
import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from sim_core import (JOINTS, build_model, get_qadr, get_ctrl_idx,
                      get_base_body_id, body_rp, body_fwd_z)
from reward import RewardAccumulator


# Action-space bounds (radians, matching URDF joint limits for hip/knee).
# Hip: [5°, 55°] → ~[0.087, 0.960] rad
# Knee: [-100°, -25°] → ~[-1.745, -0.436] rad
ACTION_LOW = np.array([
    np.radians(5), np.radians(-100),    # hip_fl, knee_fl
    np.radians(5), np.radians(-100),    # hip_fr, knee_fr
    np.radians(5), np.radians(-100),    # hip_rl, knee_rl
    np.radians(5), np.radians(-100),    # hip_rr, knee_rr
], dtype=np.float32)
ACTION_HIGH = np.array([
    np.radians(55), np.radians(-25),
    np.radians(55), np.radians(-25),
    np.radians(55), np.radians(-25),
    np.radians(55), np.radians(-25),
], dtype=np.float32)

# Midpoint + half-range used by the action-extension bonus. Normalizing by
# half-range means "distance from midpoint" is in [0, 1] per joint regardless
# of whether it's a hip (range 50°) or knee (range 75°).
ACTION_MIDPOINT = 0.5 * (ACTION_LOW + ACTION_HIGH)
ACTION_HALF_RANGE = 0.5 * (ACTION_HIGH - ACTION_LOW)


LEG_ORDER = ["FR", "RL", "FL", "RR"]   # crawl gait stride order
# The URDF defines `foot_*` links, but MuJoCo's URDF loader merges fixed-
# joint children into their parent, so the foot sphere geom ends up attached
# to `lower_link_*`. That's the body the contact solver sees — use those
# names for body lookups. (The earlier `foot_*` names returned -1 from
# mj_name2id, silently disabling gait/stride/weight_transfer rewards from
# v3 through v18.)
FOOT_BODY_NAMES = {
    "FL": "lower_link_fl", "FR": "lower_link_fr",
    "RL": "lower_link_rl", "RR": "lower_link_rr",
}
# Contact-force threshold (Newtons). A foot is considered "planted" if the
# z-component of its ground contact force exceeds this; "lifted" otherwise.
# Body mass 1.292 kg → total weight 12.67 N → per-foot fair share ≈ 3.17 N
# when balanced. 0.5 N (~15% of fair share) is well above sensor noise but
# well below normal stance loading, so genuine lifts are detected cleanly.
#
# Earlier versions used world-frame foot-body z thresholds here (FOOT_LIFT_Z
# = 0.015, FOOT_PLANT_TOLERANCE = 0.03). That was a bug: the foot body's
# world-z sits around 0.1 m (the body frame is at leg-tip anatomical height,
# not at ground contact), so the thresholds never fired — gait_reward was
# effectively a constant −0.5 and stride_reward was always 0 for v3+
# through v18. Switched to contact-force detection on 2026-04-24.
FOOT_CONTACT_THRESHOLD = 0.5


class OptimusPrimalEnv(gym.Env):
    """
    Observation (24-dim):
        [0:8]   joint angles  (radians)
        [8:16]  joint velocities
        [16]    body-z height
        [17]    roll
        [18]    pitch
        [19]    body-frame forward velocity (qvel[0])
        [20]    body-frame pitch rate (qvel[4])
        [21]    body_fwd_z (nose-up/down proxy)
        [22]    sin(2π · phase)   ← gait phase clock
        [23]    cos(2π · phase)

    Action (8-dim):
        target joint angles (radians). Clipped to the bounds above.

    Reward:
        Dense per-step reward from `RewardAccumulator.step_reward(dx)`,
        alive bonus, posture + smoothness penalties, plus a
        gait-contact-pattern reward that encourages the FR → RL → FL → RR
        crawl cycle (the target swing leg in each quarter of the cycle is
        rewarded for being up while the other three stay down).

    Episode: up to `max_steps` physics steps (default 2000 ≈ 10s at dt=0.005).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps=2000, fall_tilt_deg=20.0, tilt_scale=1.0,
                 ctrl_repeat=8, render_mode=None,
                 phase_period=4.0, gait_reward_scale=0.25,
                 velocity_bonus=5.0, extension_bonus=3.0,
                 velocity_shape="quadratic", stride_bonus=10.0,
                 randomize_init=False, dynamic_posture_target=False,
                 z_init_range=(0.13, 0.18), dynamic_z_tolerance=0.015,
                 weight_transfer_bonus=0.0, start_pose_json=None,
                 body_smoothness_penalty=0.0, foot_drift_penalty=0.0,
                 fall_penalty=20.0, survival_bonus=0.0,
                 friction_range=None,
                 kp=2.5, kv=0.05):
        super().__init__()
        self.model = build_model(kp=kp, kv=kv)
        self.data = mujoco.MjData(self.model)
        self.qadr = get_qadr(self.model)
        self.ctrl_idx = get_ctrl_idx(self.model)
        self.base_body_id = get_base_body_id(self.model)
        self.foot_body_ids = {
            leg: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                   FOOT_BODY_NAMES[leg])
            for leg in LEG_ORDER
        }
        self.dt = self.model.opt.timestep
        self.fall_tilt_deg = fall_tilt_deg
        self.tilt_scale = tilt_scale
        self.ctrl_repeat = ctrl_repeat     # frame-skip: repeat the same
                                            # action for N physics steps
        self.max_steps = max_steps

        # Gait cycle: phase_period seconds per full FR→RL→FL→RR cycle.
        # Default 4.0s with ctrl_repeat=4 (env step = 20ms) = 200 env steps/
        # cycle, 50 steps per leg swing — long enough for the policy to pre-
        # shift weight before each commit swing.
        self.phase_period = phase_period
        self.gait_reward_scale = gait_reward_scale
        self.velocity_bonus = velocity_bonus
        self.extension_bonus = extension_bonus
        # Velocity-reward shape: how the qvel[0] bonus scales with speed.
        #   "linear"    — k·v           (constant gradient, v5 baseline)
        #   "quadratic" — k·v²          (gradient grows linearly with v)
        #   "cubic"     — k·v³          (gradient grows quadratically; nearly
        #                                zero reward at slow speed, explodes
        #                                when fast — strongest "go big" signal)
        #   "trig"      — k·(sin(v) + 1 - cos(v))  (Taylor-like combination,
        #                                monotonic, grows super-linearly)
        valid = {"linear", "quadratic", "cubic", "trig"}
        if velocity_shape not in valid:
            raise ValueError(
                f"velocity_shape={velocity_shape!r} not in {valid}")
        self.velocity_shape = velocity_shape
        # v8: stride-length bonus. On each plant event (foot transitions from
        # lifted to planted), award stride_bonus · max(0, world-frame
        # x-displacement of that foot between liftoff and plant). Encourages
        # the policy to take fewer, larger strides instead of many tiny ones.
        # Linear in displacement, so a 10 cm stride is worth 10× a 1 cm stride.
        self.stride_bonus = stride_bonus
        self._foot_x_at_lift = {leg: None for leg in LEG_ORDER}
        self._foot_lifted = {leg: False for leg in LEG_ORDER}
        # v9: domain randomization of the starting pose. When on, each reset
        # samples an initial body-z and mildly perturbs joint angles, and the
        # posture penalty targets that per-episode starting z (tight band).
        # Teaches the policy to maintain whatever height it was placed at,
        # which is more robust for real-robot deployment.
        self.randomize_init = randomize_init
        self.dynamic_posture_target = dynamic_posture_target
        self.z_init_range = z_init_range
        self.dynamic_z_tolerance = dynamic_z_tolerance
        self._z_at_reset = 0.15    # set in reset()
        # v10: weight-transfer bonus. Rewards the target swing leg being
        # UNLOADED (low ground-contact z-force) — captures the "shift weight
        # off this leg before lifting it" pre-condition that the gait-contact
        # reward alone doesn't signal. 0 disables the term.
        self.weight_transfer_bonus = weight_transfer_bonus
        # Body weight / 4 = fair share carried by each foot at perfect balance.
        # Used to normalize the contact-force reward.
        self._fair_share_N = float(self.model.body_mass.sum() * 9.81) / 4.0
        # Optional: replace the hardcoded squatted-stance reset pose with
        # the "start" phase of a CMA-trained gait JSON. Lets RL training
        # begin from a verified-stable asymmetric pose instead of the
        # symmetric default.
        self._start_pose = None
        if start_pose_json is not None:
            import json as _json
            from mujoco_gait import decode_params as _decode_params
            with open(start_pose_json) as _f:
                _d = _json.load(_f)
            _poses, _ = _decode_params(np.array(_d["params"]))
            self._start_pose = dict(_poses["start"])

        # v26 difference-based reward terms (default 0 = disabled):
        # - body_smoothness_penalty: penalty on |Δroll| + |Δpitch| step-to-step.
        #   Targets "rocking before stride" failure mode where the policy
        #   builds up body sway to launch a leg, oscillating rather than
        #   smoothly walking.
        # - foot_drift_penalty: penalty on Σ |Δfoot_y| step-to-step across
        #   all 4 feet. Targets "waddly" gaits where feet drift laterally
        #   instead of staying under the body.
        self.body_smoothness_penalty = body_smoothness_penalty
        self.foot_drift_penalty = foot_drift_penalty
        self._prev_roll = 0.0
        self._prev_pitch = 0.0
        self._prev_foot_y = {leg: 0.0 for leg in LEG_ORDER}

        # v27 survival-priority knobs (default keeps prior behavior):
        # - fall_penalty: one-shot magnitude on episode-ending fall.
        #   Default 20.0 was the long-standing v3+ value. Bump to 200+ for
        #   "falling is catastrophic" experiments.
        # - survival_bonus: one-shot bonus paid only if episode reaches
        #   max_steps without a fall. Default 0 = disabled. Try 500+ to
        #   directly reward full-episode survival.
        self.fall_penalty = fall_penalty
        self.survival_bonus = survival_bonus

        # v30 — friction randomization for sim-to-real robustness.
        # When set to (lo, hi), each reset() samples the floor's
        # tangential friction uniformly from [lo, hi]. Default 1.0 is
        # MuJoCo's "grippy hard surface" — real-world TPU-on-floor is
        # closer to 0.3-0.6 once dust + surface variation are factored in.
        # Training across [0.3, 1.2] forces the policy to learn gaits
        # that don't rely on perfect grip.
        self.friction_range = friction_range
        # Cache the floor geom id so we can update its friction on reset.
        try:
            self._floor_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        except Exception:
            self._floor_geom_id = -1

        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        self.render_mode = render_mode
        self._viewer = None
        self._step_count = 0
        self._last_x = 0.0
        self._prev_action = None
        self.acc = None

        # Posture shaping: reward body-z inside this band (squatted stance).
        # Outside the band, penalty grows linearly. Wider tolerance than v2
        # gives early-training PPO room to explore without immediately
        # accumulating posture penalties.
        self._z_target = 0.15
        self._z_tolerance = 0.03    # band = [0.12, 0.18]

    def _get_obs(self):
        q = np.array([self.data.qpos[self.qadr[j]] for j in JOINTS],
                     dtype=np.float32)
        qd = np.zeros(len(JOINTS), dtype=np.float32)
        for i, j in enumerate(JOINTS):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            qd[i] = self.data.qvel[self.model.jnt_dofadr[jid]]

        roll, pitch = body_rp(self.data)
        phase = self._phase()
        obs = np.concatenate([
            q, qd,
            [float(self.data.qpos[2]),       # body z
             float(roll),
             float(pitch),
             float(self.data.qvel[0]),        # world-x velocity
             float(self.data.qvel[4]),        # pitch rate
             float(body_fwd_z(self.data, self.base_body_id)),
             float(np.sin(2 * np.pi * phase)),
             float(np.cos(2 * np.pi * phase))],
        ]).astype(np.float32)
        return obs

    def _foot_contact_forces(self):
        """Return a dict mapping leg → summed normal contact-force magnitude
        (Newtons) on that foot body this physics step.

        Iterates MuJoCo's active-contact list, calls mj_contactForce to get
        each contact's 6-D wrench in its contact frame (the first component
        is the normal force). Sums contributions per foot body.

        Note: `data.cfrc_ext` does NOT contain ground contact forces — it's
        only populated by user-applied `mj_applyFT` calls. That was the root
        cause of the silent-zero bug that made gait_reward constant and
        stride/weight_transfer rewards effectively disabled from v8 through
        v18.
        """
        forces = {leg: 0.0 for leg in LEG_ORDER}
        cf = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            for leg, foot_id in self.foot_body_ids.items():
                if body1 == foot_id or body2 == foot_id:
                    mujoco.mj_contactForce(self.model, self.data, i, cf)
                    forces[leg] += abs(float(cf[0]))
                    break
        return forces

    def _phase(self):
        """Current gait phase in [0, 1). 0.0-0.25 = FR swing, etc."""
        t = self._step_count * self.dt * self.ctrl_repeat
        return (t % self.phase_period) / self.phase_period

    def _target_swing_leg(self):
        """Which leg is supposed to be in swing right now."""
        phase = self._phase()
        return LEG_ORDER[int(phase * len(LEG_ORDER)) % len(LEG_ORDER)]

    def _gait_reward(self, foot_forces):
        """Per-step reward for matching the expected contact pattern.

        A foot is "planted" when its ground contact force exceeds
        FOOT_CONTACT_THRESHOLD, "lifted" otherwise.

        Target leg should be LIFTED: +1 if airborne, -1 if planted.
        Non-target legs should be PLANTED: +1 if planted, -1 if airborne.
        Max +4 / min -4 per step, scaled by gait_reward_scale.

        `foot_forces` is the dict returned by `_foot_contact_forces()`.
        """
        target = self._target_swing_leg()
        score = 0.0
        for leg in LEG_ORDER:
            planted = foot_forces[leg] > FOOT_CONTACT_THRESHOLD
            if leg == target:
                score += 1.0 if not planted else -1.0
            else:
                score += 1.0 if planted else -1.0
        return self.gait_reward_scale * score

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # v30 — randomize floor friction for this episode if configured.
        if self.friction_range is not None and self._floor_geom_id >= 0:
            lo, hi = self.friction_range
            mu = float(self.np_random.uniform(lo, hi))
            # geom_friction is [tangential, torsional, rolling]
            self.model.geom_friction[self._floor_geom_id, 0] = mu
        # Start pose: by default, mid-range hip/knee — policy learns stance
        # from here. If a CMA gait was passed via start_pose_json, use its
        # "start" phase pose instead (verified-stable asymmetric stance).
        if self._start_pose is not None:
            default_q = dict(self._start_pose)
        else:
            default_q = {
                "hip_fl": np.radians(35), "knee_fl": np.radians(-80),
                "hip_fr": np.radians(35), "knee_fr": np.radians(-80),
                "hip_rl": np.radians(35), "knee_rl": np.radians(-50),
                "hip_rr": np.radians(35), "knee_rr": np.radians(-50),
            }
        if self.randomize_init:
            # ±5° jitter on every hip/knee joint so the policy sees a spread
            # of initial stances each episode (domain randomization).
            for name in default_q:
                default_q[name] += self.np_random.uniform(
                    -np.radians(5), np.radians(5))
            # Sample initial body-z from the configured range.
            z0 = float(self.np_random.uniform(*self.z_init_range))
        else:
            z0 = 0.16
        for name, angle in default_q.items():
            self.data.qpos[self.qadr[name]] = angle
            self.data.ctrl[self.ctrl_idx[name]] = angle
        self.data.qpos[2] = z0
        self.data.qpos[3] = 1.0   # qw
        self._z_at_reset = z0
        mujoco.mj_forward(self.model, self.data)
        # Brief settle so the robot isn't freefalling on step 0.
        for _ in range(int(0.5 / self.dt)):
            mujoco.mj_step(self.model, self.data)

        self.acc = RewardAccumulator(
            self.dt,
            fall_tilt_deg=self.fall_tilt_deg,
            tilt_scale=self.tilt_scale)
        self._step_count = 0
        self._last_x = float(self.data.qpos[0])
        self._prev_action = None
        # Reset per-foot stride-tracking state. All feet start planted.
        self._foot_x_at_lift = {leg: None for leg in LEG_ORDER}
        self._foot_lifted = {leg: False for leg in LEG_ORDER}
        # Reset v26 difference-tracking state.
        roll0, pitch0 = body_rp(self.data)
        self._prev_roll = float(roll0)
        self._prev_pitch = float(pitch0)
        for leg in LEG_ORDER:
            self._prev_foot_y[leg] = float(
                self.data.xpos[self.foot_body_ids[leg]][1])
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, ACTION_LOW, ACTION_HIGH)
        for i, name in enumerate(JOINTS):
            self.data.ctrl[self.ctrl_idx[name]] = float(action[i])

        dx_total = 0.0
        fell = False
        for _ in range(self.ctrl_repeat):
            mujoco.mj_step(self.model, self.data)
            self.acc.sample_step(self.data, self.base_body_id)
            x_now = float(self.data.qpos[0])
            dx_total += x_now - self._last_x
            self._last_x = x_now
            if self.acc.is_fallen(self.data, self.base_body_id):
                fell = True
                break

        self._step_count += 1
        reward = self.acc.step_reward(dx_total)

        # Posture: penalty grows linearly outside an allowed z-band. By
        # default the band is fixed (z_target ± z_tolerance). With
        # dynamic_posture_target, the band is centered on whatever z the
        # episode *started* at, with a tighter tolerance — so the policy
        # learns to maintain its initial height, not a hardcoded one.
        z = float(self.data.qpos[2])
        if self.dynamic_posture_target:
            z_dev = max(0.0, abs(z - self._z_at_reset)
                        - self.dynamic_z_tolerance)
        else:
            z_dev = max(0.0, abs(z - self._z_target) - self._z_tolerance)
        posture_penalty = 10.0 * z_dev

        # Action smoothness: penalize jerky motor commands step-to-step so
        # the policy doesn't produce high-frequency wiggles that are
        # unrealistic on real servos.
        if self._prev_action is not None:
            da = action - self._prev_action
            smoothness_penalty = 0.1 * float(np.dot(da, da))
        else:
            smoothness_penalty = 0.0
        self._prev_action = action.copy()

        # v26 — body-angle smoothness penalty: |Δroll| + |Δpitch| step-to-step.
        # Discourages "rocking" gaits where the policy builds up body sway
        # before each stride.
        roll_now, pitch_now = body_rp(self.data)
        if self.body_smoothness_penalty > 0.0:
            d_roll = abs(float(roll_now) - self._prev_roll)
            d_pitch = abs(float(pitch_now) - self._prev_pitch)
            body_smooth_pen = self.body_smoothness_penalty * (d_roll + d_pitch)
        else:
            body_smooth_pen = 0.0
        self._prev_roll = float(roll_now)
        self._prev_pitch = float(pitch_now)

        # Compute ground-contact forces for all feet once (used by gait,
        # weight-transfer, and stride-detection logic below).
        foot_forces = self._foot_contact_forces()

        # v26 — foot lateral drift penalty: Σ |Δfoot_y| across all 4 feet.
        # Targets "waddly" gaits where feet drift sideways instead of
        # staying under the body. World-frame foot_y deltas catch lateral
        # motion that contributes nothing to forward locomotion.
        if self.foot_drift_penalty > 0.0:
            drift = 0.0
            for leg in LEG_ORDER:
                fy = float(self.data.xpos[self.foot_body_ids[leg]][1])
                drift += abs(fy - self._prev_foot_y[leg])
                self._prev_foot_y[leg] = fy
            foot_drift_pen = self.foot_drift_penalty * drift
        else:
            foot_drift_pen = 0.0
            # Still update tracking state so toggling this on doesn't
            # produce a huge first-step delta.
            for leg in LEG_ORDER:
                self._prev_foot_y[leg] = float(
                    self.data.xpos[self.foot_body_ids[leg]][1])

        # Gait-pattern reward: encourages the FR → RL → FL → RR crawl cycle.
        # This fires regardless of fall so the policy gets signal early.
        gait_reward = self._gait_reward(foot_forces)

        # Weight-transfer bonus (v10): reward low ground-contact force on
        # the target swing leg. Captures the "shift weight off this leg
        # before lifting" pre-stride behavior that real quadrupeds use.
        # max(0, 1 - fz/fair_share): 1 when fully unloaded, 0 at fair share,
        # clipped so there's no penalty for carrying extra weight.
        if self.weight_transfer_bonus > 0.0:
            target_leg = self._target_swing_leg()
            fz = foot_forces[target_leg]
            unload_frac = max(0.0, 1.0 - fz / self._fair_share_N)
            weight_transfer_reward = self.weight_transfer_bonus * unload_frac
        else:
            weight_transfer_reward = 0.0

        # Stride-length bonus: on each plant event (foot transitions lifted
        # → planted), reward the world-frame forward displacement of that
        # foot between liftoff and plant. Encourages committing to larger
        # strides rather than taking many tiny high-cadence steps.
        stride_reward = 0.0
        for leg in LEG_ORDER:
            foot_id = self.foot_body_ids[leg]
            foot_x = float(self.data.xpos[foot_id][0])
            currently_lifted = foot_forces[leg] < FOOT_CONTACT_THRESHOLD
            was_lifted = self._foot_lifted[leg]
            if not was_lifted and currently_lifted:
                # lift event — anchor the stride's starting x
                self._foot_x_at_lift[leg] = foot_x
            elif was_lifted and not currently_lifted:
                # plant event — reward forward displacement since liftoff
                if self._foot_x_at_lift[leg] is not None:
                    disp = foot_x - self._foot_x_at_lift[leg]
                    stride_reward += self.stride_bonus * max(0.0, disp)
                self._foot_x_at_lift[leg] = None
            self._foot_lifted[leg] = currently_lifted

        # Speed bonus — shape controls how strongly fast is rewarded.
        # See self.velocity_shape docstring for the options and their gradients.
        fwd_vel = max(0.0, float(self.data.qvel[0]))
        if self.velocity_shape == "linear":
            v_shape = fwd_vel
        elif self.velocity_shape == "quadratic":
            v_shape = fwd_vel * fwd_vel
        elif self.velocity_shape == "cubic":
            v_shape = fwd_vel * fwd_vel * fwd_vel
        else:  # "trig"
            v_shape = np.sin(fwd_vel) + 1.0 - np.cos(fwd_vel)
        vel_reward = self.velocity_bonus * float(v_shape)

        # Action extension bonus: reward commanded joint angles that sit far
        # from the midpoint of their bound. Normalized per-joint so hip
        # (range 50°) and knee (range 75°) contribute equally. Value ∈ [0, 1]
        # per joint: 0 = dead center, 1 = commanded at a bound. Counters the
        # "jitter at stance" local optimum by explicitly paying the policy
        # to commit to full-stretch leg poses (either flexed or extended).
        a_extension = float(np.mean(
            np.abs((action - ACTION_MIDPOINT) / ACTION_HALF_RANGE)))
        extension_reward = self.extension_bonus * a_extension

        # Alive bonus tuned so stable-standing ≈ 0 per step (slightly positive).
        # Falling ends the episode AND loses future +alive, so sustained
        # walking easily beats any "fall fast" strategy.
        if not fell:
            reward += (2.0 + gait_reward + vel_reward + extension_reward
                       + stride_reward + weight_transfer_reward
                       - posture_penalty - smoothness_penalty
                       - body_smooth_pen - foot_drift_pen)
        else:
            reward -= self.fall_penalty    # one-shot fall penalty (v27 tunable)

        # v27 — survival completion bonus paid once at episode end if
        # the agent walked the full max_steps without falling. Directly
        # rewards "got through a full rollout" rather than "walked far".
        truncated_full_episode = (not fell
                                   and self._step_count >= self.max_steps)
        if truncated_full_episode and self.survival_bonus > 0.0:
            reward += self.survival_bonus

        terminated = fell
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        # Reward-component breakdown for post-hoc correlation / PCA analysis.
        # Every term is reported as its *contribution to the scalar reward*
        # at this step (penalties are negative). See reward_pca.py for
        # how this matrix is consumed.
        info = {
            "dx": dx_total,
            "fell": fell,
            "x": self._last_x,
            "reward_components": {
                "alive": 2.0 if not fell else 0.0,
                "step_reward": self.acc.step_reward(dx_total),
                "gait": gait_reward if not fell else 0.0,
                "velocity": vel_reward if not fell else 0.0,
                "extension": extension_reward if not fell else 0.0,
                "stride": stride_reward if not fell else 0.0,
                "weight_transfer": weight_transfer_reward if not fell else 0.0,
                "posture": -posture_penalty if not fell else 0.0,
                "smoothness": -smoothness_penalty if not fell else 0.0,
                "body_smooth": -body_smooth_pen if not fell else 0.0,
                "foot_drift": -foot_drift_pen if not fell else 0.0,
                "fall": -self.fall_penalty if fell else 0.0,
                "survival_bonus": (self.survival_bonus
                                    if truncated_full_episode else 0.0),
            },
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# TODO before first PPO training run:
# - Tune step_reward coefficients — the CMA formula assumes cumulative;
#   dense rewards may need rescaling so learning signal doesn't drown in noise.
# - Decide on ctrl_repeat (frame-skip): 4 ≈ 50Hz control. Might want 2 or 8.
# - Add observation normalization (SB3's VecNormalize wraps this cleanly).
# - Consider seeding the starting pose from decode_params(X0)["start"]
#   for parity with CMA rollouts.
# - Behavioral cloning from best CMA gait could bootstrap the policy before PPO.

if __name__ == "__main__":
    # Smoke test: create env, step it randomly, make sure shapes are right.
    env = OptimusPrimalEnv()
    obs, _ = env.reset()
    print(f"obs shape: {obs.shape}  action shape: {env.action_space.shape}")
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"r={r:+.3f}  x={info['x']:+.3f}  fell={info['fell']}")
        if term or trunc:
            break
    env.close()
