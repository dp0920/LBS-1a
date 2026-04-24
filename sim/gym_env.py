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
FOOT_BODY_NAMES = {
    "FL": "foot_fl", "FR": "foot_fr",
    "RL": "foot_rl", "RR": "foot_rr",
}
# "Lifted" threshold — target-leg foot z above this counts as in swing.
FOOT_LIFT_Z = 0.015
# Non-target feet may drift up to this height without penalty — gives the
# policy room to briefly unload a leg for rebalancing without being scored
# as "wrong leg in swing". Penalty only triggers above this.
FOOT_PLANT_TOLERANCE = 0.03


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
                 ctrl_repeat=4, render_mode=None,
                 phase_period=4.0, gait_reward_scale=0.25,
                 velocity_bonus=5.0, extension_bonus=3.0):
        super().__init__()
        self.model = build_model()
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
        # joint velocities — the free-joint takes qvel[0:6], then joints follow.
        # Map the joint qposadr → qveladr; for 1-DoF hinge joints the offset
        # from free-joint is qposadr - 7 + 6 = qposadr - 1 (approx), but safer
        # to pull by jnt_dofadr:
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

    def _phase(self):
        """Current gait phase in [0, 1). 0.0-0.25 = FR swing, etc."""
        t = self._step_count * self.dt * self.ctrl_repeat
        return (t % self.phase_period) / self.phase_period

    def _target_swing_leg(self):
        """Which leg is supposed to be in swing right now."""
        phase = self._phase()
        return LEG_ORDER[int(phase * len(LEG_ORDER)) % len(LEG_ORDER)]

    def _gait_reward(self):
        """Per-step reward for matching the expected contact pattern.

        Target leg: +1 if foot z > FOOT_LIFT_Z (in swing), -1 otherwise.
        Non-target legs: +1 if foot z < FOOT_PLANT_TOLERANCE (planted or
        barely unloading), -1 only if lifted above the tolerance.

        The gap between FOOT_LIFT_Z and FOOT_PLANT_TOLERANCE is a "free zone"
        where non-target legs can briefly lift for weight-shifting without
        penalty.
        """
        target = self._target_swing_leg()
        score = 0.0
        for leg in LEG_ORDER:
            z = self.data.xpos[self.foot_body_ids[leg]][2]
            if leg == target:
                score += 1.0 if z > FOOT_LIFT_Z else -1.0
            else:
                score += 1.0 if z < FOOT_PLANT_TOLERANCE else -1.0
        return self.gait_reward_scale * score

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Start pose: mid-range hip/knee — policy learns stance from here.
        default_q = {
            "hip_fl": np.radians(35), "knee_fl": np.radians(-80),
            "hip_fr": np.radians(35), "knee_fr": np.radians(-80),
            "hip_rl": np.radians(35), "knee_rl": np.radians(-50),
            "hip_rr": np.radians(35), "knee_rr": np.radians(-50),
        }
        for name, angle in default_q.items():
            self.data.qpos[self.qadr[name]] = angle
            self.data.ctrl[self.ctrl_idx[name]] = angle
        self.data.qpos[2] = 0.16
        self.data.qpos[3] = 1.0   # qw
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

        # Posture: penalty grows linearly outside the [z_target ± tolerance]
        # band. Prevents the policy from standing on fully-extended legs.
        z = float(self.data.qpos[2])
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

        # Gait-pattern reward: encourages the FR → RL → FL → RR crawl cycle.
        # This fires regardless of fall so the policy gets signal early.
        gait_reward = self._gait_reward()

        # Speed bonus: explicit positive signal for forward velocity on top of
        # step_reward(dx). dx is noisy per-step; a direct velocity term gives
        # a cleaner gradient toward "move faster" without the asymmetric
        # backward punishment mixed in.
        vel_reward = self.velocity_bonus * max(0.0, float(self.data.qvel[0]))

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
                       - posture_penalty - smoothness_penalty)
        else:
            reward -= 20.0    # one-shot fall penalty

        terminated = fell
        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()
        info = {"dx": dx_total, "fell": fell, "x": self._last_x}
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
