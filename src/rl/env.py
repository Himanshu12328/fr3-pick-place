"""
A Gymnasium environment wrapping the same simulator, controller and success
criterion the evaluation harness uses.

Three decisions shape this file.

**Privileged observation.** The block pose is in the observation here, which
is the opposite of the rule the datasets follow. That is deliberate and it
is confined to this environment. The policy trained on it is a teacher whose
only job is to relabel actions for a vision student in Stage 3. The student
never sees any of this. Excluding block pose from the student is what stops
it skipping perception; including it here is what makes the teacher cheap
to train, because a privileged observation needs no cameras and therefore
no rendering, which is the expensive half of the simulator.

**Delta actions, absolute targets.** The agent emits a small change to the
target pose, which the environment integrates into an absolute target pose
before handing it to the impedance controller. That mirrors exactly what
the DualSense integrator does during collection: sticks produce deltas, the
recorded action is the absolute target. The integrated absolute target is
exposed on every step as `info["absolute_action"]`, in the same 8 element
layout the dataset uses, so Stage 3 can relabel a student's states without
any conversion. A teacher that emitted actions in a different space would
need a translation layer, and translation layers are where silent
divergence lives.

**No rendering.** `need_images` is never true here. That is what buys
4,753 policy steps per second across 20 workers, measured in
`src/scripts/bench_env.py`.

The environment is validated the same way the harness was: by driving it
with the scripted oracle and checking it still solves the task. See
`src/scripts/test_rl_env.py`.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.config import GRIPPER_CLOSED, GRIPPER_OPEN
from src.data.task import PLACE_TARGET, check_success
from src.eval import rollout as R
from src.eval.oracle import GRASP_OFFSET, down_quat, quat_yaw, wrap
from src.rl import reference as REF
from src.rl import reward as RW

# Per-step limits on the target. The demonstrations saturated at 7.00 mm per
# step with both sticks at full deflection, so allowing more would let the
# teacher move in a way no demonstration ever did and no student has seen.
MAX_DELTA_M = 0.007
MAX_DYAW_RAD = np.deg2rad(5.0)

OBS_DIM = 38


class FR3PickPlaceEnv(gym.Env):
    """
    Privileged, render-free pick-and-place for training the Stage 2 teacher.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=R.MAX_POLICY_STEPS, seed=None):
        """
        input:  max_steps (int) policy steps before truncation,
                seed (int or None)
        output: FR3PickPlaceEnv instance
        """
        super().__init__()

        self.model, self.data = R.setup_model()
        self.ctrl = R.ImpedanceController(
            self.model, self.data,
            kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
            kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
        )
        self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")

        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # 3 position deltas, 1 yaw delta, 1 gripper. The gripper is binary in
        # the dataset, so a continuous command here is thresholded rather than
        # passed through: an intermediate width averages across demonstrations
        # into a gradual closure that clips the block.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0
        self.grip = GRIPPER_OPEN

    def _ee(self):
        """
        Returns the tool centre point position.

        input:  none
        output: numpy array of shape (3,)
        """
        return np.array(self.data.site_xpos[self.tcp_id])

    def _block(self):
        """
        Returns the block position and its yaw folded into the cube's
        90 degree symmetry.

        input:  none
        output: (pos (3,), yaw float)
        """
        pos = np.array(self.data.xpos[self.block_id])
        q = np.array(self.data.xquat[self.block_id])
        yaw = 2.0 * np.arctan2(q[3], q[0])
        return pos, wrap(yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0

    def _obs(self):
        """
        Assembles the privileged observation.

        Angles enter as cosine and sine pairs rather than raw radians so the
        network never sees the discontinuity at the wrap point, which
        otherwise shows up as a policy that behaves differently either side
        of the same physical orientation.

        input:  none
        output: numpy array of shape (OBS_DIM,) float32
        """
        ee = self._ee()
        block, byaw = self._block()
        eyaw = quat_yaw(self._ee_quat())
        holding = RW.is_holding(block, self.grip)

        return np.concatenate([
            self.data.qpos[:R.N_ARM],
            self.data.qvel[:R.N_ARM],
            self.data.qpos[R.FINGER_DOFS],
            ee,
            [np.cos(eyaw), np.sin(eyaw)],
            self.target_pos,
            [np.cos(self.target_yaw), np.sin(self.target_yaw)],
            block,
            [np.cos(byaw), np.sin(byaw)],
            block - ee,
            block[:2] - PLACE_TARGET[:2],
            [self.grip, float(holding)],
        ]).astype(np.float32)

    def _ee_quat(self):
        """
        Returns the tool orientation as a quaternion.

        input:  none
        output: numpy array of shape (4,) as (w, x, y, z)
        """
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.array(self.data.site_xmat[self.tcp_id]).ravel())
        return q

    def reset(self, seed=None, options=None):
        """
        Starts a new episode with a freshly randomised block placement.

        Placements are drawn uniformly across the full region, matching the
        evaluation distribution rather than any biased training set, so the
        teacher is trained on what it will be measured on.

        input:  seed (int or None), options (dict or None)
        output: (observation, info)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.block_start, _ = R.reset_trial(self.model, self.data, self.ctrl, self.rng)

        pos, quat = self.ctrl.current_pose(self.data)
        self.target_pos = np.array(pos)
        self.target_yaw = quat_yaw(quat) if abs(quat[0]) < 0.5 else np.pi / 4.0
        self.grip = GRIPPER_OPEN
        self.steps = 0
        self.holding = False
        self.grasped_once = False
        self.released = False
        self.retreated = False
        self.prev_block = self._block()[0]
        self.prev_target = self.target_pos.copy()
        self.prev_speed = 0.0

        return self._obs(), {}

    def step(self, action):
        """
        Integrates one delta into the absolute target and simulates it.

        input:  action (array (5,)) in -1 to 1
        output: (observation, reward, terminated, truncated, info)
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        self.target_pos = np.clip(
            self.target_pos + a[:3] * MAX_DELTA_M, R.WORKSPACE_MIN, R.WORKSPACE_MAX
        )
        self.target_yaw = wrap(self.target_yaw + a[3] * MAX_DYAW_RAD)
        self.grip = GRIPPER_CLOSED if a[4] < 0.0 else GRIPPER_OPEN

        quat = down_quat(self.target_yaw)
        self.ctrl.set_target(self.target_pos, quat)

        # Target speed in mm per policy step, the same quantity the
        # demonstration profile in reference.py is written in.
        speed_mm = float(np.linalg.norm(self.target_pos - self.prev_target) * 1000.0)
        self.prev_target = self.target_pos.copy()

        for _ in range(R.STEPS_PER_ACTION):
            self.data.qfrc_applied[: R.N_ARM] = self.ctrl.compute_torque(self.data)
            self.data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(self.data, self.grip)
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        success, distance = check_success(self.data, self.block_id)
        block, _ = self._block()

        r, self.holding, dropped, grasped_now, phase = RW.compute(
            self._ee(), block, self.prev_block, block + GRASP_OFFSET, self.grip,
            self.holding, self.grasped_once, self.released, success,
            speed_mm, self.prev_speed,
        )
        self.grasped_once = self.grasped_once or grasped_now
        self.prev_block = block
        self.prev_speed = speed_mm

        # Block vertical speed, so a block still falling cannot count as
        # placed the instant it passes through the right height.
        block_vz = float(self.data.qvel[R.BLOCK_QVEL_ADR + 2])
        placed = success and RW.is_placed(block, self.grip, self._ee(), block_vz)
        if placed:
            self.released = True

        # The arm has to clear AND come back out to home, the way the
        # demonstrations do. Height alone was not enough: the arm was
        # already above the retreat height when it let the block go, so the
        # retreat fired instantly and the episode ended 0.7 steps later.
        if self.released:
            ee = self._ee()
            home = float(np.linalg.norm(ee[:2] - np.array(REF.HOME_XY)))
            if ee[2] >= REF.RETREAT_Z and home <= REF.RETREAT_HOME_TOL:
                self.retreated = True

        info = {
            "success": success,
            "placed": placed,
            "released": self.released,
            "retreated": self.retreated,
            "phase": phase,
            "speed_mm": speed_mm,
            "distance": distance,
            "holding": self.holding,
            "grasped": self.grasped_once,
            "dropped": dropped,
            # The 8 element absolute action, in the dataset's own layout, so
            # Stage 3 can relabel a student's states with no conversion.
            "absolute_action": np.concatenate(
                [self.target_pos, quat, [self.grip]]
            ).astype(np.float32),
        }

        # Terminate only once the arm has retreated, not at release and
        # certainly not at check_success. Each earlier termination rule cut
        # the episode off before the behaviour it was meant to elicit could
        # happen: check_success ended it mid-hover with the gripper shut,
        # and placed ended it before the arm ever lifted clear.
        terminated = bool(self.retreated or dropped)
        if terminated and not dropped:
            r += RW.terminal(self.steps, self.retreated)

        return self._obs(), r, terminated, self.steps >= self.max_steps, info


def make_env(seed=None, max_steps=R.MAX_POLICY_STEPS):
    """
    Factory for vectorised construction, which needs a picklable callable.

    input:  seed (int or None), max_steps (int)
    output: callable returning FR3PickPlaceEnv
    """
    def _init():
        return FR3PickPlaceEnv(max_steps=max_steps, seed=seed)
    return _init
