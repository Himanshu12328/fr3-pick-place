"""
A scripted pick-and-place that reads the true block pose from the
simulator.

This exists to measure the ceiling. Every success rate in this project is
reported against an unstated assumption: that the task is solvable often
enough for the number to be worth improving. Nobody has checked. A policy
at 83.5% might be 12 points from a solved task or 12 points from a hard
limit set by the contact model, the gripper geometry or the 600 step
budget, and those two situations call for completely different work.

The oracle answers that. It has perfect perception and perfect timing, so
whatever it scores is an upper bound on any vision policy trained in this
environment. If it scores 100%, the entire remaining gap is learnable. If
it scores 96%, then chasing 99% from pixels is chasing something that is
not there.

It deliberately does NOT use a planner or inverse kinematics. It drives the
same absolute target pose interface the demonstrations used, through the
same impedance controller, rate limited to the same speeds a human
produced. An oracle that cheated by teleporting the target or by using a
different control path would measure the ceiling of a different system.

The waypoints below are not invented. Every one of them was extracted from
the 221 recorded demonstrations, so the oracle attempts the grasp the human
operators actually used:

    grasp target     block centre + (-2.5, 0.1, -3.0) mm
    gripper yaw      block yaw, within -4.6 +- 5.6 degrees
    hover z          0.4485 +- 0.0078
    transit z        0.4993 +- 0.0155
    release target   place target + (-14.5, 1.3, -3.1) mm
"""

import mujoco
import numpy as np

from src.data.task import BLOCK_Z, PLACE_TARGET
from src.rl.reference import TRANSIT_Z as REF_TRANSIT_Z

# Offsets measured from the demonstrations, metres. See module docstring.
GRASP_OFFSET = np.array([-0.0025, 0.0001, -0.0030])
RELEASE_OFFSET = np.array([-0.0145, 0.0013, -0.0031])
HOVER_Z = 0.4485
TRANSIT_Z = 0.4993
RETREAT_Z = 0.5181        # where the demonstrations lift the arm to afterwards
HOME_XY = (0.5545, 0.0)   # the scene's home keyframe, where retreats head back to

# Target motion limits. A single speed for the whole episode was enough to
# measure the ceiling, but the oracle is now also the seed data for the RL
# teacher's replay buffer, so it moves at the speed the demonstrations
# actually used in each phase: slow to position, quicker to carry, quicker
# still to leave once the task is done. Measured in src/rl/reference.py.
STEP_M = 0.004          # fallback, and what the ceiling was measured at
PHASE_STEP_M = {
    "approach": 0.00185,
    "carry": 0.00327,
    "retreat": 0.00478,
}
YAW_STEP_RAD = np.deg2rad(4.0)

# How far ahead of the tool the reactive labeller places its target, per
# phase, as a multiple of the base lead.
#
# A fixed lead produces a fixed speed. The base 12 mm lead was measured to
# drive the tool at about 2.7 mm per step regardless of phase, which is why
# every episode collected through `label()` came out flat at 2.28 / 2.86 /
# 2.74 against demonstrations of 1.85 / 3.27 / 4.78. The phase machine gets
# the profile right because it rate limits per phase; the labeller has to
# get it right through the only control it has, which is the lead.
#
# The scales are the ratio of each demonstration speed to that measured
# 2.7 mm per step. `label()` cannot be replaced by the phase machine for
# DAgger, because it has to answer for states the oracle's own trajectory
# never visits, so this is the fix that applies there.
LEAD_BASE_SPEED_M = 0.0027
LEAD_SCALE = {
    "approach": PHASE_STEP_M["approach"] / LEAD_BASE_SPEED_M,
    "carry": PHASE_STEP_M["carry"] / LEAD_BASE_SPEED_M,
    "retreat": PHASE_STEP_M["retreat"] / LEAD_BASE_SPEED_M,
}

# Frames to hold still while the fingers close or open. The gripper is a PD
# loop, so the command has to persist long enough for the fingers to move.
CLOSE_HOLD = 20
OPEN_HOLD = 12

# Per-phase step cap. A phase that cannot reach its waypoint should end the
# trial as a failure rather than silently eat the whole step budget.
PHASE_CAP = 200

# Tolerances on the end effector, metres. The commanded target leads the end
# effector, so advancing a phase when the target arrives would close the
# gripper while the arm is still short of the block.
EE_TOL_TIGHT = 0.006    # grasp and release, where millimetres matter
EE_TOL_LOOSE = 0.020    # transit, where they do not

# LIFT_OFF sits between RELEASE and RETREAT. Without it the tool heads
# straight for home from the release pose, which is 24 mm BELOW the top
# of the block it has just set down, so the open fingers drag it along:
# measured 17.4 mm of block movement on average, up to 38.4 mm, in ten
# episodes out of fifteen. Rising clear before translating costs a few
# steps and leaves the block where it was put.
HOVER, DESCEND, CLOSE, LIFT, TRANSIT, LOWER = range(6)
RELEASE, LIFT_OFF, RETREAT, DONE = range(6, 10)


def down_quat(yaw):
    """
    Builds a gripper-down orientation at a given yaw.

    A 180 degree rotation about an axis lying in the xy plane at angle
    yaw/2 sends the tool z axis to point at the table and leaves a rotation
    of yaw about vertical. The scene's home keyframe is exactly this form
    with yaw at 45 degrees, which is how the convention was confirmed.

    input:  yaw (float) radians
    output: numpy array of shape (4,), quaternion as (w, x, y, z)
    """
    return np.array([0.0, np.cos(yaw / 2.0), np.sin(yaw / 2.0), 0.0])


def quat_yaw(quat):
    """
    Recovers the yaw of a gripper-down quaternion.

    input:  quat (array (4,)) as (w, x, y, z)
    output: float, radians
    """
    return 2.0 * np.arctan2(quat[2], quat[1])


def wrap(angle):
    """
    Wraps an angle to the interval -pi to pi.

    input:  angle (float) radians
    output: float, radians
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class ScriptedOracle:
    """
    A privileged pick-and-place policy with the harness's callable
    interface.

    Reads the block pose straight out of MjData rather than from the
    observation, which is the whole point: it is measuring what the
    environment allows, not what is learnable from pixels.
    """

    def __init__(self, step_m=None, verbose=False, jitter=0.0, seed=None):
        """
        input:  step_m (float or None) fixed target speed in metres per
                policy step; None uses the per-phase demonstration speeds,
                verbose (bool) print phase transitions,
                jitter (float) 0 for the deterministic oracle, 1.0 for the
                full per-episode randomisation described in reset(),
                seed (int or None) for the jitter generator
        output: ScriptedOracle instance
        """
        self.step_m = step_m
        self.verbose = verbose
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.data = None
        self.reset()

    def bind(self, model, data):
        """
        Gives the oracle the simulator handles it needs to read the block.

        The harness calls this once per evaluation run, in the same way it
        calls reset() once per trial. Binding rather than constructing with
        a model keeps the oracle usable with whatever scene the harness
        decides to build.

        input:  model (MjModel), data (MjData)
        output: None
        """
        self.model = model
        self.data = data
        self.block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        if self.tcp_id < 0:
            raise RuntimeError("scene has no 'tcp' site, cannot run the oracle")

    def reset(self):
        """
        Clears phase state between trials.

        input:  none
        output: None
        """
        self.phase = HOVER
        self.target = None
        self.yaw = None
        self.grip = 0.04
        self.hold = 0
        self.phase_steps = 0
        self.stalled = False

        # Per-episode variation. A deterministic demonstrator produces
        # degenerate data: every episode with the same block pose is the
        # same episode, so a policy can fit the demonstrator rather than the
        # task. The operator never repeated themselves either, and the
        # recorded episodes span hover heights of +-7.8 mm, transit heights
        # of +-15.5 mm and per-episode mean speeds from 2.46 to 3.64 mm per
        # step. The ranges here are drawn to sit inside those.
        j = self.jitter
        r = self.rng
        self.jit_hover = HOVER_Z + j * r.uniform(-0.010, 0.010)
        self.jit_transit = TRANSIT_Z + j * r.uniform(-0.015, 0.015)
        self.jit_grasp = GRASP_OFFSET + j * np.array(
            [r.uniform(-0.003, 0.003), r.uniform(-0.003, 0.003), 0.0]
        )
        self.jit_speed = 1.0 + j * r.uniform(-0.20, 0.20)
        self.jit_lead = 0.012 + j * r.uniform(-0.003, 0.003)

    def _ee(self):
        """
        Returns the current tool centre point position.

        input:  none
        output: numpy array of shape (3,)
        """
        return np.array(self.data.site_xpos[self.tcp_id])

    def _block(self):
        """
        Returns the block's current position and yaw.

        input:  none
        output: (pos (3,), yaw float radians)
        """
        pos = np.array(self.data.xpos[self.block_id])
        quat = np.array(self.data.xquat[self.block_id])
        yaw = 2.0 * np.arctan2(quat[3], quat[0])
        # A cube is symmetric every 90 degrees, so grasp at the smallest
        # wrist rotation that lines the fingers up with a pair of faces.
        return pos, wrap(yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0

    def _waypoint(self):
        """
        Returns the goal pose and gripper command for the current phase.

        input:  none
        output: (pos (3,), yaw float, grip float, ee_tol float)
        """
        block, byaw = self._block()

        if self.phase in (HOVER, DESCEND, CLOSE):
            # The jittered offsets, not the bare constants. `__call__` read
            # GRASP_OFFSET, HOVER_Z and TRANSIT_Z directly while only
            # label() used the per-episode jitter, so the phase machine was
            # fully deterministic: the same block pose produced byte
            # identical episodes. That is the degenerate demonstrator the
            # jitter was added to avoid, and it was silently exempt from it.
            grasp = block + self.jit_grasp
            if self.phase == HOVER:
                return (np.array([grasp[0], grasp[1], self.jit_hover]), byaw,
                        0.04, EE_TOL_LOOSE)
            if self.phase == DESCEND:
                return grasp, byaw, 0.04, EE_TOL_TIGHT
            return grasp, byaw, 0.0, EE_TOL_TIGHT

        place = PLACE_TARGET + RELEASE_OFFSET
        yaw = self.yaw if self.yaw is not None else 0.0

        if self.phase == LIFT:
            here = self._ee()
            return (np.array([here[0], here[1], self.jit_transit]), yaw,
                    0.0, EE_TOL_LOOSE)
        if self.phase == TRANSIT:
            return (np.array([place[0], place[1], self.jit_transit]), yaw,
                    0.0, EE_TOL_LOOSE)
        if self.phase == LOWER:
            return place, yaw, 0.0, EE_TOL_TIGHT
        if self.phase == RELEASE:
            return place, yaw, 0.04, EE_TOL_TIGHT
        if self.phase == LIFT_OFF:
            # Straight up from where it let go, same xy, clearing the block
            # before any horizontal motion begins.
            return np.array([place[0], place[1], RETREAT_Z]), yaw, 0.04, EE_TOL_LOOSE
        # Then out to home. The demonstrations do not simply lift and stop,
        # they bring the arm back where it started: 282 mm of travel over 59
        # steps, against the 99 mm a vertical lift alone would take.
        return np.array([HOME_XY[0], HOME_XY[1], RETREAT_Z]), yaw, 0.04, EE_TOL_LOOSE

    def __call__(self, obs):
        """
        Returns the next absolute target pose and gripper command.

        The observation is ignored. The oracle reads the simulator directly,
        which is exactly the privilege being measured.

        input:  obs (dict) harness observation, unused
        output: numpy array of shape (8,)
        """
        if self.data is None:
            raise RuntimeError("oracle used before bind(model, data)")

        if self.target is None:
            self.target = self._ee()
            self.yaw = quat_yaw(down_quat(np.pi / 4.0))

        goal, goal_yaw, grip, ee_tol = self._waypoint()

        # Rate limited move toward the waypoint, at the speed the
        # demonstrations used for whichever phase this is.
        # Per-phase demonstration speed, scaled by this episode's jitter.
        # The recorded episodes span per-episode mean speeds from 2.46 to
        # 3.64 mm per step, so a fixed speed is narrower than the data.
        step = (self.step_m if self.step_m is not None
                else PHASE_STEP_M[self.phase_name()] * self.jit_speed)
        delta = goal - self.target
        dist = float(np.linalg.norm(delta))
        if dist > step:
            self.target = self.target + delta / dist * step
        else:
            self.target = goal.copy()

        dyaw = wrap(goal_yaw - self.yaw)
        if abs(dyaw) > YAW_STEP_RAD:
            self.yaw += np.sign(dyaw) * YAW_STEP_RAD
        else:
            self.yaw = goal_yaw

        self.grip = grip
        self._advance(goal, goal_yaw, ee_tol)

        return np.concatenate([self.target, down_quat(self.yaw), [self.grip]])

    def phase_name(self):
        """
        Maps the phase index onto the three demonstration phases.

        input:  none
        output: str, one of approach, carry, retreat
        """
        if self.phase <= CLOSE:
            return "approach"
        if self.phase <= RELEASE:
            return "carry"
        return "retreat"

    def label(self, lead_m=None):
        """
        Returns the action a demonstrator would take from the world state as
        it is right now, with no memory of how it got here.

        This is the method for DAgger relabelling, and it is deliberately
        reactive. Two earlier versions were not, and both were wrong.

        `__call__` integrates the oracle's own target from its own previous
        target, which is correct while the oracle drives and wrong when a
        student does: measured during student-driven rollouts its labels led
        the real tool by 41.9 mm on average and up to 119.8 mm, against the
        11.8 mm the demonstrations lead by.

        The version after that fixed the lead but kept the latched phase
        machine, which advances only when the real arm reaches the oracle's
        waypoint within 6 mm. A student following its own path never
        satisfies that, so the phase stalled in DESCEND and the label said
        "keep the gripper open" for the whole episode. The collected data
        had the gripper open in 96% of frames against 57.5% in the recorded
        demonstrations, and a policy trained on it scored 11%.

        A DAgger teacher has to answer for any state the student can reach,
        including states its own trajectory never visits. That rules out a
        latched machine entirely. Everything below is decided from the
        current block pose, tool pose and finger width.

        input:  lead_m (float) how far ahead of the tool to place the
                target, metres; 0.012 matches the demonstrations
        output: numpy array of shape (8,), target pose and gripper
        """
        if self.data is None:
            raise RuntimeError("oracle used before bind(model, data)")
        if lead_m is None:
            lead_m = self.jit_lead

        block, byaw = self._block()
        ee = self._ee()
        # The commanded gripper, not the measured finger width. Fingers
        # resting on the 44 mm block settle around 0.030 and never reach the
        # 0.02 a "closed" test would want, so keying off the measurement
        # left the arm shut on the block and standing still forever.
        closed = float(self.grip) < 0.02

        grasp = block + self.jit_grasp
        place = PLACE_TARGET + RELEASE_OFFSET
        # Is the block in the hand right now?
        #
        # Two ways of knowing, and both are needed.
        #
        # The first is the oracle's own commanded gripper. Defining "held"
        # by lift height *alone* deadlocks when the oracle drives: lifting
        # is only commanded once held, and held would only become true once
        # lifted, so the arm closes on the block and stands there. That
        # scored 0 of 20 and is why this test was written this way.
        #
        # But `closed` is the oracle's own last command, which is latched
        # state, and it only becomes closed when the tool reaches the
        # oracle's own grasp pose within 6 mm. **When a student is driving,
        # it grasps from wherever it likes, so the oracle may never latch.**
        # Measured on 300 DAgger episodes: in 84 of them the label never
        # commanded a close at all, and in 74 of those the student had
        # actually lifted and carried the block, succeeding in 69. Those
        # episodes were labelled "keep approaching the block" from start to
        # finish while the block sat in the gripper — labels that are worse
        # than useless, because they are confidently wrong.
        #
        # So the second way: the block is off the table and next to the
        # tool. That is a fact about the world rather than about the
        # oracle's memory, it is true no matter who closed the fingers, and
        # it cannot deadlock because it is an OR with the first test.
        near = bool(np.linalg.norm(block - ee) < 0.045)
        off_table = bool(block[2] > BLOCK_Z + 0.008)
        held = bool((closed and np.linalg.norm(block - ee) < 0.035)
                    or (near and off_table))
        at_target = bool(np.linalg.norm(block[:2] - PLACE_TARGET[:2]) < 0.04)
        settled = bool(abs(block[2] - BLOCK_Z) < 0.01)

        phase = "carry" if held else (
            "retreat" if (at_target and settled) else "approach")

        if held:
            # Carrying. Lift clear before translating, then descend on the
            # target, then let go once low.
            yaw_goal = self.yaw if self.yaw is not None else byaw
            # Whether the block is over the target decides everything here,
            # and it is measured on the block against the real target rather
            # than on the tool against the offset release pose: that pose
            # sits 14.5 mm away, so a 12 mm tolerance on the tool could never
            # be satisfied and the arm hovered over the target forever.
            over_target = np.linalg.norm(block[:2] - PLACE_TARGET[:2]) <= 0.02

            if not over_target:
                # Lift clear first, then translate. Checking the lift only
                # on this side of the branch matters: testing block height
                # once the block was already over the target made the target
                # chatter between "rise" and "descend" every step, because
                # the block hovers within a millimetre of the transit
                # threshold while being carried.
                if block[2] < REF_TRANSIT_Z:
                    goal, grip = np.array([ee[0], ee[1], self.jit_transit]), 0.0
                else:
                    goal, grip = np.array([place[0], place[1], self.jit_transit]), 0.0
            elif ee[2] > place[2] + 0.008:
                goal, grip = place.copy(), 0.0
            else:
                goal, grip = place.copy(), 0.04

        elif at_target and settled:
            # Placed. Rise straight up clear of the block, then go home.
            yaw_goal = self.yaw if self.yaw is not None else byaw
            if ee[2] < RETREAT_Z - 0.005:
                goal = np.array([ee[0], ee[1], RETREAT_Z])
            else:
                goal = np.array([HOME_XY[0], HOME_XY[1], RETREAT_Z])
            grip = 0.04

        else:
            # Going for the block. Line up above it, descend, then close.
            yaw_goal = byaw
            xy_off = float(np.linalg.norm(ee[:2] - grasp[:2]))
            if xy_off > 0.010 or ee[2] > self.jit_hover + 0.010:
                goal, grip = np.array([grasp[0], grasp[1], self.jit_hover]), 0.04
            elif ee[2] > grasp[2] + 0.006:
                goal, grip = grasp.copy(), 0.04
            else:
                # On the block. Close, and keep the target here so the
                # fingers have something steady to close around.
                goal, grip = grasp.copy(), 0.0

        # Scale the lead to the phase, so the tool moves at the speed the
        # demonstrations moved at during this part of the episode rather
        # than at one speed throughout.
        lead = lead_m * LEAD_SCALE[phase]

        delta = goal - ee
        dist = float(np.linalg.norm(delta))
        target = goal.copy() if dist <= lead else ee + delta / dist * lead

        yaw = quat_yaw(self._ee_quat())
        dyaw = wrap(yaw_goal - yaw)
        if abs(dyaw) > YAW_STEP_RAD:
            yaw += np.sign(dyaw) * YAW_STEP_RAD
        else:
            yaw = yaw_goal

        self.target, self.yaw, self.grip = target, yaw, grip
        return np.concatenate([target, down_quat(yaw), [grip]])

    def _ee_quat(self):
        """
        Returns the tool orientation as a quaternion.

        input:  none
        output: numpy array of shape (4,) as (w, x, y, z)
        """
        q = np.zeros(4)
        mujoco.mju_mat2Quat(
            q, np.array(self.data.site_xmat[self.tcp_id]).ravel()
        )
        return q

    def _advance(self, goal, goal_yaw, ee_tol, target_reached=False):
        """
        Moves to the next phase once this one has actually completed.

        Two conditions, both necessary. The commanded target has to have
        reached the waypoint, and the arm has to have caught up to it. The
        second is what stops the gripper closing on empty air while the end
        effector is still 15 mm short, which is the failure the target lead
        would otherwise cause.

        input:  goal (array (3,)), goal_yaw (float), ee_tol (float)
        output: None
        """
        self.phase_steps += 1
        if self.phase_steps > PHASE_CAP:
            self.stalled = True

        if self.phase in (CLOSE, RELEASE):
            self.hold += 1
            if self.hold >= (CLOSE_HOLD if self.phase == CLOSE else OPEN_HOLD):
                self._next()
            return

        target_there = target_reached or np.linalg.norm(self.target - goal) < 1e-3
        ee_there = np.linalg.norm(self._ee() - goal) < ee_tol
        yaw_there = abs(wrap(self.yaw - goal_yaw)) < np.deg2rad(2.0)

        if target_there and ee_there and yaw_there:
            self._next()

    def _next(self):
        """
        Steps the phase counter and clears the per-phase timers.

        input:  none
        output: None
        """
        if self.phase < DONE:
            self.phase += 1
        self.hold = 0
        self.phase_steps = 0
        if self.verbose:
            print(f"    phase -> {self.phase}")
