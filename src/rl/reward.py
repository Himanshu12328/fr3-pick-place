"""
The staged dense reward for the RL teacher.

Reward shaping on a pick-and-place is where the whole thing quietly goes
wrong. Four exploits got through earlier versions of this file, each fix
feeling complete until the next was found, so the reasoning is written down
rather than left in the coefficients.

    1  loitering       +2.0 per step for holding made carrying the block for
                       600 steps pay ~1000 against 100 for finishing
    2  repeat grasping the grasp bonus fired on every transition into
                       holding, so grasp, release, grasp paid +25 each time
    3  shoving         `check_success` is satisfied by a block slid across
                       the table, and the completion bonus was not gated on
                       a lift. Scored 99.0% while never holding the block in
                       two thirds of episodes
    4  never releasing `check_success` allows 20 mm of resting tolerance
                       while holding needs 10 mm, so a block hovering in
                       that window counts as held and resting at once.
                       Scored 99.8% with the gripper shut in 98 of 100

The fifth problem was not an exploit but a consequence. Making every
per-step term negative, which is what killed the loitering hack, means the
optimal policy is always the fastest one: speed is never traded against
anything. The result solved the task in 73 steps against the
demonstrations' 295, carried the block 31 mm up against their 80 mm, and
never retreated because the episode ended at release.

So this version stops asking only whether the task is done and starts
asking that it be done the way it was demonstrated. The flat time penalty
is replaced by a **quadratic cost on deviation from the demonstrated speed
for the current phase**, which has an interior optimum where a flat penalty
has none. Measured values live in `src/rl/reference.py`.
"""

import numpy as np

from src.data.task import BLOCK_Z, PLACE_TARGET
from src.rl import reference as REF

# --- what counts as holding, placed and released ------------------------

HOLD_LIFT_M = 0.010
RELEASE_OPEN_M = 0.02
SETTLED_M = 0.010
FLOOR_Z = 0.35

# --- task terms, unchanged from the version that reached 98.8% ----------

W_REACH = 1.0
W_TRANSPORT = 1.5
R_GRASP = 25.0
R_SUCCESS = 100.0
P_DROPPED = 50.0
P_PUSH = 100.0
P_FUMBLE = 10.0

# Per mm the block is nudged after it has been placed. The tool releases
# from 24 mm below the top of the block, so any horizontal motion before it
# rises clear drags the block along: measured at 17.4 mm on average, up to
# 38.4 mm, in ten episodes out of fifteen.
P_DISTURB = 0.5

# --- trajectory shaping, new ---------------------------------------------

# Cost on (speed - demonstrated speed for this phase) squared, in mm per
# step, and capped.
#
# The first version used 0.05 with no cap and it swamped the task. A policy
# moving at the 7 mm/step ceiling during approach deviates 5.15 from the
# 1.85 target, which squared and weighted came to 1.33 per step against a
# reach shaping term of about 0.15. The dominant gradient was therefore
# "move at 1.85 mm/step", not "go to the block", and the run that followed
# never learned the task at all: it shut the gripper on step zero, held it
# shut so it could never straddle the block, and ran out the clock at 0%
# success for 1.2 million steps.
#
# The cap matters as much as the weight. Trajectory shaping should nudge a
# policy that is solving the task, never outvote the task itself.
W_SPEED = 0.02
CAP_SPEED = 0.4

# A sixth problem, and the deepest. At gamma=0.99 the effective horizon is
# about 100 steps against a 295 step episode, so the agent is myopic:
# collecting the +140 terminal bonus 150 steps sooner is worth roughly 4.5
# times more in present value. Discounting was paying the policy to rush far
# harder than any of these weights were paying it not to, which is why a
# teacher trained with them still saturated the 7 mm/step ceiling and
# finished in 98 steps. The fix is gamma, set to 0.997 in the trainer so the
# horizon matches the episode. These weights only matter once it does.

# Same reasoning for the smoothness term.
W_JERK = 0.02
CAP_JERK = 0.2

# Cost on horizontal motion while carrying the block below transit height.
# This is what forces lift-then-move rather than the 31 mm skim.
W_LOW_CARRY = 2.0

# Pull toward retreat height once the block is down and let go.
W_RETREAT = 2.0
R_RETREAT = 40.0

# Terminal cost on episode length outside the demonstration band, free
# inside 255 to 335 steps.
W_DURATION = 0.15
DURATION_FREE = 40.0


def is_holding(block_pos, grip_cmd):
    """
    Decides whether the block is genuinely in the gripper.

    input:  block_pos (array (3,)), grip_cmd (float)
    output: bool
    """
    return bool(block_pos[2] > BLOCK_Z + HOLD_LIFT_M and grip_cmd < RELEASE_OPEN_M)


def is_placed(block_pos, grip_cmd, ee_pos=None, block_vz=0.0):
    """
    Decides whether the block was placed, rather than merely ending up in
    the right spot.

    Three conditions, and each was added because its absence was exploited.
    The gripper must be open, which stops a hover counting. The block must
    be within 5 mm of resting and barely moving, which stops a block still
    falling from counting the instant it passes through the right height.
    And the tool must have descended, which is the one that matters most:
    a teacher released from 95 mm up at 300 mm/s, let the block fall, and
    the criterion fired the moment it landed. That is a drop, not a place,
    and it looked identical in every aggregate number.

    input:  block_pos (array (3,)), grip_cmd (float),
            ee_pos (array (3,) or None) tool position,
            block_vz (float) block vertical speed in m/s
    output: bool
    """
    if grip_cmd <= RELEASE_OPEN_M:
        return False
    if abs(block_pos[2] - BLOCK_Z) >= REF.PLACE_SETTLED_M:
        return False
    if abs(block_vz) >= REF.PLACE_MAX_VZ:
        return False
    if ee_pos is not None and ee_pos[2] > REF.PLACE_EE_Z:
        return False
    return True


def compute(ee_pos, block_pos, prev_block_pos, grasp_point, grip_cmd,
            was_holding, grasped_once, released, success, speed_mm,
            prev_speed_mm):
    """
    Returns the reward for one policy step.

    input:  ee_pos (array (3,)), block_pos (array (3,)),
            prev_block_pos (array (3,)), grasp_point (array (3,)),
            grip_cmd (float), was_holding (bool), grasped_once (bool),
            released (bool) the gripper has opened after a real grasp,
            success (bool) from check_success,
            speed_mm (float) target displacement this step, mm,
            prev_speed_mm (float) the same last step
    output: (reward, holding, dropped, grasped_now, phase)
    """
    holding = is_holding(block_pos, grip_cmd)
    dropped = bool(block_pos[2] < FLOOR_Z)
    grasped_now = holding and not was_holding and not grasped_once
    picked = grasped_once or grasped_now

    phase = "retreat" if released else ("carry" if picked else "approach")

    r = 0.0

    # Fumbling: losing the block anywhere other than on the target.
    if was_holding and not holding and not success:
        r -= P_FUMBLE

    if holding:
        r -= W_TRANSPORT * float(np.linalg.norm(block_pos[:2] - PLACE_TARGET[:2]))
        if grasped_now:
            r += R_GRASP
        # Carrying the block across the table at skimming height is what the
        # previous teacher did. Charge for horizontal progress made below
        # transit height, so the only cheap route is to lift first.
        if block_pos[2] < REF.TRANSIT_Z - 0.02:
            r -= W_LOW_CARRY * float(
                np.linalg.norm(block_pos[:2] - prev_block_pos[:2]) * 1000.0
            ) * 0.01
    elif released:
        # Retreat in two stages, not diagonally. The tool lets go from 24 mm
        # below the top of the block, so pulling toward home and toward
        # height at once drags the open fingers across what was just placed.
        # Rise clear first; horizontal distance is only charged for after.
        shortfall = max(0.0, REF.RETREAT_Z - float(ee_pos[2]))
        r -= W_RETREAT * shortfall
        if shortfall <= 0.005:
            home = np.array(REF.HOME_XY)
            r -= W_RETREAT * float(np.linalg.norm(np.asarray(ee_pos)[:2] - home))

        # Anything that moves the block after it is placed partly undoes the
        # task, so it is charged for directly rather than left to the
        # success criterion to catch only when it goes 50 mm wrong.
        r -= P_DISTURB * float(
            np.linalg.norm(block_pos[:2] - prev_block_pos[:2]) * 1000.0
        )
    else:
        r -= W_REACH * float(np.linalg.norm(ee_pos - grasp_point))
        r -= P_PUSH * float(np.linalg.norm(block_pos[:2] - prev_block_pos[:2]))

    # Speed and smoothness against the demonstrated profile for this phase.
    # This replaces the flat time penalty, which had no interior optimum and
    # so always rewarded going faster.
    r -= min(CAP_SPEED, W_SPEED * (speed_mm - REF.SPEED_TARGET[phase]) ** 2)
    r -= min(CAP_JERK,
             W_JERK * max(0.0, abs(speed_mm - prev_speed_mm) - REF.JERK) ** 2)

    if dropped:
        r -= P_DROPPED

    return r, holding, dropped, grasped_now, phase


def terminal(steps, retreated):
    """
    Returns the one-time reward for finishing the episode properly.

    input:  steps (int) episode length, retreated (bool) the arm cleared
    output: float
    """
    if not retreated:
        return 0.0
    off = abs(steps - REF.TOTAL_STEPS[0]) - DURATION_FREE
    return R_SUCCESS + R_RETREAT - W_DURATION * max(0.0, off)
