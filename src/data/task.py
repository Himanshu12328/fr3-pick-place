"""
Defines the pick-and-place task: where the block starts, where it must end
up, and what counts as success.

Keeping this separate from the collection loop means the same definitions
drive both data collection and policy evaluation. If success were checked
one way during collection and another during eval, the reported success
rate would not mean what it appears to mean.
"""

import numpy as np

# Randomisation region for the block start, metres. Centred on the table
# and kept clear of the place target so the two never overlap.
BLOCK_X_RANGE = (0.46, 0.64)
BLOCK_Y_RANGE = (-0.12, 0.06)
BLOCK_Z = 0.422                # table top plus half the block edge

# Where the block must be delivered. Matches the place_target site.
PLACE_TARGET = np.array([0.55, 0.20, BLOCK_Z])
SUCCESS_RADIUS = 0.05           # metres, horizontal distance
SUCCESS_Z_TOLERANCE = 0.02      # block must be resting, not held aloft


def sample_block_pose(rng, bias_near=False, split_x=0.53):
    """
    Draws a random block start position and yaw.

    Randomising the start is what forces the policy to actually look at the
    scene. With a fixed start, a policy can reproduce one memorised
    trajectory and score perfectly while having learned nothing
    transferable, and every evaluation trial becomes the same trial.

    bias_near restricts sampling to the near half of the x range.
    Evaluation showed the trained policy roughly 16 points weaker there,
    and the uniform distribution centred at 0.55 puts only about a third of
    placements below 0.53 — so that region is both harder and less
    represented. Collecting a session with this flag corrects the imbalance
    without discarding existing episodes. Evaluation always samples
    uniformly, so the reported number stays on the real distribution.

    Only yaw is randomised, not full orientation, because a cube tipped
    onto an edge is a different and much harder grasp problem.

    input:  rng (numpy Generator), bias_near (bool), split_x (float) metres
    output: (pos, quat) with pos shape (3,) and quat shape (4,) as (w,x,y,z)
    """
    x_hi = split_x if bias_near else BLOCK_X_RANGE[1]

    pos = np.array([
        rng.uniform(BLOCK_X_RANGE[0], x_hi),
        rng.uniform(*BLOCK_Y_RANGE),
        BLOCK_Z,
    ])

    yaw = rng.uniform(-np.pi / 4, np.pi / 4)
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    return pos, quat

def set_block_pose(model, data, pos, quat, qpos_adr=9, qvel_adr=9):
    """
    Writes a block pose into the simulation state and zeroes its velocity.

    The block's free joint occupies seven qpos entries and six qvel
    entries. These addresses differ in general, because a free joint's
    quaternion takes four qpos slots but only three qvel slots, so any
    joint appearing before it shifts the two indices apart. In this scene
    the block is the last joint and both happen to be 9.

    input:  model (MjModel), data (MjData), pos (array (3,)),
            quat (array (4,)), qpos_adr (int), qvel_adr (int)
    output: None
    """
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
    data.qvel[qvel_adr:qvel_adr + 6] = 0.0

def check_success(data, block_body_id):
    """
    Returns whether the block has been delivered to the target.

    Two conditions, both necessary. Horizontal distance under the success
    radius means it reached the right place. The height check means it was
    released rather than still gripped, which stops an episode ending
    mid-air from counting.

    input:  data (MjData), block_body_id (int)
    output: (success, distance_m)
    """
    pos = data.xpos[block_body_id]
    horizontal = np.linalg.norm(pos[:2] - PLACE_TARGET[:2])
    resting = abs(pos[2] - BLOCK_Z) < SUCCESS_Z_TOLERANCE
    return bool(horizontal < SUCCESS_RADIUS and resting), float(horizontal)