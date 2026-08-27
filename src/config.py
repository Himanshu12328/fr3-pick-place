"""
config.py

Project paths and shared constants, resolved relative to the repository
root rather than hardcoded to one machine.

Two problems this solves. Every script previously carried its own absolute
path, so the project only ran from C:\\pick_place on one computer. And
several constants were duplicated across the collection, evaluation, and
re-rendering paths, where divergence between copies produces a policy that
trains correctly and evaluates at zero without raising anything.

Anything that must agree between training and evaluation belongs here.
"""

import os
from pathlib import Path

# ------------------------------------------------------------- paths ----

ROOT = Path(__file__).resolve().parent.parent

MENAGERIE_DIR = Path(os.environ.get("MENAGERIE_DIR", ROOT / "mujoco_menagerie"))
FR3_DIR = MENAGERIE_DIR / "franka_fr3"

MODELS_DIR = ROOT / "models"
SCENE_PATH = MODELS_DIR / "fr3_pick_place.xml"

# FR3_DATA_ROOT lets datasets live on a separate drive. A 70-episode
# dataset at full resolution is tens of gigabytes.
DATA_ROOT = Path(os.environ.get("FR3_DATA_ROOT", ROOT / "data"))
RAW_DATA_DIR = DATA_ROOT / "pick_place_v1"
SMALL_DATA_DIR = DATA_ROOT / "pick_place_v1_small"
LEROBOT_DIR = DATA_ROOT / "lerobot"

OUTPUT_ROOT = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
DOCS_DIR = ROOT / "docs"


def ensure_dirs():
    """
    Creates the output directories that scripts write into.

    input:  none
    output: None
    """
    for d in (MODELS_DIR, DATA_ROOT, OUTPUT_ROOT, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------- model and scene ------

# Degrees of freedom belonging to the arm. The scene also contains two
# finger joints and a free-floating block, so model.nv is 15 while the
# controller drives only the leading seven.
N_ARM = 7
FINGER_DOFS = [7, 8]

# The block's free joint occupies seven qpos entries and six qvel entries.
# These addresses differ in general, because a free joint's quaternion
# takes four qpos slots but only three qvel slots. In this scene the block
# is the last joint and both happen to be 9.
BLOCK_QPOS_ADR = 9
BLOCK_QVEL_ADR = 9

# FR3 datasheet torque limits, Nm.
FR3_TORQUE_LIMITS = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]

# ------------------------------------------------------- cameras --------

CAMERAS = ["external", "wrist_left", "wrist_right"]

# Collection resolution. Stored at this size so the dataset can be
# re-rendered smaller later without re-collecting.
COLLECT_WIDTH, COLLECT_HEIGHT = 640, 480

# Training and evaluation resolution. These MUST match, and must match the
# resolution the policy was trained at. The vision backbone is fully
# convolutional and accepts any size silently, but crop_shape is applied
# in pixels: rendering at 640x480 for a policy trained at 128x160 turns a
# whole-scene 90 percent crop into a small centre patch and the policy
# goes effectively blind. This dropped measured success from 60 percent to
# 0.4 percent with no error raised.
TRAIN_WIDTH, TRAIN_HEIGHT = 160, 128

# ------------------------------------------------------- rates ----------

CONTROL_HZ = 1000
POLICY_HZ = 30
RECORD_FPS = 30
STEPS_PER_ACTION = CONTROL_HZ // POLICY_HZ

# ------------------------------------------------- controller gains -----

# Measured on this model with joint friction disabled. See
# logs/zeta_sweep_kp800.png for the characterisation these came from.
#
#   transit   1.5 Hz bandwidth, 2.9 percent overshoot
#   contact   0.4 Hz bandwidth, no overshoot, softer on impact
KP_TRANS, KP_ROT, ZETA = 800.0, 60.0, 0.7
KP_NULL, ZETA_NULL = 10.0, 1.0

CONTACT_KP_TRANS, CONTACT_KP_ROT, CONTACT_ZETA = 300.0, 30.0, 1.0

# Finger joints are driven by a simple PD, not by the impedance
# controller. The force clamp lets a closed gripper squeeze rather than
# crush, which keeps the contact solver stable.
GRIP_KP, GRIP_KD, GRIP_FORCE_LIMIT = 300.0, 15.0, 60.0

# ------------------------------------------------------ workspace -------

# Metres. The lower z bound sits just above the table surface at 0.40, so
# the fingers can straddle the block rather than only touching its top
# face, without being able to drive through the table.
WORKSPACE_MIN = [0.25, -0.35, 0.405]
WORKSPACE_MAX = [0.85, 0.35, 0.85]

# ----------------------------------------------------------- task -------

TASK_STRING = "pick up the red block and place it on the green target"

# Action layout: target position 3, target quaternion 4, gripper 1.
ACTION_DIM = 8
ACTION_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]

# State layout: 7 joint positions, 7 joint velocities, 2 finger positions.
STATE_DIM = 16

# Binary gripper. A rigid cube has no use for intermediate openings, and a
# continuous target gets averaged across demonstrations into a gradual
# closure that clips the block. Binary also matches what VLA action heads
# are pretrained on.
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
