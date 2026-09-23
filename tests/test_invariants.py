"""
Guards the cross-module invariants whose violation produces silently wrong
results rather than an error.

Every check here corresponds to a bug that actually happened. Ordinary unit
tests of individual functions are not the point: the failures that cost the
most time in this project were all disagreements between two modules that
each looked correct in isolation.

These tests deliberately avoid loading the MuJoCo scene or any dataset, so
they run in CI without a GPU, without the generated model, and without the
tens of gigabytes of episodes.

Run:
    pytest
"""

import ast
from pathlib import Path

import pytest

from src import config

SRC = Path(config.__file__).parent


def function_source(module_path, func_name):
    """
    Extracts one function's source from a file without importing it.

    Importing would pull in mujoco, lerobot and torch, which makes the test
    suite slow and couples it to a working GPU install. Parsing the AST
    gives the same information for a fraction of the cost.

    input:  module_path (Path), func_name (str)
    output: str source of the function, or None if not found
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.unparse(node)
    return None


def literal_or_config_value(node):
    """
    Evaluates an assignment's right-hand side, resolving bare names against
    config rather than giving up on them.

    This exists because of a test that was not testing anything.
    `test_eval_resolution_matches_training_resolution` guards the most
    expensive defect in this project's history: rendering at 640x480 for a
    policy trained at 160x128 applies crop_shape in pixels, turns a
    whole-scene view into a small centre patch, and drops measured success
    from 60% to 0.4% with nothing raised anywhere.

    `rollout.py` assigns `CAM_WIDTH, CAM_HEIGHT = TRAIN_WIDTH, TRAIN_HEIGHT`.
    The tuple form was handled, but the right-hand side is two *names*, not
    two literals, so `ast.literal_eval` raised and the helper returned None.
    The guard then compared None against 160 and failed — and with CI
    disabled since "chore: disable CI", nothing reported it. A broken guard
    is worse than no guard, because the absence of a complaint reads as a
    pass.

    input:  node (ast.AST) the value side of an assignment
    output: the value, or None if it cannot be resolved
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        pass

    if isinstance(node, ast.Name):
        return getattr(config, node.id, None)

    if isinstance(node, ast.Tuple):
        return tuple(literal_or_config_value(e) for e in node.elts)

    return None


def module_constant(module_path, name):
    """
    Reads a module-level constant assignment without importing the module.

    input:  module_path (Path), name (str)
    output: the value, or None if not found or not resolvable
    """
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return literal_or_config_value(node.value)
                # Handles tuple assignment such as A, B = 1, 2
                if isinstance(target, ast.Tuple):
                    names = [e.id for e in target.elts if isinstance(e, ast.Name)]
                    if name in names:
                        values = literal_or_config_value(node.value)
                        if isinstance(values, (tuple, list)):
                            return values[names.index(name)]
                        return None
    return None


# --------------------------------------------------------------- config ---


def test_action_layout_is_self_consistent():
    """
    ACTION_DIM must match the number of named action components.
    """
    assert config.ACTION_DIM == len(config.ACTION_NAMES)


def test_gripper_states_are_distinct():
    """
    A binary gripper needs two distinguishable values, and the closed state
    must be the smaller one.
    """
    assert config.GRIPPER_CLOSED < config.GRIPPER_OPEN


def test_workspace_bounds_are_ordered():
    """
    Every lower bound must be below its upper bound, or np.clip silently
    inverts the interval and pins the target to one face of it.
    """
    for lo, hi in zip(config.WORKSPACE_MIN, config.WORKSPACE_MAX, strict=True):
        assert lo < hi


def test_workspace_floor_clears_the_table():
    """
    The lower z bound must sit above the table surface but below the block
    centre, so the fingers can straddle the block rather than only touching
    its top face.

    Set too high, this makes the task impossible to demonstrate: the
    gripper can only ever contact the block's top edge.
    """
    table_top = 0.40
    block_centre = 0.422
    assert table_top < config.WORKSPACE_MIN[2] < block_centre


def test_decimation_is_close_to_the_nominal_policy_rate():
    """
    The control rate need not divide evenly by the policy rate, but the
    achieved rate must be close to the nominal one.

    1000 Hz control at a nominal 30 Hz policy rate gives 33 steps per
    action, so the real rate is 30.30 Hz. That 1 percent error is far
    below anything a 1.5 Hz-bandwidth arm responds to, and it is identical
    across collection, re-rendering and evaluation, so nothing is
    misaligned relative to anything else. What would matter is a larger
    discrepancy, or a discrepancy that differed between those three paths.

    input:  none
    output: None
    """
    achieved = config.CONTROL_HZ / config.STEPS_PER_ACTION
    error = abs(achieved - config.POLICY_HZ) / config.POLICY_HZ
    assert error < 0.02, f"achieved {achieved:.2f} Hz vs nominal {config.POLICY_HZ}"


# ----------------------------------------------------- cross-module -------


def test_eval_resolution_matches_training_resolution():
    """
    The rollout harness must render at the resolution the policy trained on.

    The vision backbone is fully convolutional and accepts any input size
    without complaint, but crop_shape is applied in pixels. Rendering at
    640x480 for a policy trained at 128x160 turns a whole-scene crop into a
    small centre patch, the policy goes effectively blind, and it runs on
    proprioception alone. This dropped measured success from 60 percent to
    0.4 percent with nothing raised anywhere.
    """
    rollout = SRC / "eval" / "rollout.py"
    width = module_constant(rollout, "CAM_WIDTH")
    height = module_constant(rollout, "CAM_HEIGHT")

    assert (width, height) == (config.TRAIN_WIDTH, config.TRAIN_HEIGHT), (
        f"rollout renders at {width}x{height} but training used "
        f"{config.TRAIN_WIDTH}x{config.TRAIN_HEIGHT}"
    )


def test_rerender_resolution_matches_training_resolution():
    """
    The re-render step writes the dataset the policy trains on, so its
    resolution defines what the harness must match.
    """
    rerender = SRC / "scripts" / "rerender.py"
    width = module_constant(rerender, "CAM_WIDTH")
    height = module_constant(rerender, "CAM_HEIGHT")

    assert (width, height) == (config.TRAIN_WIDTH, config.TRAIN_HEIGHT)


def test_build_state_is_identical_across_modules():
    """
    collect.build_state and rollout.build_state must produce the same
    layout.

    Any divergence feeds the policy a differently-ordered observation at
    evaluation than it saw at training, and the symptom is a policy that
    trains cleanly and evaluates at zero with no error anywhere.
    """
    collect_src = function_source(SRC / "scripts" / "collect.py", "build_state")
    rollout_src = function_source(SRC / "eval" / "rollout.py", "build_state")

    assert collect_src is not None
    assert rollout_src is not None

    # Compare the return expression only; docstrings legitimately differ.
    def return_expr(src):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                return ast.unparse(node)
        return None

    assert return_expr(collect_src) == return_expr(rollout_src)


@pytest.mark.parametrize(
    "module",
    [
        "scripts/collect.py",
        "eval/rollout.py",
        "scripts/rerender.py",
    ],
)
@pytest.mark.parametrize(
    "name,expected",
    [
        ("N_ARM", None),
        ("KP_TRANS", None),
        ("KP_ROT", None),
        ("ZETA", None),
        ("GRIP_KP", None),
        ("GRIP_KD", None),
    ],
)
def test_shared_constants_match_config(module, name, expected):
    """
    Modules that redefine a constant already in config must agree with it.

    Duplication is tolerated here rather than forbidden, because a module
    may legitimately want a local override. What is not tolerated is a
    silent disagreement: collection, re-rendering and evaluation must all
    drive the arm with identical gains or the replayed and evaluated
    physics diverge from the demonstrations.
    """
    path = SRC / module
    local = module_constant(path, name)

    if local is None:
        pytest.skip(f"{module} does not define {name}")

    assert local == getattr(config, name), (
        f"{module} defines {name}={local}, config says {getattr(config, name)}"
    )


def test_block_addresses_match_across_modules():
    """
    The block's free joint qpos and qvel addresses must agree everywhere.

    Writing a block pose to the wrong offset corrupts arm joint angles
    instead, which shows up as an arm that starts in a strange
    configuration rather than as an obvious indexing error.
    """
    for module in ["scripts/collect.py", "eval/rollout.py", "scripts/rerender.py"]:
        path = SRC / module
        qpos = module_constant(path, "BLOCK_QPOS_ADR")
        qvel = module_constant(path, "BLOCK_QVEL_ADR")

        if qpos is not None:
            assert qpos == config.BLOCK_QPOS_ADR, f"{module} BLOCK_QPOS_ADR"
        if qvel is not None:
            assert qvel == config.BLOCK_QVEL_ADR, f"{module} BLOCK_QVEL_ADR"


def test_no_absolute_paths_remain():
    """
    No source file may hardcode a machine-specific path.

    The project ran from exactly one directory on one computer until these
    were removed, which made it unreproducible for anyone else.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "C:\\pick_place" in text or "C:/pick_place" in text:
            offenders.append(str(path.relative_to(SRC)))

    assert not offenders, f"absolute paths in: {', '.join(offenders)}"


def test_no_sys_path_manipulation():
    """
    No source file may modify sys.path.

    Path insertion was how scripts found their own package before the
    project became installable. Leaving it in place means the import that
    actually resolves depends on the working directory.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "sys.path.insert" in text or "sys.path.append" in text:
            offenders.append(str(path.relative_to(SRC)))

    assert not offenders, f"sys.path manipulation in: {', '.join(offenders)}"


def test_supervisor_only_ever_writes_the_gripper_channel():
    """
    The runtime supervisor may withhold a gripper close and it may flush the
    policy's action chunk. It may not issue motion.

    That constraint is the whole basis on which the supervised result is
    still a statement about a learned policy rather than about a
    hand-written state machine with a policy attached, and a docstring
    promising it is worth nothing once somebody needs the arm to back off
    by two centimetres and it is one line to do it. So it is checked
    mechanically: every subscripted assignment to the action vector in
    supervisor.py must target index 7, the gripper.

    action = target_pos[0:3], target_quat[3:7], gripper[7]
    """
    path = SRC / "eval" / "supervisor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    writes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not (isinstance(target.value, ast.Name)
                    and target.value.id == "action"):
                continue
            writes.append(ast.unparse(target))

    assert writes, "no assignment to the action vector found; has it been renamed?"
    for w in writes:
        assert w == "action[7]", (
            f"supervisor.py assigns to {w}. Only action[7], the gripper, may "
            "be written: every target pose the arm is driven to has to be the "
            "policy's own output."
        )


def test_supervisor_thresholds_separate_the_measured_populations():
    """
    The veto and detector thresholds sit between two populations measured
    over 200 monitor-mode trials, and the gaps are wide but not infinite.
    Nudging a constant to "be safer" in one direction walks it into the
    other population, and the symptom is not an error: it is a policy that
    refuses grasps that would have worked, or accepts closes on air.

    Measured, at the first close:

        passed strict (n=194)   height -3.27 to +1.55 mm
                                finger settle 29.90 to 33.94 mm
        closed on air (n=4)     height +20.74 to +30.71 mm
                                finger settle 15.35 to 19.11 mm
    """
    path = SRC / "eval" / "supervisor.py"

    grasp_max = module_constant(path, "GRASP_MAX_ABOVE_M")
    finger_min = module_constant(path, "FINGER_HOLD_MIN_M")
    stall = module_constant(path, "STALL_STEPS")

    assert 0.00155 < grasp_max < 0.02074, (
        f"GRASP_MAX_ABOVE_M={grasp_max} is outside the measured gap. Above "
        "+1.55 mm or it refuses grasps that worked; below +20.74 mm or it "
        "admits the air closes it exists to stop."
    )
    assert 0.01911 < finger_min < 0.02990, (
        f"FINGER_HOLD_MIN_M={finger_min} is outside the measured gap between "
        "air closes (max 19.11 mm) and real grasps (min 29.90 mm)."
    )
    # Every one of the 194 passing trials had closed by step 125.
    assert stall > 125, (
        f"STALL_STEPS={stall} would flush the plan of a grasp that was still "
        "going to succeed; the latest observed passing grasp was step 125."
    )


def test_recovery_allowance_is_one_measured_approach():
    """
    The episode-length band is widened per recovery, and the widening is a
    measured phase duration rather than a number chosen to make a result
    pass. It must equal the demonstrations' mean approach, because that is
    what a recovery repeats.
    """
    strict = SRC / "eval" / "strict.py"
    allowance = module_constant(strict, "RECOVERY_ALLOWANCE_STEPS")

    from src.rl import reference as REF

    assert allowance == int(REF.APPROACH_STEPS[0]), (
        f"RECOVERY_ALLOWANCE_STEPS={allowance} no longer equals the measured "
        f"mean approach of {int(REF.APPROACH_STEPS[0])} steps."
    )
