"""
A runtime layer that refuses physically impossible gripper closes, notices
when a grasp has failed, and — measured over 1,000 trials — changes the
success rate by exactly nothing.

The negative result is the reason this file is still here, along with the
instrumentation that produced it. Read the whole docstring before extending
it, because the obvious extensions are already ruled out.

## What it was built to fix

`PATH_TO_97.md` S17 diagnosed the ensemble's residual 3% as one
reproducible mode: every failure closed the fingers at +21.6 mm above the
block's centre, level with its top face, against -1.6 mm for every success,
and the fingers settled at 8 to 18 mm instead of the 30 mm a held block
holds them at. That was read as a **timing** failure — ACT predicts 32
actions from one observation and executes them open loop, so a close
predicted from an observation taken at hover height fires while the arm is
still high and nothing can correct it until the chunk ends.

On that reading, the fix is to stop honouring that one command. Hence the
rules below.

## What actually happens, measured

The reading was wrong, and monitor mode is what showed it. In a failing
trial the policy **commands the descent correctly** — target at -3.5 mm
relative to the block, held for 15 consecutive steps — while the tool sits
at +21.9 mm moving 0.01 to 0.08 mm per step, against 1.2 to 1.5 mm per step
in a free descent. The commanded orientation is tracked to 0.4 degrees and
the wrist joints are nowhere near their limits.

The arm is not mistimed. It is **jammed**, resting a fingertip on the
block's top face, because the commanded gripper yaw is 12 to 18 degrees off
the block's alignment against 1.5 to 2.2 degrees in a success, and open
fingers cannot straddle a cube at that angle.

And the yaw error has a cause that no runtime layer can reach. A cube is
symmetric every 90 degrees, so the oracle's commanded gripper yaw is a
**sawtooth** in the block's yaw: two blocks at 44 and 46 degrees look nearly
identical and their absolute yaw labels differ by 88 degrees. An
L1-regressed policy smears across that discontinuity. Failures therefore
cluster monotonically toward the symmetry boundary:

    block yaw offset from the nearest face alignment    failure rate
        0 to 15 degrees                                  0 of 57
       15 to 30 degrees                                  0 of 65
       30 to 40 degrees                                  5.6%
       40 to 45 degrees                                 12.5%

## Why the rules cannot help, and the numbers

    configuration                      strict, 200 trials
    monitor (the baseline)                  97.00%
    veto, threshold 5 mm                    97.00%
    veto, threshold 8 mm                    97.00%
    veto, threshold 12 mm                   97.00%
    retry                                   97.00%

Every configuration fails **the same six trials**, which is what it looks
like when the outcome is fixed by the block's initial pose before the
episode starts. The veto fired on exactly the right trials — 36 refused
closes, up to 29 in a single episode — and refusing a close does not
unjam an arm. The retry fired on exactly the right four trials and not one
recovered, because reopening the fingers and asking the same policy to
replan from the same observation returns the same averaged yaw.

The fix is a better representation of the orientation target, which is a
training change and not a runtime one. See docs/PATH_TO_99.md.

## What the layer is allowed to do, and why that is enforced in a test

This layer may **withhold a close** and it may **flush the action chunk**.
That is all. Every target pose the arm is driven to is the policy's own
output, unmodified: there is no scripted approach, no hand-written descent
and no waypoint here. The constraint is the point — a layer that issued its
own motion commands would turn a statement about a learned policy into a
statement about a state machine with a policy attached — and a docstring
promising it is worth nothing once somebody needs the arm to back off by
two centimetres and it is one line to do it. So
`tests/test_invariants.py::test_supervisor_only_ever_writes_the_gripper_channel`
parses this file and fails if anything but `action[7]` is ever assigned.

The flush needs its cooldown, and the reason is measured. Clearing the
chunk every step is exactly `n_action_steps=1`, which this project measured
at 32.5% against 92.5%. A chunked policy has to be allowed to execute its
plan, so a flush is an event and not a policy: it happens when a close is
refused or a grasp is found to have failed, and then not again for
`FLUSH_COOLDOWN` steps.

## Monitor mode is the useful part

With veto and retry both False the inner policy's action is returned byte
for byte, and the wrapper becomes the instrument that measured everything
above: it reproduces the headline at 97.00% over 200 trials while recording
every gripper close, the height and lateral offset it happened at, the
finger width twelve steps later, where a descent was begun from, whether it
jammed, and how far apart the ensemble members' commanded positions were.
Every threshold in this file was set from those distributions over hundreds
of trials rather than from the three failures that motivated the file, and
two constants in the older diagnostic were wrong by that standard.
"""

import numpy as np

from src.data.task import BLOCK_Z

# The gripper is binary in the data and the evaluation calls anything under
# this closed. Same constant the ensemble and the gates use.
GRIP_THRESHOLD = 0.02
GRIP_OPEN = 0.04

# Block geometry, from the scene. A 44 mm cube resting on the table has its
# centre at BLOCK_Z and its top face 22 mm above that.
BLOCK_HALF_M = 0.022

# How far above the block's centre the tool may be and still be making a
# credible grasp.
#
# Measured in monitor mode over 200 trials on seeds already spent on
# screening, so no reporting seed informed it. The separation is not
# marginal:
#
#     first close, 194 trials that passed strict
#         min -3.27 mm, median -0.60, p95 +0.64, max **+1.55**
#     first close, the four trials that closed on air
#         **+20.74**, +20.76, +20.97, +30.71 mm
#
# The tool sits a millimetre or two *below* the block's centre on every
# single successful grasp, so the fingers straddle it. Every air close is
# level with the top face, 22 mm up. Nothing lands between +1.55 and
# +20.74.
#
# One further trial captured the block from +8.94 mm and then failed on the
# retreat, which is the only evidence in 200 trials about how high a grasp
# can still succeed from. Twelve millimetres is the most centred threshold
# in the gap that matters: 10.4 mm above the worst passing close, 8.7 mm
# below the tightest air close, and above the one demonstrated high
# capture. The value was screened at 5, 8 and 12 mm before being fixed
# here; see docs/PATH_TO_99.md.
GRASP_MAX_ABOVE_M = 0.012

# And a floor, because a tool below the block's centre by more than the
# block's half height is inside the table rather than around the block.
GRASP_MIN_ABOVE_M = -0.030

# Where the fingers settle when they have something.
#
# Measured over the same 200 trials, twelve steps after the close:
#
#     closed on the block, n=195    min **29.90** mm, median 30.36, max 33.94
#     closed on air,       n=4      15.35, 17.79, 19.03, **19.11** mm
#
# Twenty-four millimetres sits 4.9 mm clear of both edges of a 10.8 mm gap.
#
# `diagnose_failures.py` reasoned to 12 mm from the geometry — a pair of
# fingers on a 44 mm block "settles near 22 mm each and a pair closed on
# air goes to nearly zero". The geometry is right and the number is wrong:
# air closes stop at 15 to 19 mm, not near zero, because the fingers catch
# the block's top edge on the way past. That constant would have called
# every one of these four air closes a successful grasp.
FINGER_HOLD_MIN_M = 0.024

# How long to wait after a close before believing the finger width.
#
# Also measured, and it cannot be much shorter. A grasp is resolved almost
# at once: one step after the close the fingers are already at 29.9 mm or
# wider. An air close is slow, because the fingers are travelling the whole
# way to nearly shut and catch an edge en route — at 8 steps the widest was
# still 33.5 mm, indistinguishable from a grasp. By 12 steps the widest air
# close is 19.11 mm and the narrowest grasp 29.90 mm, which is the first
# point where the two populations separate cleanly.
SETTLE_WAIT_STEPS = 12

# How near grasp height a commanded target has to be to count as the start
# of the descent. Twelve millimetres above the block's centre is below the
# hover the policy approaches at and above anything the fingers can touch,
# so it brackets the last correctable moment.
DESCENT_ONSET_M = 0.012

# What counts as a jammed descent: the tool above this height relative to
# the block, moving slower than this per step, while a descent is commanded.
# A free descent runs at 1.2 to 1.5 mm per step and a jammed one at 0.01 to
# 0.08, so 0.3 separates them by a factor of four either way. The height
# floor keeps a grasp that has genuinely arrived from being called a jam.
JAM_HEIGHT_M = 0.010
JAM_RATE_MM = 0.3

# When to give up waiting for a grasp and force a fresh plan.
#
# One trial in the 200 never commanded a close at all: 600 steps, no grasp
# frame, the block nudged 3.4 mm and abandoned. Neither rule above can see
# that, because both are triggered by a close, and this is the absence of
# one. So there is a third rule, and it is the weakest action available:
# flush the chunk so the policy has to look again.
#
# Every one of the 194 passing trials had closed by step 125. A hundred and
# sixty is 35 steps past the latest grasp that has ever worked, so a flush
# at that point cannot interrupt a grasp that was about to happen.
STALL_STEPS = 160

# Steps to hold the fingers open after a failed grasp, so they are actually
# open before the policy tries again.
REOPEN_STEPS = 4

# Steps between chunk flushes. One ACT chunk is 32 actions, and flushing
# every step is the configuration this project measured at 32.5%.
FLUSH_COOLDOWN = 24

# How many recoveries one episode may attempt. Three attempts at an
# independent 3% failure rate is 2.7 in a hundred thousand, which is well
# past the point where the cap stops being what limits the result.
MAX_RECOVERIES = 3


class SupervisedPolicy:
    """
    Wraps a policy with the close veto and the grasp-failure detector.

    Presents the same callable interface the rollout harness expects, so it
    is interchangeable with a single policy or an ensemble and needs no
    harness changes.
    """

    def __init__(self, policy, veto=True, retry=True,
                 grasp_max_above_m=GRASP_MAX_ABOVE_M,
                 finger_hold_min_m=FINGER_HOLD_MIN_M,
                 settle_wait_steps=SETTLE_WAIT_STEPS,
                 max_recoveries=MAX_RECOVERIES,
                 stall_steps=STALL_STEPS):
        """
        input:  policy (callable) the learned policy or ensemble,
                veto (bool) refuse closes away from grasp height,
                retry (bool) reopen and replan after a failed grasp, and
                    flush a stalled approach,
                grasp_max_above_m (float) veto threshold,
                finger_hold_min_m (float) width that counts as holding,
                settle_wait_steps (int) steps to wait before reading width,
                max_recoveries (int) recoveries allowed per episode,
                stall_steps (int) steps without a grasp before replanning
        output: SupervisedPolicy instance

        With veto and retry both False this is a measurement instrument and
        returns the inner policy's action unchanged.
        """
        self.policy = policy
        self.veto = bool(veto)
        self.retry = bool(retry)
        self.grasp_max_above_m = float(grasp_max_above_m)
        self.finger_hold_min_m = float(finger_hold_min_m)
        self.settle_wait_steps = int(settle_wait_steps)
        self.max_recoveries = int(max_recoveries)
        self.stall_steps = int(stall_steps)

        self.model = None
        self.data = None
        self._ee_kind = None
        self._ee_id = None
        self._finger_dofs = None
        self._block_qpos_adr = None

        self.reset()

    # ------------------------------------------------------------ setup --

    def bind(self, model, data):
        """
        Takes the simulator handles and resolves the frames it reads.

        The tool pose and the finger widths are proprioception: a real arm
        reports both from its own encoders. The block's pose is resolved
        too, but only so monitor mode can record the truth alongside the
        proprioceptive estimate the rules actually use. No decision in this
        file reads it.

        input:  model (MjModel), data (MjData)
        output: None
        """
        import mujoco

        from src.config import BLOCK_QPOS_ADR, FINGER_DOFS
        from src.controllers.impedance import resolve_ee_frame

        self.model = model
        self.data = data
        self._ee_kind, self._ee_id = resolve_ee_frame(model, verbose=False)
        self._finger_dofs = list(FINGER_DOFS)
        self._block_qpos_adr = BLOCK_QPOS_ADR
        self._block_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "block"
        )

        if hasattr(self.policy, "bind"):
            self.policy.bind(model, data)

    def reset(self):
        """
        Clears per-episode state, including the inner policy's action queue.

        input:  none
        output: None
        """
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        self.step = 0
        self.confirmed_grasp = False
        self.pending_close = None
        self.force_open_until = -1
        self.flush_ok_at = 0
        self.descent_onset = None
        self.jam_steps = 0
        self.jam_event = None
        self._last_above = None
        self.spread_trace = []

        self.n_vetoes = 0
        self.n_recoveries = 0
        self.n_stall_flushes = 0
        self.close_events = []
        self.veto_events = []

    # ------------------------------------------------------- reads ------

    def _tool_pos(self):
        """
        Reads the tool position from the model, as the controller does.

        input:  none
        output: numpy array of shape (3,)
        """
        if self._ee_kind == "site":
            return np.asarray(self.data.site_xpos[self._ee_id], dtype=float)
        return np.asarray(self.data.xpos[self._ee_id], dtype=float)

    def _finger_width(self):
        """
        Mean of the two finger joint positions, in metres.

        input:  none
        output: float
        """
        return float(np.mean(self.data.qpos[self._finger_dofs]))

    def _member_spread(self):
        """
        How far apart the ensemble members' commanded positions are, in mm.

        An uncertainty signal that costs nothing and needs no perception:
        the members are two independently trained policies, so where they
        disagree is where neither is confident. Recorded for analysis. No
        rule reads it.

        input:  none
        output: float, or nan when the inner policy is a single model
        """
        members = getattr(self.policy, "last_member_actions", None)
        if members is None or len(members) < 2:
            return float("nan")
        pos = members[:, :3]
        return float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).max() * 1000.0)

    def _yaw_errors(self, action):
        """
        The commanded wrist yaw error against the block, and the block's own
        yaw offset from the nearest face alignment.

        Both folded into a quarter turn, because a cube only needs the
        fingers square to a pair of faces. The second quantity is also, and
        this is the point of recording it, how far the wrist has to rotate
        from the home pose it starts every episode in. Recorded for
        analysis; no rule reads either.

        input:  action (array (8,)) the commanded action
        output: (commanded yaw error deg, block yaw offset deg)
        """
        q = action[3:7] / max(np.linalg.norm(action[3:7]), 1e-9)
        bq = np.asarray(self.data.xquat[self._block_body], dtype=np.float64)
        byaw = 2.0 * np.arctan2(bq[3], bq[0])
        gyaw = 2.0 * np.arctan2(q[2], q[1])

        e = (gyaw - byaw) % (np.pi / 2.0)
        err = np.degrees(min(e, np.pi / 2.0 - e))

        o = byaw % (np.pi / 2.0)
        off = np.degrees(min(o, np.pi / 2.0 - o))
        return float(err), float(off)

    def _block_pos(self):
        """
        True block position. Recorded for analysis, never used to decide.

        input:  none
        output: numpy array of shape (3,)
        """
        return np.asarray(self.data.xpos[self._block_body], dtype=float)

    def _flush(self):
        """
        Clears the inner policy's action chunk if the cooldown has expired.

        input:  none
        output: bool, whether the flush happened
        """
        if self.step < self.flush_ok_at:
            return False
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.flush_ok_at = self.step + FLUSH_COOLDOWN
        return True

    # -------------------------------------------------------- the rules --

    def __call__(self, obs):
        """
        Returns the action the arm is driven with.

        input:  obs (dict) harness observation
        output: numpy array of shape (8,)
        """
        action = np.asarray(self.policy(obs), dtype=np.float64).copy()

        if self.model is None:
            # Unbound: nothing to read, so nothing can be judged.
            self.step += 1
            return action

        tool = self._tool_pos()
        width = self._finger_width()
        spread = self._member_spread()
        cmd_yaw_err, block_offset = self._yaw_errors(action)
        if not self.confirmed_grasp:
            self.spread_trace.append(spread)

        # Height of the tool above where a resting block's centre is. The
        # block sits on the table until it is picked up, so this is forward
        # kinematics against a known table, not perception.
        above_block = float(tool[2] - BLOCK_Z)

        # --- resolve a close that is waiting to be judged ----------------

        if self.pending_close is not None:
            age = self.step - self.pending_close["step"]
            self.pending_close["widths"].append(round(width * 1000.0, 3))

            if age >= self.settle_wait_steps:
                ev = self.pending_close
                ev["settled_width_mm"] = width * 1000.0
                ev["gripped"] = bool(width >= self.finger_hold_min_m)
                ev["lift_mm"] = float(
                    (self._block_pos()[2] - BLOCK_Z) * 1000.0
                )
                self.close_events.append(ev)
                self.pending_close = None

                if ev["gripped"]:
                    # The fingers are around something of the right width.
                    # From here a close command is the grasp being held, so
                    # the veto disarms: vetoing during the carry would drop
                    # the block.
                    self.confirmed_grasp = True
                elif self.retry and self.n_recoveries < self.max_recoveries:
                    self.n_recoveries += 1
                    self.force_open_until = self.step + REOPEN_STEPS
                    ev["recovery"] = True
                    self._flush()

        # --- record where the descent was begun from ----------------------
        #
        # The moment the policy first commands a target near grasp height is
        # the last moment before the fingers can touch anything, and it is
        # therefore the only moment at which a misaligned approach is still
        # correctable. Lateral offset measured at the *close* does not
        # separate passing from failing trials, because by then a failing
        # trial has already shoved the block. Measured here, before contact,
        # it may. Recorded in every mode; no rule below reads it.

        if self.descent_onset is None and not self.confirmed_grasp:
            target_above = float(action[2] - BLOCK_Z)
            if target_above < DESCENT_ONSET_M:
                block = self._block_pos()
                self.descent_onset = {
                    "step": int(self.step),
                    "target_above_block_mm": target_above * 1000.0,
                    "tool_above_block_mm": above_block * 1000.0,
                    "tool_block_xy_mm": float(
                        np.linalg.norm(tool[:2] - block[:2]) * 1000.0
                    ),
                    "member_spread_mm": spread,
                    "cmd_yaw_err_deg": cmd_yaw_err,
                    "block_yaw_offset_deg": block_offset,
                }

        # --- a jammed descent --------------------------------------------
        #
        # The commanded target is at grasp height and the tool is not
        # getting there. Measured on a failing trial: the tool sits at
        # +21.9 mm descending at 0.01 to 0.08 mm per step while the command
        # is 25 mm below it, against 1.2 to 1.5 mm per step in a free
        # descent. That is contact, not sluggishness, and it is visible
        # without any perception: a commanded height, an achieved height
        # from forward kinematics, and the difference between them.

        if not self.confirmed_grasp:
            target_above = float(action[2] - BLOCK_Z)
            if target_above < DESCENT_ONSET_M and above_block > JAM_HEIGHT_M:
                moved = (
                    abs(above_block - self._last_above) * 1000.0
                    if self._last_above is not None else None
                )
                if moved is not None and moved < JAM_RATE_MM:
                    self.jam_steps += 1
                    if self.jam_steps == 1:
                        self.jam_event = {
                            "step": int(self.step),
                            "tool_above_block_mm": above_block * 1000.0,
                            "target_above_block_mm": target_above * 1000.0,
                            "tool_block_xy_mm": float(
                                np.linalg.norm(tool[:2] - self._block_pos()[:2])
                                * 1000.0
                            ),
                            "member_spread_mm": spread,
                        }
        self._last_above = above_block

        # --- a stalled approach ------------------------------------------
        #
        # No grasp long after every grasp that has ever worked. Force a
        # fresh observation and nothing else; there is no motion command
        # here, only the removal of a stale plan.

        if (self.retry and not self.confirmed_grasp
                and self.step >= self.stall_steps
                and self.pending_close is None):
            if self._flush():
                self.n_stall_flushes += 1

        # --- hold the fingers open through a reopen ----------------------

        if self.step < self.force_open_until:
            action[7] = GRIP_OPEN
            self.step += 1
            return action

        # --- the close veto ----------------------------------------------

        closing = bool(action[7] < GRIP_THRESHOLD)

        if closing and not self.confirmed_grasp:
            plausible = (
                self.grasp_max_above_m >= above_block >= GRASP_MIN_ABOVE_M
            )

            if self.veto and not plausible:
                self.veto_events.append({
                    "step": int(self.step),
                    "above_block_mm": above_block * 1000.0,
                    "tool_block_xy_mm": float(
                        np.linalg.norm(tool[:2] - self._block_pos()[:2])
                        * 1000.0
                    ),
                })
                self.n_vetoes += 1
                action[7] = GRIP_OPEN
                self._flush()
                self.step += 1
                return action

            if self.pending_close is None:
                # An honoured close that has not been judged yet.
                self.pending_close = {
                    "step": int(self.step),
                    "above_block_mm": above_block * 1000.0,
                    "tool_block_xy_mm": float(
                        np.linalg.norm(tool[:2] - self._block_pos()[:2])
                        * 1000.0
                    ),
                    "widths": [],
                    "recovery": False,
                    "cmd_yaw_err_deg": cmd_yaw_err,
                    "block_yaw_offset_deg": block_offset,
                }

        self.step += 1
        return action

    # ------------------------------------------------------- reporting --

    @property
    def recovery_events(self):
        """
        How many times this episode reopened after a failed grasp.

        The strict evaluation reads this to widen the episode-length band,
        because a recovery makes the episode longer than any demonstration
        and the untouched band would score that as a failure of duration
        rather than reporting what it is.

        input:  none
        output: int
        """
        return int(self.n_recoveries)

    def diagnostics(self):
        """
        Everything the layer saw this episode.

        input:  none
        output: dict
        """
        return {
            "vetoes": int(self.n_vetoes),
            "recoveries": int(self.n_recoveries),
            "stall_flushes": int(self.n_stall_flushes),
            "descent_onset": self.descent_onset,
            "jam_steps": int(self.jam_steps),
            "jam_event": self.jam_event,
            "member_spread_max_mm": (
                float(np.nanmax(self.spread_trace)) if self.spread_trace
                else float("nan")
            ),
            "member_spread_median_mm": (
                float(np.nanmedian(self.spread_trace)) if self.spread_trace
                else float("nan")
            ),
            "confirmed_grasp": bool(self.confirmed_grasp),
            "close_events": list(self.close_events),
            "veto_events": list(self.veto_events),
        }
