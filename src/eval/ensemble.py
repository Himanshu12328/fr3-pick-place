"""
Averages two trained policies into one.

The reason to try it is specific rather than general. Three configurations
land within 2.8 points of each other on 600 trials, so architecture is not
the binding constraint — but their **regional profiles are near mirror
images**:

    block start x        ResNet18 chunk 32     ResNet34 chunk 32
    x <  0.53                 94.0%                 87.2%
    x >= 0.53                 91.3%                 96.7%

Both were trained on the same 997 episodes with the same uniform block
distribution, so that is not a data-coverage difference. It is two models
that have learned the task differently, which is the only situation where
averaging them can beat either.

Two details matter and both concern the gripper.

The gripper is binary in the data, 0 or 0.04, and the evaluation calls
anything under 0.02 closed. Averaging a closed policy with an open one gives
exactly 0.02, the boundary, which is the worst possible answer: it would
make the grasp and release frames ambiguous precisely when the two models
disagree about them. So the gripper is **not averaged**. When the two agree,
their shared decision is used. When they disagree, the previously commanded
value is held, which is hysteresis rather than a coin flip and keeps the
fingers from chattering at every disagreement.

The pose is averaged componentwise, and the quaternion is renormalised
afterwards. Both models are gripper-down at all times, so the two
quaternions are always close and a linear average followed by
renormalisation is a good approximation to the proper interpolation.
"""

import numpy as np


class EnsemblePolicy:
    """
    Two policies in, one action out.

    Presents the same callable interface the rollout harness expects, so it
    is interchangeable with a single policy and needs no harness changes.
    """

    def __init__(self, policies, grip_threshold=0.02):
        """
        input:  policies (list of callables) each returning an 8-vector,
                grip_threshold (float) below which the gripper counts closed
        output: EnsemblePolicy instance
        """
        if len(policies) < 2:
            raise ValueError("an ensemble needs at least two policies")
        self.policies = policies
        self.grip_threshold = grip_threshold
        self.last_grip = 0.04
        # The members' individual outputs from the most recent call, kept so
        # a diagnostic can ask what they disagreed about without calling
        # them again and advancing their action queues twice.
        self.last_member_actions = None

    def reset(self):
        """
        Clears every member's action queue and the gripper hysteresis.

        Chunked policies buffer predicted actions and pop one per call, so
        without this the first steps of a trial execute leftovers from the
        previous one.

        input:  none
        output: None
        """
        for p in self.policies:
            if hasattr(p, "reset"):
                p.reset()
        self.last_grip = 0.04

    def bind(self, model, data):
        """
        Passes simulator handles to any member that wants them.

        input:  model (MjModel), data (MjData)
        output: None
        """
        for p in self.policies:
            if hasattr(p, "bind"):
                p.bind(model, data)

    def __call__(self, obs):
        """
        Returns the combined action.

        input:  obs (dict) harness observation
        output: numpy array of shape (8,)
        """
        # Truncate to the eight dimensions that are the action.
        #
        # A policy trained with auxiliary supervision emits a wider vector:
        # `to_lerobot --with-block-pose` appends the block's pose to the
        # action so that ACT regresses it alongside the task, and the
        # rollout harness reads action[:3], action[3:7] and action[7] and
        # ignores the rest. Ensembling such a policy with a plain one is a
        # legitimate and useful configuration — measured, they fail on
        # disjoint trials — but np.stack refuses to combine an 8-vector
        # with a 10-vector, and the failure is a ValueError several frames
        # deep rather than anything that names the cause.
        #
        # The auxiliary columns are a training target and never an action,
        # so dropping them here is not a compromise; it is what the harness
        # does with them anyway.
        actions = np.stack([np.asarray(p(obs), dtype=np.float64)[:8]
                            for p in self.policies])
        self.last_member_actions = actions

        pos = actions[:, :3].mean(axis=0)

        quat = actions[:, 3:7].mean(axis=0)
        quat = quat / max(np.linalg.norm(quat), 1e-9)

        # Vote, do not average. See the module docstring: averaging a closed
        # command with an open one lands exactly on the threshold.
        closed = actions[:, 7] < self.grip_threshold
        if closed.all():
            grip = float(actions[:, 7].min())
        elif (~closed).all():
            grip = float(actions[:, 7].max())
        else:
            grip = self.last_grip
        self.last_grip = grip

        return np.concatenate([pos, quat, [grip]])
