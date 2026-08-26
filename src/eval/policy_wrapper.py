"""
Wraps a trained LeRobot policy plus its processor pipelines in the
callable interface the rollout harness expects.

LeRobot 0.4.x separates normalisation from the policy itself. A checkpoint
holds three things: the network weights, a preprocessor pipeline that
turns a raw observation into normalised model input, and a postprocessor
pipeline that turns the model's normalised output back into real units.
Loading only the weights yields a policy whose actions sit in [-1, 1]
normalised space, which looks like a working policy emitting nonsense.

The preprocessor pipeline for this checkpoint runs four steps: rename,
add batch dimension, move to device, normalise. So the observation handed
to it must be unbatched, on CPU, with images as CHW float in 0..1.
Batching or moving to the device here would double-apply those steps.
"""

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

POLICY_CLASSES = {
    "diffusion": DiffusionPolicy,
    "act": ACTPolicy,
}


def load_policy_and_processors(checkpoint_dir, policy_type="diffusion",
                               device="cuda", n_action_steps=None):
    """
    Loads the policy weights and both processor pipelines from a
    checkpoint directory.

    The processors are built from the policy's own config and the saved
    normalisation statistics, so they are guaranteed to match the weights.
    Constructing them separately from hand-written statistics is how
    evaluation silently diverges from training.

    n_action_steps (int or None) override how many steps of each
    predicted chunk are executed before re-planning. Lower values
    re-plan more often, which limits how far the arm can drift on
    a stale plan.

    input:  checkpoint_dir (str) path to the pretrained_model folder,
            policy_type (str) one of the keys in POLICY_CLASSES,
            device (str),
            n_action_steps (int or None)
    output: (policy, preprocessor, postprocessor)
    """
    policy = POLICY_CLASSES[policy_type].from_pretrained(checkpoint_dir)

    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
        print(f"  n_action_steps overridden to {n_action_steps}")

    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=checkpoint_dir)

    print(f"Loaded {policy_type} with processors from {checkpoint_dir}")
    return policy, preprocessor, postprocessor


class LeRobotPolicyAdapter:
    """
    Adapts a LeRobot policy and its processors to the harness's
    policy(obs) -> action(8,) interface.
    """

    def __init__(self, policy, preprocessor, postprocessor,
                 device="cuda", task=None):
        """
        input:  policy (LeRobot policy), preprocessor, postprocessor,
                device (str),
                task (str or None) language instruction for policies that
                condition on one
        output: LeRobotPolicyAdapter instance
        """
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.task = task

    def reset(self):
        """
        Clears the policy's internal observation and action queues between
        trials.

        Chunked policies cache observations and buffer predicted actions,
        popping one per call. Without a reset, the first steps of trial N
        execute leftover actions predicted from trial N-1, which quietly
        corrupts the success rate in a way that looks like poor
        performance rather than a bug.

        input:  none
        output: None
        """
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _raw_batch(self, obs):
        """
        Assembles the observation into the layout the preprocessor expects:
        CHW float images in 0..1, unbatched, on CPU.

        The pipeline adds the batch dimension, moves tensors to the device,
        and normalises. Only the HWC-to-CHW transpose and the 0..255 to
        0..1 scale conversion belong here, because the saved normalisation
        statistics assume images already in 0..1.

        input:  obs (dict) numpy arrays keyed as in the dataset
        output: dict of torch tensors on CPU
        """
        batch = {}

        for key, value in obs.items():
            if key.startswith("observation.images."):
                batch[key] = torch.from_numpy(value).permute(2, 0, 1).float() / 255.0
            else:
                batch[key] = torch.from_numpy(np.asarray(value, dtype=np.float32))

        if self.task is not None:
            batch["task"] = self.task

        return batch

    @torch.no_grad()
    def __call__(self, obs):
        """
        Returns one action in real units for the given observation.

        input:  obs (dict) harness observation
        output: numpy array of shape (8,), target pose and gripper in
                metres and quaternion components
        """
        batch = self.preprocessor(self._raw_batch(obs))
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)

        if isinstance(action, dict):
            action = action["action"]

        action = action.squeeze().detach().cpu().numpy()
        return np.asarray(action, dtype=np.float64)