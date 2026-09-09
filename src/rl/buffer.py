"""
A replay buffer that samples half of every batch from a frozen set of
scripted oracle transitions.

The first attempt at seeding simply wrote oracle transitions into SAC's
normal buffer before training. That does almost nothing, for a reason worth
writing down: the buffer is FIFO with a one million capacity, and the run
that followed pushed 1.2 million fresh transitions through it. The 50,000
oracle transitions were diluted immediately and then **evicted entirely**
somewhere around step 950,000. For most of the run the oracle data the plan
was relying on was not in the buffer at all.

That is not what RLPD does. The recipe keeps the demonstrations in their own
buffer, never evicts them, and draws a fixed share of every gradient batch
from them. The demonstrations stay an anchor for the whole run instead of
being a brief and vanishing head start.

The oracle scores 0.986 against the demonstration trajectory profile, so
holding the policy near it is most of the objective.
"""

import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class MixedReplayBuffer(ReplayBuffer):
    """
    A normal replay buffer that blends in samples from a frozen companion.

    The companion is set after construction because Stable Baselines builds
    the buffer itself from `replay_buffer_class`, so there is no opportunity
    to pass one in.
    """

    demo = None
    demo_ratio = 0.5

    def attach_demo(self, demo, ratio=0.5):
        """
        Supplies the frozen demonstration buffer to draw from.

        input:  demo (ReplayBuffer) already filled, ratio (float) share of
                each batch to take from it
        output: None
        """
        self.demo = demo
        self.demo_ratio = ratio

    def sample(self, batch_size, env=None):
        """
        Returns a batch drawn partly from the online data and partly from
        the frozen demonstrations.

        input:  batch_size (int), env (VecNormalize or None)
        output: ReplayBufferSamples
        """
        if self.demo is None or self.demo.size() == 0:
            return super().sample(batch_size, env)

        n_demo = int(batch_size * self.demo_ratio)
        n_online = batch_size - n_demo
        if n_online <= 0:
            return self.demo.sample(batch_size, env)
        if self.size() == 0:
            return self.demo.sample(batch_size, env)

        online = super().sample(n_online, env)
        demo = self.demo.sample(n_demo, env)

        # Not every field is a tensor. Stable Baselines 2.9 added a
        # `discounts` field that is None unless n-step returns are in use,
        # so concatenating blindly across the tuple fails on it.
        return ReplayBufferSamples(*[
            torch.cat([a, b], dim=0) if a is not None and b is not None else None
            for a, b in zip(online, demo, strict=True)
        ])


def make_demo_buffer(observation_space, action_space, n_envs, transitions, device):
    """
    Builds an empty buffer sized to hold every oracle transition.

    `buffer_size` here is a count of transitions, not of buffer positions.
    Stable Baselines divides it by `n_envs` internally, so dividing again
    before passing it in shrinks the buffer by a factor of n_envs. That is
    not a hypothetical: it silently reduced a 50,000 transition oracle
    buffer to 2,500, which then wrapped and kept only the last nine
    episodes. Half of every gradient batch was drawn from those nine, for
    an entire run.

    input:  observation_space, action_space, n_envs (int),
            transitions (int) total transitions to hold, device (str)
    output: ReplayBuffer
    """
    return ReplayBuffer(
        buffer_size=int(transitions),
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        n_envs=n_envs,
        handle_timeout_termination=True,
    )
