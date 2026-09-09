"""
Behaviour-clones SAC's actor on the oracle data before reinforcement
learning starts.

RLPD seeds the critic well. Half of every gradient batch comes from oracle
transitions, so the value function learns quickly what a completed episode
is worth. It leaves the actor to discover the behaviour on its own, and on
this task that is the part that fails: the completed sequence is a chain of
roughly 280 steps through approach, grasp, lift, carry, low descent, gentle
release, vertical lift-off and return to home, and the terminal bonus pays
only if every link holds. Runs v9 and v10 explored for hundreds of
thousands of steps without ever completing the chain once.

So the actor is given the chain directly. Supervised regression of the
oracle's actions onto its observations puts the policy inside the right
behaviour before a single environment step is taken, and reinforcement
learning then has something to improve rather than something to find.

Two details that matter.

SAC's actor is squashed: it outputs a pre-tanh mean which passes through
tanh into the action range. Regressing on the raw mean would fight the
squash near the limits, where oracle actions frequently sit because the
gripper channel is binary. The target is therefore inverse-tanh transformed
and clipped short of the asymptote.

Only the actor is pretrained. The critics are left alone because they will
be trained on real returns from the demonstration buffer anyway, and a
critic fitted to actions it has no value estimate for is worse than an
untrained one.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Actions land on the boundary often, and atanh diverges there. Clipping
# just short keeps the regression target finite without meaningfully
# changing what is being asked for.
ATANH_LIMIT = 0.999


def _pre_tanh(actions):
    """
    Maps actions in -1 to 1 back through the actor's tanh squash.

    input:  actions (Tensor) in -1 to 1
    output: Tensor, pre-squash values
    """
    return torch.atanh(torch.clamp(actions, -ATANH_LIMIT, ATANH_LIMIT))


def pretrain_actor(model, demo, epochs=8, batch_size=512, lr=3e-4,
                   verbose=True):
    """
    Fits the actor's mean output to the oracle's actions.

    input:  model (SAC), demo (ReplayBuffer) holding oracle transitions,
            epochs (int) passes over the demo data, batch_size (int),
            lr (float), verbose (bool)
    output: dict with the final loss and the number of samples seen
    """
    n = demo.size() * demo.n_envs
    if n == 0:
        raise ValueError("demo buffer is empty, nothing to pretrain on")

    actor = model.actor
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    batches = max(1, n // batch_size)
    last = float("nan")

    for epoch in range(epochs):
        total = 0.0
        for _ in range(batches):
            batch = demo.sample(batch_size)
            obs = batch.observations
            target = _pre_tanh(batch.actions)

            # The actor returns (mean, log_std, kwargs) before squashing.
            mean, _, _ = actor.get_action_dist_params(obs)
            loss = F.mse_loss(mean, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())

        last = total / batches
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}  mse {last:.4f}", flush=True)

    return {"loss": last, "samples": n, "batches": batches * epochs}


def evaluate_actor(model, env, episodes=10):
    """
    Runs the pretrained actor to see how much of the chain it reproduces
    before any reinforcement learning.

    This is worth checking on its own. If behaviour cloning alone completes
    the sequence most of the time, reinforcement learning is starting from
    a good place and the remaining work is refinement. If it does not, the
    pretraining did not take and no amount of fine-tuning will rescue it.

    input:  model (SAC), env (FR3PickPlaceEnv), episodes (int)
    output: dict of rates
    """
    ok = retreated = placed = 0
    lengths = []

    for _ in range(episodes):
        obs, _ = env.reset()
        steps, ever_placed = 0, False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            steps += 1
            # `placed` is only true at the moment of release, because it
            # requires the tool to still be low. Reading it at the terminal
            # step reports 0% for a policy that placed the block perfectly
            # and then correctly lifted clear.
            ever_placed = ever_placed or info["placed"]
            if term or trunc:
                break
        ok += int(info["success"])
        placed += int(ever_placed)
        retreated += int(info["retreated"])
        lengths.append(steps)

    return {
        "success": ok / episodes,
        "placed": placed / episodes,
        "retreated": retreated / episodes,
        "mean_steps": float(np.mean(lengths)),
    }
