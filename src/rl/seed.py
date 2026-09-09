"""
Fills SAC's replay buffer with scripted oracle rollouts before training.

The original Stage 2 plan called for this and it was skipped, which cost a
lot. Without it the teacher had to discover the task from scratch, found
whatever the reward failed to forbid, and needed four reward revisions to be
argued out of it.

The oracle already performs the trajectory the demonstrations show, at the
speed they show, with the phase structure they show. Putting a few hundred
of its episodes into the buffer means SAC starts with good behaviour in
reach instead of having to stumble on it, and the shaping terms mostly have
to keep the policy near that behaviour rather than invent it.

The oracle is driven through the environment rather than replayed from a
file, so the transitions carry this environment's exact observations,
rewards and termination. Data recorded any other way would be describing a
slightly different problem.

Envs are stepped in lockstep in this process rather than through
SubprocVecEnv, because the buffer stores transitions shaped (n_envs, ...)
and lockstep produces exactly that shape with no reshaping games.
"""

import numpy as np

from src.eval.oracle import ScriptedOracle, quat_yaw, wrap
from src.rl.env import MAX_DELTA_M, MAX_DYAW_RAD, FR3PickPlaceEnv


def oracle_action(env, oracle):
    """
    Converts the oracle's absolute target pose into the environment's delta
    action.

    input:  env (FR3PickPlaceEnv), oracle (ScriptedOracle)
    output: numpy array of shape (5,) float32
    """
    want = oracle(None)
    dpos = (want[:3] - env.target_pos) / MAX_DELTA_M
    dyaw = wrap(quat_yaw(want[3:7]) - env.target_yaw) / MAX_DYAW_RAD
    grip = -1.0 if want[7] < 0.02 else 1.0
    return np.clip(
        np.concatenate([dpos, [dyaw], [grip]]), -1.0, 1.0
    ).astype(np.float32)


def fill(model, n_envs, steps, seed=900, verbose=True):
    """
    Runs the oracle in lockstep across n_envs environments and writes every
    transition into the model's replay buffer.

    input:  model (SAC), n_envs (int) must match the buffer's n_envs,
            steps (int) lockstep steps to collect, seed (int),
            verbose (bool)
    output: dict with transitions written and episodes completed
    """
    envs = [FR3PickPlaceEnv(seed=seed + i) for i in range(n_envs)]
    oracles = [ScriptedOracle() for _ in range(n_envs)]

    obs = []
    for env, oracle in zip(envs, oracles, strict=True):
        oracle.bind(env.model, env.data)
        o, _ = env.reset()
        oracle.reset()
        obs.append(o)
    obs = np.array(obs)

    episodes, successes = 0, 0

    for k in range(steps):
        actions = np.array([oracle_action(e, o) for e, o in zip(envs, oracles, strict=True)])

        # cur_obs is what the action was actually taken from and is what the
        # buffer must store. It has to be captured before any environment
        # resets, or the transition records an observation from the next
        # episode entirely.
        cur_obs = obs.copy()
        next_obs = np.empty_like(cur_obs)
        rewards, dones, infos = [], [], []

        for i, (env, oracle) in enumerate(zip(envs, oracles, strict=True)):
            o2, r, term, trunc, info = env.step(actions[i])
            next_obs[i] = o2

            if term or trunc:
                episodes += 1
                successes += int(info.get("retreated", False))
                info = dict(info)
                # SB3 reads the pre-reset observation from here, and treats
                # a timeout as non-terminal so the value function still
                # bootstraps past it.
                info["terminal_observation"] = o2.copy()
                if trunc and not term:
                    info["TimeLimit.truncated"] = True
                o2, _ = env.reset()
                oracle.reset()

            rewards.append(r)
            dones.append(bool(term))
            infos.append(info)
            obs[i] = o2

        model.replay_buffer.add(
            cur_obs, next_obs, actions,
            np.array(rewards, dtype=np.float32), np.array(dones), infos,
        )

        if verbose and (k + 1) % 100 == 0:
            print(f"    {k + 1}/{steps} lockstep steps, {episodes} episodes",
                  flush=True)

    return {
        "transitions": steps * n_envs,
        "episodes": episodes,
        "retreated": successes,
    }
