"""
Scores a policy's trajectories against the recorded demonstrations.

Success rate cannot see any of what this measures. A teacher scored 98.8%
solving the task in 73 steps against the demonstrations' 295, carrying the
block 31 mm up against their 80 mm, and never retreating at all. Every one
of those is invisible in a success rate and obvious here.

Prints a table of the measured profile against the demonstration bands,
gives a similarity score in 0 to 1, and plots the speed and height traces
against the demonstration envelope.

Run:
    python -m src.scripts.trajectory_report --oracle
    python -m src.scripts.trajectory_report --run rl_teacher_v4 --model latest
"""

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import DOCS_DIR, LOG_DIR, OUTPUT_ROOT  # noqa: E402
from src.rl import reference as REF  # noqa: E402
from src.rl.env import FR3PickPlaceEnv  # noqa: E402
from src.rl.seed import oracle_action  # noqa: E402


def collect(policy, env, n, oracle=None):
    """
    Runs episodes and records the commanded target and gripper traces.

    input:  policy (SAC or None), env (FR3PickPlaceEnv), n (int),
            oracle (ScriptedOracle or None)
    output: list of dicts with targets, gripper and outcome
    """
    out = []
    for _ in range(n):
        obs, _ = env.reset()
        if oracle is not None:
            oracle.reset()
        targets, grips = [], []

        while True:
            if oracle is not None:
                action = oracle_action(env, oracle)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            targets.append(env.target_pos.copy())
            grips.append(env.grip)
            if term or trunc:
                break

        out.append({
            "targets": np.array(targets),
            "gripper": np.array(grips),
            "success": bool(info["success"]),
            "retreated": bool(info.get("retreated", False)),
        })
    return out


def report(episodes, label):
    """
    Prints the measured profile beside the demonstration bands.

    input:  episodes (list from collect()), label (str)
    output: (mean_score float, list of per-episode measurement dicts)
    """
    rows = [REF.describe(e["targets"], e["gripper"]) for e in episodes]
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"{label}: no episode ever grasped, nothing to score")
        return 0.0, []

    scores = [REF.score(r)[0] for r in rows]

    print(f"\n{label}: {len(rows)} episodes scored")
    print(f"{'metric':<18}{'policy':>12}{'demos':>12}{'demo band':>18}{'score':>8}")
    print("-" * 68)

    per = {}
    for key, (mean, lo, hi), name in REF.METRICS:
        v = float(np.mean([r[key] for r in rows]))
        s = REF.band_score(v, lo, hi)
        per[key] = s
        print(f"{name:<18}{v:>12.2f}{mean:>12.2f}{f'{lo:.0f} to {hi:.0f}':>18}{s:>8.2f}")

    cz = float(np.mean([r["carry_z_max"] for r in rows]))
    rz = float(np.mean([r["retreat_z_max"] for r in rows]))
    jk = float(np.mean([r["jerk"] for r in rows]))
    rel = float(np.mean([r["released"] for r in rows]))
    print(f"{'carry z max':<18}{cz:>12.4f}{REF.CARRY_Z_MAX:>12.4f}"
          f"{'>= 0.479':>18}{REF.band_score(cz, REF.CARRY_Z_MAX - 0.02, 10.0):>8.2f}")
    print(f"{'retreat z max':<18}{rz:>12.4f}{REF.RETREAT_Z_MAX:>12.4f}"
          f"{'>= 0.488':>18}{REF.band_score(rz, REF.RETREAT_Z_MAX - 0.03, 10.0):>8.2f}")
    print(f"{'jerk':<18}{jk:>12.2f}{REF.JERK:>12.2f}"
          f"{'0 to 1.96':>18}{REF.band_score(jk, 0.0, REF.JERK * 2):>8.2f}")

    succ = float(np.mean([e["success"] for e in episodes]))
    retr = float(np.mean([e["retreated"] for e in episodes]))
    print(f"\nTRAJECTORY SCORE {np.mean(scores):.3f}   "
          f"success {succ * 100:.1f}%   retreated {retr * 100:.1f}%   "
          f"released {rel * 100:.1f}%")
    return float(np.mean(scores)), rows


def plot(episodes, path, label):
    """
    Plots speed and height traces against the demonstration envelope.

    input:  episodes (list from collect()), path (Path), label (str)
    output: None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for e in episodes[:15]:
        t = e["targets"]
        speed = np.linalg.norm(np.diff(t, axis=0), axis=1) * 1000.0
        ax1.plot(speed, lw=1.0, color="#0b5566", alpha=0.6)
        ax2.plot(t[:, 2], lw=1.0, color="#0b5566", alpha=0.6)

    for y, c, txt in ((REF.APPROACH_SPEED[0], "#b8781c", "approach 1.85"),
                      (REF.CARRY_SPEED[0], "#5a8f3c", "carry 3.27"),
                      (REF.RETREAT_SPEED[0], "#96341f", "retreat 4.78")):
        ax1.axhline(y, color=c, ls="--", lw=1.3)
        ax1.text(2, y + 0.12, f"demo {txt}", fontsize=8, color=c)
    ax1.set_ylabel("target speed (mm/step)")
    ax1.set_title(f"{label} against the demonstration profile")
    ax1.grid(alpha=0.25)

    for y, c, txt in ((REF.GRASP_Z, "#444", "grasp 0.419"),
                      (REF.CARRY_Z_MAX, "#5a8f3c", "demo carry 0.499"),
                      (REF.RETREAT_Z_MAX, "#96341f", "demo retreat 0.518")):
        ax2.axhline(y, color=c, ls="--", lw=1.3)
        ax2.text(2, y + 0.002, txt, fontsize=8, color=c)
    ax2.axvspan(REF.TOTAL_STEPS[1], REF.TOTAL_STEPS[2], color="#0b5566", alpha=0.07)
    ax2.text(REF.TOTAL_STEPS[1] + 4, 0.425, "demo episode length band",
             fontsize=8, color="#0b5566")
    ax2.set_ylabel("target height (m)")
    ax2.set_xlabel("policy step")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rl_teacher_v4")
    parser.add_argument("--model", default="latest")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    env = FR3PickPlaceEnv()
    env.rng = np.random.default_rng(args.seed)

    if args.oracle:
        from src.eval.oracle import ScriptedOracle
        oracle = ScriptedOracle()
        oracle.bind(env.model, env.data)
        episodes = collect(None, env, args.episodes, oracle=oracle)
        label = "oracle"
    else:
        from stable_baselines3 import SAC
        path = OUTPUT_ROOT / args.run / f"{args.model}.zip"
        if not path.exists():
            raise SystemExit(f"no model at {path}")
        episodes = collect(SAC.load(str(path), device="cuda"), env, args.episodes)
        label = f"{args.run}/{args.model}"

    score, rows = report(episodes, label)

    tag = args.tag or label.replace("/", "_")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"trajectory_{tag}.png"
    plot(episodes, out, label)
    (DOCS_DIR / f"trajectory_{tag}.png").write_bytes(out.read_bytes())

    (LOG_DIR / f"trajectory_{tag}.json").write_text(json.dumps({
        "label": label, "episodes": len(rows), "score": score,
        "measured": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]
                     if isinstance(rows[0][k], (int, float))},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
