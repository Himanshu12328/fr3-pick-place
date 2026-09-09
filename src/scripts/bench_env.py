"""
Measures how fast this environment actually steps, and whether it scales
across processes on Windows.

Two numbers decide the shape of the RL work. The first is the state-only
rate, because Stage 2 renders nothing and needs millions of steps. The
second is the rendered rate, because Stage 3 does render and needs far
fewer. Both are dominated by something that is easy to overlook: the
impedance controller is Python, it runs at 1 kHz, and a single policy step
calls it 33 times, each call doing a Jacobian and a mass matrix solve.
MuJoCo is not necessarily the bottleneck here. The controller may be.

The multi-process figures also test a specific risk. MuJoCo renders through
WGL on Windows with one OpenGL context per process. Several contexts at
once should work and is unverified in this project, so a rendered
multi-process run either proves it or fails loudly here rather than
halfway through a training run.

Run:
    python -m src.scripts.bench_env
    python -m src.scripts.bench_env --steps 500 --workers 1 4 8 16 20
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

from src.eval import rollout as R


def bench_worker(args):
    """
    Steps the environment as the harness does and returns the rate.

    Uses a fixed target rather than a network so the measurement is of the
    environment alone. A learned policy adds its own forward pass on top,
    and mixing the two would hide which is the constraint.

    input:  args (tuple) of (steps int, need_images bool, seed int)
    output: float, policy steps per second
    """
    steps, need_images, seed = args

    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
    )
    rng = np.random.default_rng(seed)
    R.reset_trial(model, data, ctrl, rng)

    renderer = None
    if need_images:
        renderer = mujoco.Renderer(model, height=R.CAM_HEIGHT, width=R.CAM_WIDTH)

    pos, quat = ctrl.current_pose(data)
    target = pos.copy()

    # One untimed step so first-call allocation and any lazy GL setup do not
    # land inside the measurement.
    R.build_observation(data, renderer, R.CAMERAS, need_images)

    start = time.perf_counter()
    for k in range(steps):
        R.build_observation(data, renderer, R.CAMERAS, need_images)
        target[2] = pos[2] + 0.02 * np.sin(k * 0.05)
        ctrl.set_target(target, quat)
        for _ in range(R.STEPS_PER_ACTION):
            data.qfrc_applied[: R.N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(data, 0.04)
            mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - start

    if renderer is not None:
        renderer.close()

    return steps / elapsed


def run(steps, need_images, workers):
    """
    Runs the benchmark across a number of parallel processes.

    input:  steps (int), need_images (bool), workers (int)
    output: (aggregate float, per_worker float)
    """
    jobs = [(steps, need_images, 100 + i) for i in range(workers)]

    if workers == 1:
        rates = [bench_worker(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rates = list(pool.map(bench_worker, jobs))

    return float(np.sum(rates)), float(np.mean(rates))


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16, 20])
    args = parser.parse_args()

    print(f"{args.steps} policy steps per worker, "
          f"{R.STEPS_PER_ACTION} physics steps each\n")
    print(f"{'mode':<14}{'workers':>9}{'total step/s':>15}{'per worker':>13}"
          f"{'scaling':>10}")
    print("-" * 61)

    for label, need_images in (("state only", False), ("3 cameras", True)):
        base = None
        for w in args.workers:
            try:
                total, per = run(args.steps, need_images, w)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"{label:<14}{w:>9}   FAILED: {type(exc).__name__}: {exc}")
                continue
            if base is None:
                base = total
            print(f"{label:<14}{w:>9}{total:>15.0f}{per:>13.0f}"
                  f"{total / base:>9.1f}x")
        print()

    print("A Stage 2 run of 5 million policy steps takes "
          "5e6 / (state-only total) seconds.")


if __name__ == "__main__":
    main()
