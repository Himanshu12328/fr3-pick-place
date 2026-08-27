"""
Quantitative characterisation of the Cartesian impedance controller.

Runs three headless experiments and writes plots to logs/:

  1. Step response   - commands a step per axis, measures overshoot,
                       rise time and settling time
  2. Frequency sweep - tracks sinusoids across a band, measures gain
                       attenuation and phase lag to find bandwidth
  3. Stiffness sweep - repeats the step at several Kp values so you can
                       see the tradeoff between speed and stability

No viewer. These run far faster than real time, so a full sweep takes
seconds rather than minutes.

Run:
    python src\\scripts\\tune_step.py
"""

import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from src.config import LOG_DIR, MODEL_PATH
from src.controllers.impedance import ImpedanceController

MODEL_PATH = MODEL_PATH
LOG_DIR = LOG_DIR

AXIS_NAMES = ["x", "y", "z"]


def setup(kp_trans=300.0, kp_rot=30.0, zeta=1.0):
    """
    Builds a fresh model, data and controller at the home keyframe.

    Every experiment calls this so each run starts from identical state.
    Reusing a dirty MjData between tests is the classic way to get results
    that depend on test ordering.

    input:  kp_trans (float) N/m, kp_rot (float) Nm/rad, zeta (float)
    output: (model, data, ctrl)
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    model.opt.timestep = 0.001
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    model.dof_frictionloss[:] = 0.0

    data = mujoco.MjData(model)

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)

    ctrl = ImpedanceController(model, data,
                               kp_trans=kp_trans, kp_rot=kp_rot, zeta=zeta,
                               kp_null=10.0, zeta_null=1.0)
    return model, data, ctrl


def settle(model, data, ctrl, seconds=1.0):
    """
    Runs the controller with a fixed target until transients decay.

    Building the controller captures the current pose as the target, but
    the arm still has residual velocity from the keyframe reset. Letting it
    settle first means the step response measures the step, not leftover
    motion.

    input:  model, data, ctrl, seconds (float)
    output: None, mutates data
    """
    for _ in range(int(seconds / model.opt.timestep)):
        data.qfrc_applied[:7] = ctrl.compute_torque(data)
        mujoco.mj_step(model, data)


def run_step(model, data, ctrl, axis, magnitude=0.10, duration=2.0):
    """
    Commands an instantaneous target displacement along one world axis and
    records the response.

    A step is the harshest input you can give a second-order system, which
    is why it exposes damping problems that smooth trajectories hide. The
    target jumps at t=0 and holds; everything after is the controller's
    own dynamics.

    input:  model, data, ctrl, axis (int) 0/1/2 for x/y/z,
            magnitude (float) metres, duration (float) seconds
    output: (t, response, target) arrays. response and target are the
            position along that axis relative to the start pose, in metres
    """
    settle(model, data, ctrl, 1.0)

    start_pos, _ = ctrl.current_pose(data)
    target_pos = start_pos.copy()
    target_pos[axis] += magnitude
    ctrl.set_target(pos=target_pos)

    n_steps = int(duration / model.opt.timestep)
    t = np.arange(n_steps) * model.opt.timestep
    response = np.zeros(n_steps)

    for i in range(n_steps):
        data.qfrc_applied[:7] = ctrl.compute_torque(data)
        mujoco.mj_step(model, data)
        pos, _ = ctrl.current_pose(data)
        response[i] = pos[axis] - start_pos[axis]

    return t, response, np.full(n_steps, magnitude)


def step_metrics(t, response, magnitude, settle_band=0.02):
    """
    Extracts standard second-order step response metrics, returning nan for
    any metric the response never actually reached.

    Overshoot above roughly 5 percent means the damping ratio is too low for
    the stiffness. A response that saturates below target produces no
    meaningful rise or settle time, and nan is the honest answer there. The
    older version silently reported wrong numbers in that case, because
    np.argmax on an all-False array returns index 0 rather than signalling
    that nothing matched.

    input:  t (array) time samples in seconds,
            response (array) displacement in metres, same length as t,
            magnitude (float) commanded step in metres,
            settle_band (float) fraction of magnitude defining "settled"
    output: dict with overshoot_pct, rise_time_s, settle_time_s,
            steady_state_err_mm. rise_time_s is nan if the response never
            reached 90 percent; settle_time_s is nan if it was still
            outside the band when the window ended.
    """
    peak = np.max(response)
    overshoot = max(0.0, (peak - magnitude) / magnitude * 100.0)

    # Rise time: first crossing of 10 percent to first crossing of 90
    # percent. Both crossings must exist, otherwise the indices are
    # meaningless.
    above10 = response >= 0.10 * magnitude
    above90 = response >= 0.90 * magnitude
    if above10.any() and above90.any():
        rise = t[np.argmax(above90)] - t[np.argmax(above10)]
    else:
        rise = float("nan")

    # Settle time: last instant the response was outside the tolerance
    # band. If it is still outside at the end of the window it never
    # settled, which is a different result from settling late.
    band = settle_band * magnitude
    outside = np.abs(response - magnitude) > band

    if not outside.any():
        settle_time = 0.0
    elif outside[-1]:
        settle_time = float("nan")
    else:
        settle_time = t[np.max(np.nonzero(outside))]

    # Average the tail rather than taking the last sample, to reject any
    # residual ripple. 200 samples is 200 ms at a 1 ms timestep.
    ss_err = (magnitude - np.mean(response[-200:])) * 1000.0

    return {
        "overshoot_pct": overshoot,
        "rise_time_s": rise,
        "settle_time_s": settle_time,
        "steady_state_err_mm": ss_err,
    }

def experiment_step_all_axes(kp=300.0, zeta=1.0):
    """
    Runs the step test independently on x, y and z, plots all three, and
    prints a metrics table.

    Testing axes separately matters because the arm's apparent inertia is
    not isotropic. Vertical motion fights gravity through a different set
    of joints than lateral motion, so z often looks different from x and y
    even with identical gains.

    input:  kp (float) N/m, zeta (float)
    output: dict mapping axis name to metrics dict
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    results = {}

    for axis in range(3):
        model, data, ctrl = setup(kp_trans=kp, zeta=zeta)
        t, response, target = run_step(model, data, ctrl, axis)

        m = step_metrics(t, response, 0.10)
        results[AXIS_NAMES[axis]] = m

        ax.plot(t, response * 1000, label=f"{AXIS_NAMES[axis]} axis")

    ax.axhline(100, color="k", ls="--", lw=1, label="target (100 mm)")
    ax.axhspan(98, 102, color="gray", alpha=0.15, label="2% band")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("displacement (mm)")
    ax.set_title(f"Step response, Kp={kp:.0f} N/m, zeta={zeta:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(LOG_DIR, f"step_kp{int(kp)}_z{zeta:.2f}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nStep response  Kp={kp:.0f} N/m  zeta={zeta:.2f}")
    print(f"{'axis':<6}{'overshoot%':>12}{'rise(ms)':>11}{'settle(s)':>11}{'ss err(mm)':>12}")
    for name, m in results.items():
        print(f"{name:<6}{m['overshoot_pct']:>12.2f}{m['rise_time_s']*1000:>11.1f}"
              f"{m['settle_time_s']:>11.3f}{m['steady_state_err_mm']:>12.3f}")
    print(f"saved: {path}")

    return results


def run_sine(model, data, ctrl, axis, freq, amplitude=0.03, cycles=6):
    """
    Tracks a sinusoidal target along one axis and returns the achieved
    amplitude ratio and phase lag.

    The first two cycles are discarded so the measurement reflects
    steady-state tracking rather than the initial transient. Phase is
    recovered by correlating the response against sine and cosine
    references, which is a single-frequency Fourier projection and is far
    more robust to noise than looking for zero crossings.

    input:  model, data, ctrl, axis (int), freq (float) Hz,
            amplitude (float) metres, cycles (int)
    output: (gain, phase_deg) where gain is achieved/commanded amplitude
    """
    settle(model, data, ctrl, 1.0)
    start_pos, _ = ctrl.current_pose(data)

    dt = model.opt.timestep
    n_steps = int(cycles / freq / dt)
    skip = int(2 / freq / dt)  # discard first two cycles

    t = np.arange(n_steps) * dt
    response = np.zeros(n_steps)

    target_pos = start_pos.copy()
    for i in range(n_steps):
        target_pos[axis] = start_pos[axis] + amplitude * np.sin(2 * np.pi * freq * t[i])
        ctrl.set_target(pos=target_pos)

        data.qfrc_applied[:7] = ctrl.compute_torque(data)
        mujoco.mj_step(model, data)

        pos, _ = ctrl.current_pose(data)
        response[i] = pos[axis] - start_pos[axis]

    tt, rr = t[skip:], response[skip:]
    ref_sin = np.sin(2 * np.pi * freq * tt)
    ref_cos = np.cos(2 * np.pi * freq * tt)

    a = 2.0 * np.mean(rr * ref_sin)   # in-phase component
    b = 2.0 * np.mean(rr * ref_cos)   # quadrature component

    gain = np.hypot(a, b) / amplitude
    phase_deg = np.degrees(np.arctan2(b, a))
    return gain, phase_deg


def experiment_frequency_sweep(kp=300.0, zeta=1.0, axis=0):
    """
    Sweeps tracking frequency and plots gain and phase, marking the
    bandwidth.

    Bandwidth here is the frequency where gain drops to -3 dB, about 0.707.
    It sets the ceiling on how fast a teleop operator or a policy can drive
    the arm before commands start arriving attenuated and late, which is
    the number that matters when you get to data collection.

    input:  kp (float), zeta (float), axis (int)
    output: (freqs, gains, phases) arrays
    """
    freqs = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
    gains = np.zeros(len(freqs))
    phases = np.zeros(len(freqs))

    for i, f in enumerate(freqs):
        model, data, ctrl = setup(kp_trans=kp, zeta=zeta)
        gains[i], phases[i] = run_sine(model, data, ctrl, axis, f)
        print(f"  {f:5.2f} Hz   gain {gains[i]:.3f}   phase {phases[i]:7.1f} deg")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.semilogx(freqs, 20 * np.log10(np.maximum(gains, 1e-6)), "o-")
    ax1.axhline(-3, color="r", ls="--", lw=1, label="-3 dB")
    ax1.set_ylabel("gain (dB)")
    ax1.set_title(f"Frequency response, {AXIS_NAMES[axis]} axis, "
                  f"Kp={kp:.0f} N/m, zeta={zeta:.2f}")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    ax2.semilogx(freqs, phases, "o-")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("phase (deg)")
    ax2.grid(alpha=0.3, which="both")

    path = os.path.join(LOG_DIR, f"bode_kp{int(kp)}_z{zeta:.2f}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    below = np.nonzero(gains < 0.707)[0]
    if len(below):
        print(f"\n-3 dB bandwidth: approximately {freqs[below[0]]:.2f} Hz")
    else:
        print(f"\nGain still above -3 dB at {freqs[-1]:.1f} Hz; extend the sweep.")
    print(f"saved: {path}")

    return freqs, gains, phases

def experiment_friction_comparison(kp=300.0, zeta=1.0, axis=2):
    """
    Runs the same z-axis step under three friction conditions to separate
    what the controller does from what the model's passive terms do.

    The three cases isolate different things. Zero friction shows the
    controller's true behaviour. Full friction shows what you would see on
    hardware. Half friction approximates a well-lubricated or
    friction-compensated real arm, which is roughly where a good real
    system sits.

    input:  kp (float) N/m, zeta (float), axis (int)
    output: dict mapping condition name to metrics dict
    """
    conditions = [("no friction", 0.0), ("half friction", 0.5), ("full friction", 1.0)]

    fig, ax = plt.subplots(figsize=(9, 5))
    results = {}

    for label, scale in conditions:
        model, data, ctrl = setup(kp_trans=kp, zeta=zeta)
        # setup() already zeroed frictionloss; scale the original values back in
        ref = mujoco.MjModel.from_xml_path(MODEL_PATH)
        model.dof_frictionloss[:] = ref.dof_frictionloss * scale

        t, response, _ = run_step(model, data, ctrl, axis)
        results[label] = step_metrics(t, response, 0.10)
        ax.plot(t, response * 1000, label=label)

    ax.axhline(100, color="k", ls="--", lw=1)
    ax.axhspan(98, 102, color="gray", alpha=0.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("displacement (mm)")
    ax.set_title(f"Effect of joint friction, {AXIS_NAMES[axis]} axis, Kp={kp:.0f} N/m")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(LOG_DIR, f"friction_kp{int(kp)}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFriction comparison, {AXIS_NAMES[axis]} axis, Kp={kp:.0f} N/m")
    print(f"{'condition':<16}{'overshoot%':>12}{'rise(ms)':>11}{'settle(s)':>11}{'ss err(mm)':>12}")
    for label, m in results.items():
        print(f"{label:<16}{m['overshoot_pct']:>12.2f}{m['rise_time_s']*1000:>11.1f}"
              f"{m['settle_time_s']:>11.3f}{m['steady_state_err_mm']:>12.3f}")
    print(f"saved: {path}")

    return results

def experiment_stiffness_sweep(kp_values=(50, 150, 300, 800, 2000), zeta=1.0):
    """
    Repeats the z-axis step at several stiffness values on shared axes.

    This is the plot that actually chooses your gains. Low Kp is slow and
    compliant, high Kp is fast but eventually starts to ring or saturate
    the torque limits. The knee of that tradeoff is what you want.

    input:  kp_values (tuple of float), zeta (float)
    output: dict mapping Kp to metrics dict
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    results = {}

    for kp in kp_values:
        model, data, ctrl = setup(kp_trans=float(kp), zeta=zeta)
        t, response, _ = run_step(model, data, ctrl, axis=2)
        results[kp] = step_metrics(t, response, 0.10)
        ax.plot(t, response * 1000, label=f"Kp={kp}")

    ax.axhline(100, color="k", ls="--", lw=1)
    ax.axhspan(98, 102, color="gray", alpha=0.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("displacement (mm)")
    ax.set_title(f"z-axis step vs stiffness, zeta={zeta:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(LOG_DIR, f"stiffness_sweep_z{zeta:.2f}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nStiffness sweep, z axis, zeta={zeta:.2f}")
    print(f"{'Kp':>6}{'overshoot%':>12}{'rise(ms)':>11}{'settle(s)':>11}{'ss err(mm)':>12}")
    for kp, m in results.items():
        print(f"{kp:>6}{m['overshoot_pct']:>12.2f}{m['rise_time_s']*1000:>11.1f}"
              f"{m['settle_time_s']:>11.3f}{m['steady_state_err_mm']:>12.3f}")
    print(f"saved: {path}")

    return results

def experiment_zeta_sweep(kp=800.0, zeta_values=(0.5, 0.7, 1.0, 1.4), axis=2):
    """
    Sweeps the damping ratio at fixed stiffness, plotting step response and
    reporting the bandwidth each setting achieves.

    Stiffness sets how hard the arm resists displacement; damping ratio sets
    how much bandwidth you get for that stiffness and how much overshoot you
    pay. Underdamping below about 0.6 starts to ring, which is dangerous in
    contact. Overdamping above 1.0 buys nothing and costs response speed.

    input:  kp (float) N/m, zeta_values (tuple of float), axis (int)
    output: dict mapping zeta to a dict of metrics plus bandwidth_hz
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    results = {}

    for zeta in zeta_values:
        model, data, ctrl = setup(kp_trans=kp, zeta=zeta)
        t, response, _ = run_step(model, data, ctrl, axis)
        m = step_metrics(t, response, 0.10)

        # Find the -3 dB point by sweeping a few frequencies
        bw = float("nan")
        for f in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
            model_f, data_f, ctrl_f = setup(kp_trans=kp, zeta=zeta)
            gain, _ = run_sine(model_f, data_f, ctrl_f, axis, f)
            if gain < 0.707:
                bw = f
                break
        m["bandwidth_hz"] = bw

        results[zeta] = m
        ax.plot(t, response * 1000, label=f"zeta={zeta}")

    ax.axhline(100, color="k", ls="--", lw=1)
    ax.axhspan(98, 102, color="gray", alpha=0.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("displacement (mm)")
    ax.set_title(f"Damping ratio sweep, {AXIS_NAMES[axis]} axis, Kp={kp:.0f} N/m")
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(LOG_DIR, f"zeta_sweep_kp{int(kp)}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDamping sweep, {AXIS_NAMES[axis]} axis, Kp={kp:.0f} N/m")
    print(f"{'zeta':>6}{'overshoot%':>12}{'rise(ms)':>11}{'settle(s)':>11}"
          f"{'ss err(mm)':>12}{'BW(Hz)':>9}")
    for z, m in results.items():
        print(f"{z:>6.2f}{m['overshoot_pct']:>12.2f}{m['rise_time_s']*1000:>11.1f}"
              f"{m['settle_time_s']:>11.3f}{m['steady_state_err_mm']:>12.3f}"
              f"{m['bandwidth_hz']:>9.2f}")
    print(f"saved: {path}")

    return results


def main():
    """
    Runs all three experiments in sequence.

    input:  none
    output: None
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    experiment_step_all_axes(kp=300.0, zeta=1.0)

    print("\nFrequency sweep, x axis:")
    experiment_frequency_sweep(kp=300.0, zeta=1.0, axis=0)

    experiment_stiffness_sweep()
    experiment_friction_comparison()
    experiment_zeta_sweep()

    print(f"\nAll plots written to {LOG_DIR}")


if __name__ == "__main__":
    main()