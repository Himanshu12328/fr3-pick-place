"""
Cartesian impedance controller for the Franka FR3 in MuJoCo.

The arm behaves like a 6-DOF spring-damper attached to the end effector,
with a secondary spring in the nullspace that keeps the elbow near a
comfortable posture without disturbing the tip.

Control law:
    tau = J^T (Kp * pose_error - Kd * ee_velocity)     task space
        + N (Kp_n * posture_error - Kd_n * qvel)       nullspace
        + qfrc_bias                                    gravity + Coriolis
        - qfrc_passive                                 joint damping/springs

The controller drives only the arm DOFs. The scene also contains finger
joints and a free-floating block, so the model's nv is larger than the
arm's DOF count and every array access is indexed explicitly rather than
sliced from the front.

Measured gain sets on this model, friction disabled:
    transit  Kp=800, zeta=0.7   1.5 Hz bandwidth, 2.9% overshoot
    contact  Kp=300, zeta=1.0   0.4 Hz bandwidth, no overshoot
"""

import mujoco
import numpy as np

# FR3 datasheet torque limits, Nm. Clamping to these means an unstable gain
# set shows up as saturation rather than the sim exploding.
FR3_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# Candidate end-effector frame names, tried in order. tcp sits between the
# fingertips and is the correct control point once a gripper is attached.
EE_SITE_CANDIDATES = ["tcp", "attachment_site", "ee_site", "grip_site"]
EE_BODY_CANDIDATES = ["fr3_link7", "link7", "fr3_hand"]


def resolve_ee_frame(model, verbose=True):
    """
    Finds a usable end-effector frame in the model, preferring a site over
    a body.

    Sites are massless markers placed deliberately at a tool tip, so they
    are the right thing to control. If the scene defines none, the last
    link body is a workable stand-in, though its origin sits at the flange
    rather than at a gripper tip.

    input:  model (MjModel), verbose (bool) print the resolved frame
    output: (kind, id) where kind is the string "site" or "body"
    """
    for name in EE_SITE_CANDIDATES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            if verbose:
                print(f"End-effector frame: site '{name}' (id {sid})")
            return "site", sid

    for name in EE_BODY_CANDIDATES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            if verbose:
                print(f"End-effector frame: body '{name}' (id {bid}) — no site found")
            return "body", bid

    raise RuntimeError("No end-effector frame found. Run list_frames() to inspect.")


def list_frames(model):
    """
    Prints every site and body name in the model.

    Diagnostic helper for when resolve_ee_frame fails and you need to see
    what the scene actually defines.

    input:  model (MjModel)
    output: None, prints to stdout
    """
    print("Sites:")
    for i in range(model.nsite):
        print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")
    print("Bodies:")
    for i in range(model.nbody):
        print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")


def mass_matrix(model, data, buf):
    """
    Returns the dense joint-space inertia matrix M(q) as an nv x nv array,
    across MuJoCo versions that differ in field name, storage layout, and
    mj_fullM's own signature.

    Three things have moved between versions. The inertia field was renamed
    from data.qM to data.M. Depending on version it holds either the dense
    matrix or the sparse packed lower triangle. And mj_fullM changed from
    (model, dst, src) to (model, data, dst), reading the sparse matrix off
    data directly. This tries the current signature first and falls back.

    input:  model (MjModel), data (MjData),
            buf (array (nv, nv)) writeable C-contiguous scratch buffer
    output: numpy array of shape (nv, nv), inertia in kg m^2
    """
    src = np.asarray(data.M if hasattr(data, "M") else data.qM)

    if src.size == model.nv * model.nv:
        return src.reshape(model.nv, model.nv)

    try:
        mujoco.mj_fullM(model, data, buf)  # current: reads sparse from data
    except TypeError:
        mujoco.mj_fullM(model, buf, src)  # legacy: sparse passed explicitly

    return buf


def quat_from_mat(mat9):
    """
    Converts a flat 9-element row-major rotation matrix to a unit
    quaternion in MuJoCo's (w, x, y, z) order.

    MuJoCo stores frame orientations as flattened rotation matrices in
    data.site_xmat and data.xmat, but quaternions are far easier to take
    differences of, so everything orientation-related converts here first.

    input:  mat9 (array, shape (9,)) row-major rotation matrix
    output: numpy array of shape (4,), unit quaternion
    """
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(mat9, dtype=np.float64))
    return quat


def orientation_error(quat_des, quat_cur):
    """
    Computes the rotation vector that takes the current orientation to the
    desired one, expressed in world frame.

    Works out the relative rotation q_err = q_des * q_cur^-1, flips its
    sign if the scalar part is negative so the arm always takes the short
    way round rather than rotating 350 degrees to reach 10, then converts
    to an axis-angle vector whose magnitude is the angle in radians. That
    vector is directly usable as a spring error term.

    input:  quat_des (array (4,)), quat_cur (array (4,))
    output: numpy array of shape (3,), rotation vector in radians
    """
    quat_cur_inv = np.zeros(4)
    quat_err = np.zeros(4)
    err_vec = np.zeros(3)

    mujoco.mju_negQuat(quat_cur_inv, quat_cur)
    mujoco.mju_mulQuat(quat_err, quat_des, quat_cur_inv)

    if quat_err[0] < 0.0:
        quat_err = -quat_err

    mujoco.mju_quat2Vel(err_vec, quat_err, 1.0)
    return err_vec


class ImpedanceController:
    """
    Cartesian impedance controller with dynamically-consistent nullspace
    posture regulation, driving a fixed set of arm DOFs within a larger
    model.
    """

    def __init__(
        self,
        model,
        data,
        kp_trans=300.0,
        kp_rot=30.0,
        zeta=1.0,
        kp_null=10.0,
        zeta_null=1.0,
        n_arm=7,
        verbose=True,
    ):
        """
        Sets up the controller and captures the current joint configuration
        as the rest posture.

        Damping is not passed in directly. It is recomputed every step from
        the task-space inertia so that the damping ratio stays at zeta no
        matter how the arm is configured, which is the whole point of doing
        this in task space rather than joint space.

        input:  model (MjModel), data (MjData),
                kp_trans (float) translational stiffness N/m,
                kp_rot (float) rotational stiffness Nm/rad,
                zeta (float) task-space damping ratio,
                kp_null (float) nullspace posture stiffness,
                zeta_null (float) nullspace damping ratio,
                n_arm (int) number of leading DOFs belonging to the arm,
                verbose (bool) print the resolved end-effector frame
        output: ImpedanceController instance
        """
        self.model = model
        self.nv = model.nv  # full model DOF count, for buffer sizing
        self.n_arm = n_arm  # arm DOFs, always the leading block
        self.arm_dofs = np.arange(n_arm)

        self.ee_kind, self.ee_id = resolve_ee_frame(model, verbose)

        self.kp_diag = np.array([kp_trans] * 3 + [kp_rot] * 3)
        self.zeta = zeta
        self.kp_null = kp_null
        self.zeta_null = zeta_null

        self.q_rest = data.qpos[: self.n_arm].copy()

        # Preallocated buffers. Allocating inside a 1 kHz loop is wasteful
        # and creates garbage-collection jitter. Jacobian buffers are full
        # model width because mj_jacSite writes all nv columns and errors on
        # a narrow array; the arm columns are selected afterwards.
        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))
        self._M = np.ascontiguousarray(np.zeros((self.nv, self.nv)))

        self.x_des, self.quat_des = self.current_pose(data)

    def current_pose(self, data):
        """
        Reads the current end-effector position and orientation.

        input:  data (MjData)
        output: (pos, quat) where pos is shape (3,) in metres and quat is
                shape (4,) in (w, x, y, z) order
        """
        if self.ee_kind == "site":
            pos = data.site_xpos[self.ee_id].copy()
            quat = quat_from_mat(data.site_xmat[self.ee_id])
        else:
            pos = data.xpos[self.ee_id].copy()
            quat = quat_from_mat(data.xmat[self.ee_id])
        return pos, quat

    def jacobian(self, data):
        """
        Computes the 6 x n_arm end-effector Jacobian, stacking the
        translational rows above the rotational rows.

        The Jacobian maps joint velocities to end-effector twist. Its
        transpose does the reverse for forces, which is what turns a
        desired Cartesian wrench into joint torques. Only the arm columns
        are kept: the finger and block DOFs also move the model but are not
        this controller's to drive.

        input:  data (MjData)
        output: numpy array of shape (6, n_arm)
        """
        if self.ee_kind == "site":
            mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.ee_id)
        else:
            mujoco.mj_jacBody(self.model, data, self._jacp, self._jacr, self.ee_id)
        return np.vstack([self._jacp, self._jacr])[:, self.arm_dofs]

    def task_space_inertia(self, data, J):
        """
        Computes the task-space inertia matrix Lambda and the
        dynamically-consistent pseudoinverse Jbar.

        Lambda = (J M^-1 J^T)^-1 is the apparent mass the end effector
        presents in each Cartesian direction. It varies enormously with
        configuration, which is why fixed damping gains feel right in one
        pose and mushy in another. Jbar = M^-1 J^T Lambda is the
        pseudoinverse that projects nullspace torques without them leaking
        into end-effector motion.

        The mass matrix is sub-selected to the arm block. Inverting the
        full matrix would couple the arm to the free-floating block through
        terms that have no physical meaning for this controller.

        input:  data (MjData), J (array (6, n_arm))
        output: (Lambda, Jbar, M_inv) of shape (6,6), (n_arm,6), (n_arm,n_arm)
        """
        M_full = mass_matrix(self.model, data, self._M)
        M_arm = M_full[np.ix_(self.arm_dofs, self.arm_dofs)]
        M_inv = np.linalg.inv(M_arm)

        # pinv rather than inv: near a singularity J M^-1 J^T loses rank and
        # a plain inverse produces enormous torques.
        Lambda = np.linalg.pinv(J @ M_inv @ J.T, rcond=1e-4)
        Jbar = M_inv @ J.T @ Lambda
        return Lambda, Jbar, M_inv

    def compute_torque(self, data):
        """
        Computes the joint torques implementing the impedance behaviour at
        the current state.

        Order of operations: pose error, task-space wrench with
        inertia-scaled damping, mapping to joint torques, nullspace posture
        term projected so it cannot move the tip, gravity and Coriolis
        compensation, cancellation of the model's own passive joint forces,
        then saturation to the FR3 limits.

        Subtracting qfrc_passive matters. MuJoCo adds joint damping and
        spring forces to the equation of motion independently of anything
        written into qfrc_applied, so leaving them in means the model's
        damping stacks on top of the controller's and the achieved damping
        ratio is no longer the requested zeta.

        input:  data (MjData)
        output: numpy array of shape (n_arm,), torques in Nm
        """
        pos, quat = self.current_pose(data)
        J = self.jacobian(data)
        Lambda, Jbar, M_inv = self.task_space_inertia(data, J)

        # 6-vector pose error: translation then rotation
        err = np.concatenate([self.x_des - pos, orientation_error(self.quat_des, quat)])

        # Damping scaled by apparent inertia keeps the damping ratio at
        # self.zeta regardless of arm configuration.
        lam_diag = np.clip(np.diag(Lambda), 1e-6, None)
        kd_diag = 2.0 * self.zeta * np.sqrt(lam_diag * self.kp_diag)

        ee_vel = J @ data.qvel[: self.n_arm]
        wrench = self.kp_diag * err - kd_diag * ee_vel
        tau_task = J.T @ wrench

        # Nullspace: pulls joints toward rest posture, projected through
        # N = I - J^T Jbar^T so it produces zero end-effector wrench.
        N = np.eye(self.n_arm) - J.T @ Jbar.T
        kd_null = 2.0 * self.zeta_null * np.sqrt(self.kp_null)
        tau_posture = (
            self.kp_null * (self.q_rest - data.qpos[: self.n_arm])
            - kd_null * data.qvel[: self.n_arm]
        )
        tau_null = N @ tau_posture

        tau = (
            tau_task
            + tau_null
            + data.qfrc_bias[: self.n_arm]
            - data.qfrc_passive[: self.n_arm]
        )

        return np.clip(
            tau, -FR3_TORQUE_LIMITS[: self.n_arm], FR3_TORQUE_LIMITS[: self.n_arm]
        )

    def set_target(self, pos=None, quat=None):
        """
        Updates the desired end-effector pose.

        This is the interface the teleop layer and the policy rollout will
        both write to in later phases. Keeping it as an absolute pose,
        rather than a velocity command, is what lets the same controller
        serve data collection and inference unchanged.

        input:  pos (array (3,) or None), quat (array (4,) or None)
        output: None
        """
        if pos is not None:
            self.x_des = np.asarray(pos, dtype=np.float64).copy()
        if quat is not None:
            self.quat_des = np.asarray(quat, dtype=np.float64).copy()

    def set_gains(self, kp_trans=None, kp_rot=None, zeta=None):
        """
        Updates stiffness or damping ratio at runtime.

        Useful for switching between a stiff transit gain set and a soft
        contact gain set. Change gains gradually rather than in one jump:
        an instantaneous stiffness change makes the commanded wrench jump
        discontinuously, which the arm feels as a kick. Ramping over
        roughly 200 ms is enough to avoid it.

        input:  kp_trans (float or None) N/m, kp_rot (float or None) Nm/rad,
                zeta (float or None)
        output: None
        """
        if kp_trans is not None:
            self.kp_diag[:3] = kp_trans
        if kp_rot is not None:
            self.kp_diag[3:] = kp_rot
        if zeta is not None:
            self.zeta = zeta
