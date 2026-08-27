"""
Reads a DualSense controller and integrates stick deflections into an
absolute end-effector target pose.

Axis mapping, confirmed empirically on this pad under SDL 2.28 over USB:
    a0  left stick X      rest 0.0    -1 left,  +1 right
    a1  left stick Y      rest 0.0    -1 up,    +1 down
    a2  right stick X     rest 0.0    -1 left,  +1 right
    a3  right stick Y     rest 0.0    -1 up,    +1 down
    a4  L2 trigger        rest -1.0   -1 released, +1 pressed
    a5  R2 trigger        rest -1.0   -1 released, +1 pressed

The triggers resting at -1.0 rather than 0.0 is the variant that needs
(v + 1) / 2 to normalise. Getting this backwards makes both triggers read
as fully pressed at startup, which drives the arm into the table.

Velocity scales are set from the measured 1.5 Hz closed-loop bandwidth at
Kp=800, zeta=0.7. Commanding motion faster than the arm can track makes
the logged target pose diverge from the achieved pose, and since the
logged target is what a policy learns to predict, that error goes straight
into the dataset.
"""

import numpy as np
import pygame

# ---------------------------------------------------------------- axes ----

AX_LEFT_X, AX_LEFT_Y = 0, 1
AX_RIGHT_X, AX_RIGHT_Y = 2, 3
AX_L2, AX_R2 = 4, 5

# ------------------------------------------------------------- buttons ----

# Verified with test_pad.py on this pad under SDL 2.28. This layout differs
# from the commonly documented SDL DualSense mapping, so do not substitute
# values from a reference table without re-checking on the actual hardware.
BTN_CROSS, BTN_CIRCLE, BTN_SQUARE, BTN_TRIANGLE = 0, 1, 2, 3
BTN_SHARE = 4  # small button left of the touchpad
BTN_PS = 5  # centre PlayStation button; avoid binding, some
# systems intercept it and open a system overlay
BTN_OPTIONS = 6  # small button right of the touchpad
BTN_L3, BTN_R3 = 7, 8  # stick clicks
BTN_L1, BTN_R1 = 9, 10
BTN_DPAD_UP, BTN_DPAD_DOWN = 11, 12
BTN_DPAD_LEFT, BTN_DPAD_RIGHT = 13, 14
BTN_TOUCHPAD = 15

# Episode control bindings. The D-pad sits under the left thumb, is large,
# and is unambiguous by feel, which matters for a control pressed several
# hundred times in a collection session. Share and Options are recessed and
# easy to miss mid-demonstration.
BTN_START_EP = BTN_DPAD_UP  # 11
BTN_SAVE_EP = BTN_DPAD_RIGHT  # 14
BTN_DISCARD_EP = BTN_DPAD_DOWN  # 12

# --------------------------------------------------------------- tuning ---

DEADZONE = 0.08  # sticks rest within +/-0.02; 0.08 covers drift
LINEAR_SPEED = 0.15  # m/s at full stick deflection
ANGULAR_SPEED = 0.8  # rad/s at full deflection
# GRIPPER_SPEED = 0.08     # m/s of finger travel
GRIPPER_OPEN = 0.04  # metres per finger, fully open
GRIPPER_CLOSED = 0.0  # commanded closed; the PD force clamp does the rest

# Operator viewpoint. Standing in front of the arm looking back at it means
# the robot's +x points toward you and +y is mirrored, so stick directions
# must be flipped to match what you see. Set both to +1.0 when operating
# from behind the base, in the robot's own frame.
#
# Only translation is flipped. Rotation stays in the robot frame, because
# pitch and yaw about a tool you are watching directly read correctly
# either way and flipping them tends to feel worse.
OPERATOR_SIGN_X = -1.0
OPERATOR_SIGN_Y = -1.0


def apply_deadzone(value, deadzone=DEADZONE):
    """
    Zeroes small stick deflections and rescales the remainder so the
    response starts at zero rather than jumping at the deadzone edge.

    Without the rescale, pushing a stick just past the deadzone produces an
    immediate step to 8 percent of full speed, which feels notchy and puts
    a discontinuity into the recorded action stream.

    input:  value (float) raw axis in -1..1, deadzone (float)
    output: float in -1..1, zero inside the deadzone
    """
    if abs(value) < deadzone:
        return 0.0
    sign = np.sign(value)
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def normalise_trigger(value):
    """
    Converts a trigger axis resting at -1.0 into a 0..1 press fraction.

    input:  value (float) raw axis in -1..1
    output: float in 0..1, zero when released
    """
    return (value + 1.0) * 0.5


class DualSenseTeleop:
    """
    Turns controller input into an absolute target pose and gripper width.

    Integrating to an absolute pose is deliberate. Recording velocity
    commands would mean the policy has to predict a quantity the impedance
    controller does not consume, and the controller would need a different
    interface at inference than at collection. Absolute target poses keep
    the two identical.
    """

    def __init__(
        self,
        x_init,
        quat_init,
        gripper_init=0.04,
        workspace_min=None,
        workspace_max=None,
    ):
        """
        Opens the first connected pad and seeds the target at the arm's
        current pose.

        Seeding from the current pose rather than a fixed home means the
        arm does not lurch when teleop starts.

        input:  x_init (array (3,)) starting position,
                quat_init (array (4,)) starting orientation (w,x,y,z),
                gripper_init (float) starting finger opening in metres,
                workspace_min (array (3,) or None) lower position clamp,
                workspace_max (array (3,) or None) upper position clamp
        output: DualSenseTeleop instance
        """
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No controller detected. Connect the DualSense by USB.")

        self.pad = pygame.joystick.Joystick(0)
        self.pad.init()
        print(
            f"Teleop pad: {self.pad.get_name()}  "
            f"axes={self.pad.get_numaxes()}  buttons={self.pad.get_numbuttons()}"
        )

        self.x_des = np.asarray(x_init, dtype=np.float64).copy()
        self.quat_des = np.asarray(quat_init, dtype=np.float64).copy()
        self.gripper = float(gripper_init)

        self.ws_min = np.asarray(workspace_min) if workspace_min is not None else None
        self.ws_max = np.asarray(workspace_max) if workspace_max is not None else None

        self._prev_buttons = {}

    def read_twist(self):
        """
        Polls the pad and returns the commanded end-effector twist plus
        gripper velocity.

        Stick Y axes are negated because the pad reports pushing up as
        negative, and up should mean positive motion. Translation is then
        multiplied by the operator sign constants to account for driving
        the arm while standing in front of it.

        Mapping:
            left stick      x, y translation
            L2 / R2         z translation down / up
            right stick     pitch, yaw
            L1 / R1         roll negative / positive
            square / cross  gripper close / open

        input:  none
        output: (twist, grip_vel) where twist is shape (6,) as
                [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        """
        pygame.event.pump()
        ax = [self.pad.get_axis(i) for i in range(self.pad.get_numaxes())]

        vx = apply_deadzone(-ax[AX_LEFT_Y]) * LINEAR_SPEED * OPERATOR_SIGN_X
        vy = apply_deadzone(-ax[AX_LEFT_X]) * LINEAR_SPEED * OPERATOR_SIGN_Y
        vz = (
            normalise_trigger(ax[AX_R2]) - normalise_trigger(ax[AX_L2])
        ) * LINEAR_SPEED

        wy = apply_deadzone(-ax[AX_RIGHT_Y]) * ANGULAR_SPEED  # pitch
        wz = apply_deadzone(-ax[AX_RIGHT_X]) * ANGULAR_SPEED  # yaw
        wx = (
            self.pad.get_button(BTN_R1) - self.pad.get_button(BTN_L1)
        ) * ANGULAR_SPEED  # roll

        # Binary gripper. Two states rather than an integrated width: a
        # rigid cube has no use for intermediate openings, and a continuous
        # target gets averaged across demonstrations into a gradual closure
        # that starts too early and clips the block. Binary also matches
        # what VLA action heads are pretrained on.
        if self.pad.get_button(BTN_SQUARE):
            self.gripper = GRIPPER_CLOSED
        elif self.pad.get_button(BTN_CROSS):
            self.gripper = GRIPPER_OPEN

        return np.array([vx, vy, vz, wx, wy, wz])

    def integrate(self, twist, dt):
        """
        Advances the target pose by one timestep of the commanded twist.

        Gripper state is not integrated; it is set directly in read_twist.

        Orientation is integrated by composing a small-angle quaternion
        onto the current target, then renormalising. Repeated composition
        accumulates floating point drift that slowly denormalises the
        quaternion, and a non-unit quaternion makes the orientation error
        term in the controller misbehave.

        input:  twist (array (6,)), dt (float) seconds
        output: (x_des, quat_des, gripper) current target state
        """
        self.x_des += twist[:3] * dt

        if self.ws_min is not None:
            self.x_des = np.clip(self.x_des, self.ws_min, self.ws_max)

        omega = twist[3:]
        norm = np.linalg.norm(omega)
        if norm * dt > 1e-9:
            axis = omega / norm
            half = norm * dt * 0.5
            dq = np.concatenate([[np.cos(half)], axis * np.sin(half)])

            w0, x0, y0, z0 = dq
            w1, x1, y1, z1 = self.quat_des
            self.quat_des = np.array(
                [
                    w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                    w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                    w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                    w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
                ]
            )
            self.quat_des /= np.linalg.norm(self.quat_des)

        return self.x_des, self.quat_des, self.gripper

    def button_pressed(self, index):
        """
        Returns True only on the frame a button transitions from up to down.

        Raw get_button is level-triggered, so a single press held for a few
        hundred milliseconds at 1 kHz reads as hundreds of presses. Episode
        start, save and discard all need edge detection.

        input:  index (int) button index
        output: bool
        """
        now = bool(self.pad.get_button(index))
        was = self._prev_buttons.get(index, False)
        self._prev_buttons[index] = now
        return now and not was

    def reseed(self, pos, quat, gripper=0.04):
        """
        Resets the target state to a given pose, discarding accumulated
        integration.

        Called after a scene reset. The target persists independently of
        the simulation, so without reseeding the arm would snap from the
        old target back toward the new home pose at full stiffness.

        input:  pos (array (3,)), quat (array (4,)), gripper (float)
        output: None
        """
        self.x_des = np.asarray(pos, dtype=np.float64).copy()
        self.quat_des = np.asarray(quat, dtype=np.float64).copy()
        self.gripper = float(gripper)

    def close(self):
        """
        Releases the pad and shuts down pygame.

        input:  none
        output: None
        """
        pygame.quit()
