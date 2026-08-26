"""
Prints live DualSense axis and button state so you can confirm the mapping
before writing the teleop layer.

Axis indices vary between USB and Bluetooth, and between SDL versions, so
this reads them empirically rather than trusting a table.

Run:
    python src\\scripts\\test_pad.py
"""

import time

import pygame


def open_pad():
    """
    Initialises pygame's joystick subsystem and opens the first pad found.

    input:  none
    output: pygame.joystick.Joystick instance
    """
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No controller detected. Plug in the DualSense by USB.")

    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f"{pad.get_name()}  axes={pad.get_numaxes()}  buttons={pad.get_numbuttons()}\n")
    return pad


def poll_loop(pad, rate_hz=20):
    """
    Continuously prints axis values and any pressed button indices.

    pygame.event.pump must be called every iteration or the axis values
    never update, which is the single most common reason a controller
    appears dead in a polling loop.

    input:  pad (Joystick), rate_hz (float) print rate
    output: None, runs until Ctrl+C
    """
    print("Move sticks and triggers. Press buttons. Ctrl+C to stop.\n")

    while True:
        pygame.event.pump()

        axes = [pad.get_axis(i) for i in range(pad.get_numaxes())]
        pressed = [i for i in range(pad.get_numbuttons()) if pad.get_button(i)]

        axis_str = "  ".join(f"a{i}:{v:+.2f}" for i, v in enumerate(axes))
        print(f"\r{axis_str}   btn:{pressed}          ", end="", flush=True)

        time.sleep(1.0 / rate_hz)


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    pad = open_pad()
    try:
        poll_loop(pad)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()