"""
Renders every camera in the scene to a PNG grid so you can confirm the
wrist cameras actually frame the gripper and the block before collecting
any data.

Bad camera placement is expensive to discover late. A dataset recorded
through a view where the block leaves frame during the approach is not
salvageable without re-collecting.

Run:
    python src\\scripts\\preview_cameras.py
"""

import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np

MODEL_PATH = r"C:\pick_place\models\fr3_pick_place.xml"
OUT_PATH = r"C:\pick_place\logs\camera_preview.png"

WIDTH, HEIGHT = 640, 480


def camera_names(model):
    """
    Returns every camera name defined in the model, in index order.

    input:  model (MjModel)
    output: list of str
    """
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)]


def render_all(model, data, width=WIDTH, height=HEIGHT):
    """
    Renders one RGB frame from each camera at the current sim state.

    A single Renderer is reused across cameras rather than constructed per
    camera. Renderer allocates an offscreen GL framebuffer, which is
    expensive, and in the collection loop this will run at 30 Hz.

    input:  model (MjModel), data (MjData), width (int), height (int)
    output: dict mapping camera name to uint8 array (height, width, 3)
    """
    frames = {}
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for name in camera_names(model):
            renderer.update_scene(data, camera=name)
            frames[name] = renderer.render()
    return frames


def save_grid(frames, path):
    """
    Writes all rendered frames side by side as a single labelled figure.

    input:  frames (dict of name to array), path (str)
    output: None
    """
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, frames.items()):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def main():
    """
    Loads the scene at its home keyframe and renders every camera.

    input:  none
    output: None
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)

    print(f"cameras: {camera_names(model)}")
    save_grid(render_all(model, data), OUT_PATH)


if __name__ == "__main__":
    main()