"""
build_scene.py

Generates a complete pick-and-place scene from Menagerie's fr3.xml by
adding a parallel-jaw gripper, a table, a graspable block, and cameras.

Writes models/fr3_pick_place.xml. Run this once; after that the generated
file is a plain MJCF you can inspect and hand-edit.

Run:
    python src\\scripts\\build_scene.py
"""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.config import FR3_DIR, SCENE_PATH

MENAGERIE_FR3 = FR3_DIR
OUT_PATH = str(SCENE_PATH)

# Scene geometry, all in metres. Edit these to move things around.
TABLE_CENTER = (0.55, 0.0, 0.20)   # centre of the table box
TABLE_HALF = (0.30, 0.40, 0.20)    # half-extents, so top surface is at z=0.40
BLOCK_HALF = 0.022                 # half-edge of the cube to grasp
BLOCK_START = (0.55, 0.0)          # x, y of the block on the table

FINGER_TRAVEL = 0.04               # metres each finger can slide
FINGER_LENGTH = 0.055              # how far the fingers extend past the palm

# Intel RealSense D405 approximate intrinsics. The D405 is the usual choice
# for wrist mounting in manipulation work because its minimum range is about
# 7 cm, where a D435 cannot focus. fovy is the vertical field of view in
# degrees, which is what MuJoCo's camera element takes.
RS_FOVY = 58          # D405 vertical FOV; use 42 to emulate a D435
RS_WIDTH = 640
RS_HEIGHT = 480


def gripper_xml(site_pos, site_quat):
    """
    Builds the parallel-jaw gripper as an XML element tree, positioned to
    match the arm's attachment site.

    Two prismatic joints slide the fingers symmetrically along the local x
    axis. No actuators are declared, because everything in this project is
    driven by writing torques into qfrc_applied, and adding actuators would
    mean the gain-zeroing in the existing scripts has to distinguish arm
    from gripper.

    The tcp site sits between the fingertips. That is the frame the
    impedance controller should track, rather than the flange, since it is
    where the object actually gets grasped.

    input:  site_pos (str) "x y z" from the attachment site,
            site_quat (str) "w x y z" from the attachment site, or None
    output: xml.etree.ElementTree.Element for the gripper base body
    """
    quat_attr = f'quat="{site_quat}" ' if site_quat else ""

    xml = f'''
    <body name="gripper_base" pos="{site_pos}" {quat_attr}>
      <inertial pos="0 0 0.03" mass="0.7" diaginertia="0.001 0.001 0.001"/>
      <geom name="palm" type="box" size="0.035 0.025 0.03" pos="0 0 0.03"
            rgba="0.25 0.25 0.28 1" contype="1" conaffinity="1"/>

      <site name="tcp" pos="0 0 {0.06 + FINGER_LENGTH:.4f}" size="0.005"
            rgba="1 0 0 1" group="3"/>

      <camera name="wrist_left" pos="0 -0.065 0.045"
              xyaxes="1 0 0  0 -0.781 0.625" fovy="{RS_FOVY}"/>
      <camera name="wrist_right" pos="0 0.065 0.045"
              xyaxes="-1 0 0  0 0.781 0.625" fovy="{RS_FOVY}"/>

      <body name="left_finger" pos="0 0 0.06">
        <joint name="left_finger" type="slide" axis="1 0 0"
               range="0 {FINGER_TRAVEL}" damping="5" armature="0.01"/>
        <inertial pos="0 0 0.025" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
        <geom name="left_pad" type="box"
              size="0.008 0.014 {FINGER_LENGTH / 2:.4f}"
              pos="0 0 {FINGER_LENGTH / 2:.4f}"
              rgba="0.7 0.7 0.72 1" friction="1.5 0.05 0.001"
              solimp="0.95 0.99 0.001" solref="0.005 1"/>
      </body>

      <body name="right_finger" pos="0 0 0.06">
        <joint name="right_finger" type="slide" axis="-1 0 0"
               range="0 {FINGER_TRAVEL}" damping="5" armature="0.01"/>
        <inertial pos="0 0 0.025" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
        <geom name="right_pad" type="box"
              size="0.008 0.014 {FINGER_LENGTH / 2:.4f}"
              pos="0 0 {FINGER_LENGTH / 2:.4f}"
              rgba="0.7 0.7 0.72 1" friction="1.5 0.05 0.001"
              solimp="0.95 0.99 0.001" solref="0.005 1"/>
      </body>
    </body>
    '''
    return ET.fromstring(xml)


def scene_extras_xml():
    """
    Builds the static scene furniture: floor, lights, table, target marker,
    and an external camera.

    The block is deliberately not included here. It is added separately
    because it carries a free joint, and free joints must be counted when
    the keyframe is rewritten.

    input:  none
    output: list of Elements to append to worldbody
    """
    tx, ty, tz = TABLE_CENTER
    hx, hy, hz = TABLE_HALF
    table_top = tz + hz

    xml = f'''
    <worldbody>
      <site name="target_marker" type="sphere" size="0.012"
            pos="0.55 0 0.60" rgba="0.2 0.5 1.0 0.6" group="4"/>
      
      <light name="top" pos="0 0 2.5" dir="0 0 -1" directional="true"
             diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2"/>
      
      <light name="fill" pos="1 -1 2" dir="-0.4 0.4 -1" directional="true"
             diffuse="0.4 0.4 0.4"/>
      
      <light name="wrist_fill" pos="0.55 0 1.2" dir="0 0 -1"
             diffuse="0.5 0.5 0.5"/>

      <geom name="floor" type="plane" size="3 3 0.05" rgba="0.3 0.3 0.35 1"
            contype="1" conaffinity="1"/>

      <geom name="table" type="box" size="{hx} {hy} {hz}"
            pos="{tx} {ty} {tz}" rgba="0.55 0.45 0.35 1"
            friction="1.0 0.05 0.001" contype="1" conaffinity="1"/>

      <site name="place_target" type="cylinder" size="0.04 0.001"
            pos="{tx} {ty + 0.20:.3f} {table_top + 0.002:.4f}"
            rgba="0.2 0.8 0.3 0.5"/>

      <camera name="external" pos="1.2 -0.9 1.0"
              xyaxes="0.6 0.8 0 -0.35 0.26 0.9" fovy="50"/>
    </worldbody>
    '''
    return list(ET.fromstring(xml))


def block_xml():
    """
    Builds the graspable block as a free-floating body.

    A free joint gives the block six degrees of freedom, so it can be
    lifted, tipped and dropped. Its friction and solver parameters are set
    to make grasping reliable, since default contact softness lets a cube
    squirt out from between two flat pads.

    input:  none
    output: Element for the block body
    """
    bx, by = BLOCK_START
    bz = TABLE_CENTER[2] + TABLE_HALF[2] + BLOCK_HALF

    xml = f'''
    <body name="block" pos="{bx} {by} {bz:.4f}">
      <freejoint name="block_free"/>
      <inertial pos="0 0 0" mass="0.05"
                diaginertia="2e-5 2e-5 2e-5"/>
      <geom name="block_geom" type="box"
            size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}"
            rgba="0.85 0.3 0.25 1" friction="1.5 0.05 0.001"
            solimp="0.95 0.99 0.001" solref="0.005 1"
            contype="1" conaffinity="1"/>
    </body>
    '''
    return ET.fromstring(xml)


def find_attachment_site(root):
    """
    Locates the arm's attachment site and the body it belongs to.

    The gripper must be parented to that body and placed at that site's
    pose, otherwise it floats detached or ends up inside the wrist.

    input:  root (Element) parsed fr3.xml root
    output: (parent_body, pos, quat) where pos and quat are strings or None
    """
    for body in root.iter("body"):
        for site in body.findall("site"):
            if site.get("name") == "attachment_site":
                return body, site.get("pos", "0 0 0"), site.get("quat")
    raise RuntimeError("attachment_site not found in fr3.xml")


def rewrite_keyframe(root, n_arm, n_finger, block_pos):
    """
    Rewrites the home keyframe so its qpos length matches the enlarged
    model.

    MuJoCo refuses to load a model whose keyframe qpos does not match nq
    exactly. Adding two finger joints and a free-jointed block changes nq
    from 7 to 16, so the original seven-element keyframe becomes invalid.
    Finger joints start open and the block starts at its rest pose with
    identity orientation.

    input:  root (Element), n_arm (int), n_finger (int),
            block_pos (tuple of 3 float)
    output: None, mutates the tree
    """
    keyframe = root.find("keyframe")
    if keyframe is None:
        print("No keyframe in fr3.xml; skipping.")
        return

    for key in keyframe.findall("key"):
        qpos = key.get("qpos", "").split()
        arm = qpos[:n_arm] if len(qpos) >= n_arm else ["0"] * n_arm

        fingers = [f"{FINGER_TRAVEL:.4f}"] * n_finger   # start open
        block = [f"{v:.4f}" for v in block_pos] + ["1", "0", "0", "0"]

        key.set("qpos", " ".join(arm + fingers + block))

        # qvel, if present, must also match nv
        if key.get("qvel") is not None:
            key.set("qvel", " ".join(["0"] * (n_arm + n_finger + 6)))


def build():
    """
    Assembles the full scene and writes it to disk.

    input:  none
    output: str path to the written XML
    """
    tree = ET.parse(os.path.join(MENAGERIE_FR3, "fr3.xml"))
    root = tree.getroot()

    # Mesh paths in fr3.xml are relative to its own folder. The output
    # lives elsewhere, so make the asset directory absolute.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", os.path.join(MENAGERIE_FR3, "assets"))
    compiler.set("angle", compiler.get("angle", "radian"))

    # Simulation options tuned for a stiff torque-controlled loop.
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.001")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("noslip_iterations", "3")

    # Brighter ambient light. Camera renders are much dimmer than the
    # interactive viewer, which supplies its own headlight, so a scene
    # that looks fine in the viewer can come out nearly black in a dataset.
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.5 0.5 0.5")
    headlight.set("diffuse", "0.4 0.4 0.4")

    # Attach the gripper at the flange.
    parent, site_pos, site_quat = find_attachment_site(root)
    parent.append(gripper_xml(site_pos, site_quat))
    print(f"Gripper attached to body '{parent.get('name')}' at pos {site_pos}")

    # Worldbody furniture, then the block last so its free joint DOFs
    # land at the end of qpos.
    worldbody = root.find("worldbody")
    for elem in scene_extras_xml():
        worldbody.append(elem)
    worldbody.append(block_xml())

    block_pos = (BLOCK_START[0], BLOCK_START[1],
                 TABLE_CENTER[2] + TABLE_HALF[2] + BLOCK_HALF)
    rewrite_keyframe(root, n_arm=7, n_finger=2, block_pos=block_pos)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(OUT_PATH, encoding="utf-8", xml_declaration=False)
    print(f"Wrote {OUT_PATH}")
    return OUT_PATH


def verify(path):
    """
    Loads the generated scene and reports its structure and the resting
    end-effector pose.

    Loading is the real test: MuJoCo validates the whole model at compile
    time, so a mistake in the generated XML surfaces here rather than
    halfway through a training run. The printed tcp position tells you
    whether the table is within comfortable reach.

    input:  path (str)
    output: None, prints to stdout
    """
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)

    print(f"\nnq={model.nq}  nv={model.nv}  nu={model.nu}  "
          f"nbody={model.nbody}  ncam={model.ncam}")

    print("\nJoints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = mujoco.mjtJoint(model.jnt_type[i]).name
        print(f"  [{i}] {name:<20} {jtype:<16} dofadr={model.jnt_dofadr[i]}")

    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    print(f"\ntcp at home:  {np.round(data.site_xpos[tcp_id], 4)}")
    print(f"block at:     {np.round(data.xpos[block_id], 4)}")
    print(f"reach needed: {np.linalg.norm(data.site_xpos[tcp_id] - data.xpos[block_id]):.3f} m")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    verify(build())


if __name__ == "__main__":
    main()