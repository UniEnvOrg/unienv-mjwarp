"""Space metadata of the nodes: valid in UniEnv's vocabulary, with the names and units the model determines (the node
tests need a GPU)."""

import mujoco
import numpy as np
import pytest
import trimesh
from unienv_interface.space.space_utils.metadata_utils import validate_metadata
from unienv_interface.world.env_composer import WorldEnv

from unienv_mjwarp import (
    ActuatorControlNode,
    BodyPoseNode,
    ContactForceNode,
    FreeBodyNode,
    JointStateNode,
    MjWarpWorld,
)
from unienv_mjwarp.nodes.space_metadata import actuator_action_metadata, actuator_control
from unienv_mjwarp_collection import Hand, HandNode, SimConfig, build_model

MODEL = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 .01"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.02"/>
    </body>
    <body name="arm" pos="0.5 0 0.3">
      <joint name="hinge" axis="0 0 1" damping="1"/>
      <geom type="box" size="0.1 0.01 0.01" pos="0.1 0 0" contype="0" conaffinity="0"/>
      <body name="slider">
        <joint name="slide" type="slide" axis="1 0 0" damping="1"/>
        <geom type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="hinge_pos" joint="hinge" kp="10" kv="1" ctrlrange="-1.5 1.5"/>
    <position name="slide_pos" joint="slide" kp="10" timeconst="0.01"/>
    <motor name="hinge_motor" joint="hinge"/>
    <motor name="slide_motor" joint="slide"/>
    <velocity name="slide_vel" joint="slide" kv="5"/>
    <motor name="geared_motor" joint="hinge" gear="2"/>
    <intvelocity name="hinge_intvel" joint="hinge" kp="10" actrange="-1 1"/>
    <general name="hinge_scaled" joint="hinge" gainprm="3"/>
  </actuator>
</mujoco>"""


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(MODEL)


def _ids(m, names):
    return [m.actuator(a).id for a in names]


def test_actuator_controls_are_derived_from_the_model(model):
    expected = {"hinge_pos": "joint_position", "slide_pos": "joint_position", "hinge_motor": "joint_torque",
                "slide_motor": "joint_torque", "slide_vel": "joint_velocity", "geared_motor": None,
                "hinge_intvel": None, "hinge_scaled": None}  # fmt: skip
    assert {a: actuator_control(model, model.actuator(a).id) for a in expected} == expected


@pytest.mark.parametrize(("actuators", "control", "units"), [
    (["hinge_pos", "slide_pos"], "joint_position", ["rad", "m"]),
    (["hinge_motor", "slide_motor"], "joint_torque", ["N*m", "N"]),
    (["slide_vel"], "joint_velocity", ["m/s"]),
    (["hinge_pos", "hinge_motor"], None, None),  # mixed: no single control
    (["hinge_pos", "geared_motor"], None, None),
])  # fmt: skip
def test_actuator_action_metadata(model, actuators, control, units):
    md = actuator_action_metadata(model, _ids(model, actuators), actuators)
    assert md["kind"] == "action" and md["mode"] == "absolute" and md["names"] == actuators
    assert md.get("control") == control and md.get("units") == units


def test_node_spaces_are_described():
    world = MjWarpWorld(mujoco.MjModel.from_xml_string(MODEL), 2)
    joints = JointStateNode(world, "arm_state", ["hinge", "slide"])
    control = ActuatorControlNode(world, "arm_control", ["hinge_pos", "slide_pos"])
    pose, free = BodyPoseNode(world, "arm_pose", "arm"), FreeBodyNode(world, "box", "box")
    contact = ContactForceNode(world, "contact", ["box", "arm"])
    env = WorldEnv(world, [joints, control, pose, free, contact])
    validate_metadata(env.observation_space)
    validate_metadata(env.action_space)

    obs = joints.observation_space
    assert obs["joint_position"].metadata.to_dict()["names"] == ["hinge", "slide"]
    assert obs["joint_position"].metadata.to_dict()["units"] == ["rad", "m"]
    assert obs["joint_velocity"].metadata.to_dict()["units"] == ["rad/s", "m/s"]
    assert control.action_space.metadata == {"kind": "action", "control": "joint_position", "mode": "absolute",
                                                   "units": ["rad", "m"], "names": ["hinge_pos", "slide_pos"]}
    for node in (pose, free):
        md = node.observation_space["body_pose"].metadata.to_dict()
        assert md["kind"] == "pose" and md["frame"] == "world" and md["rotation"] == "quat_wxyz"
        assert md["names"] == ["x", "y", "z", "qw", "qx", "qy", "qz"] and md["units"] == ["m"] * 3 + ["normalized"] * 4
    velocity = free.observation_space["body_velocity"].metadata.to_dict()
    assert velocity["units"] == ["m/s"] * 3 + ["rad/s"] * 3 and "frame" not in velocity
    force = contact.observation_space["net_force"].metadata.to_dict()
    assert force["units"] == "N" and force["frame"] == "world" and force["names"] == ["fx", "fy", "fz"]
    assert force["bodies"] == ["box", "arm"]
    assert env.observation_space["arm_state"]["joint_position"].metadata == obs["joint_position"].metadata  # kept
    assert env.action_space.metadata == control.action_space.metadata  # one action node: its space as is


HAND = """
<mujoco>
  <worldbody>
    <body name="palm">
      <geom type="box" size="0.04 0.04 0.01" mass="0.3"/>
      <body name="finger" pos="0.04 0 0">
        <joint name="knuckle" axis="0 1 0" range="0 1.5"/>
        <geom type="capsule" size="0.008" fromto="0 0 0 0.03 0 0" mass="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="knuckle" joint="knuckle" kp="1" kv="0.01" forcerange="-1 1"/>
  </actuator>
</mujoco>"""


def test_hand_node_space_is_described(tmp_path):
    mjcf = tmp_path / "hand.xml"
    mjcf.write_text(HAND)
    piece = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.04, 0.04, 0.04)).export(piece)
    hand = Hand(name="test", mjcf=mjcf, glove=tmp_path / "glove.npz", palm="palm", fingers={"index": ("finger",)},
                wrist_slide_gains=(500.0, 30.0), wrist_hinge_gains=(10.0, 0.3), dexonomy="test", joints=("knuckle",),
                limits=np.array([[0.0, 1.5]]), kp=np.ones(1), kd=np.full(1, 0.01), tau_max=np.ones(1))  # fmt: skip
    world = MjWarpWorld(build_model(hand, [piece], 1.0, 0.1, SimConfig()), 2)
    space = HandNode(world, "hand", hand).observation_space
    validate_metadata(space)
    assert space["joint_position"].metadata.to_dict()["names"] == ["knuckle"]
    assert space["joint_position"].metadata.to_dict()["units"] == ["rad"]
    assert space["joint_velocity"].metadata.to_dict()["units"] == ["rad/s"]
