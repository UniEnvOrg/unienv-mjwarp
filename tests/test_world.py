"""MjWarpWorld and the generic nodes on a plane, a free box and a one-joint arm with a position actuator (needs a
GPU)."""

import mujoco
import mujoco_warp as mjw
import torch
from unienv_interface.world.env_composer import WorldEnv

from unienv_mjwarp import ActuatorControlNode, BodyPoseNode, ContactForceNode, JointStateNode, MjWarpWorld

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
    </body>
  </worldbody>
  <actuator>
    <position name="hinge" joint="hinge" kp="10" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>"""


def test_reset_changes_only_the_chosen_worlds_and_recomputes_positions():
    world = MjWarpWorld(mujoco.MjModel.from_xml_string(MODEL), 4)
    box = world.mj_model.body("box").id
    for _ in range(50):
        world.step()
    before = world.tensor("qpos").clone()
    mask = torch.tensor([True, False, True, False], device=world.device)
    world.reset(mask=mask)
    world.tensor("qpos")[mask, 2] = 0.3
    world.after_reset()
    qpos, qvel = world.tensor("qpos"), world.tensor("qvel")
    assert torch.equal(qpos[~mask], before[~mask]) and (qvel[~mask, 2] < 0).all()  # still falling
    assert (qpos[mask, 2] == 0.3).all() and (qvel[mask] == 0).all()
    replayed = world.tensor("xpos").clone()
    assert torch.allclose(replayed[mask, box, 2], torch.tensor(0.3, device=world.device))
    mjw.forward(world.model, world.data)  # launched directly, not from the recorded graph
    assert torch.allclose(world.tensor("xpos"), replayed, atol=1e-6)


def test_nodes_in_a_world_env():
    """The actuator moves the hinge to its target and the joint state reads it; the resting box's pose and the
    contact force on it (its weight)."""
    world = MjWarpWorld(mujoco.MjModel.from_xml_string(MODEL), 4)
    env = WorldEnv(world, [JointStateNode(world, "arm_state", ["hinge"]),
                           ActuatorControlNode(world, "arm_control", ["hinge"]),
                           BodyPoseNode(world, "box_pose", "box"), ContactForceNode(world, "box_contact", ["box"])])
    env.reset()
    target = torch.full((4, 1), 0.5, device=world.device)  # one action node: UniEnv passes the action without a dict
    for _ in range(500):  # 1 s
        obs, *_ = env.step(target)
    assert torch.allclose(obs["arm_state"]["joint_position"], target, atol=0.01)
    z = obs["box_pose"]["body_pose"][:, 2]
    assert torch.allclose(z, torch.full_like(z, 0.02), atol=2e-3), z
    weight = float(world.mj_model.body("box").mass[0]) * 9.81
    force = obs["box_contact"]["net_force"][:, 0]
    assert torch.allclose(force, torch.tensor([0.0, 0.0, weight], device=world.device), atol=0.05 * weight), force
