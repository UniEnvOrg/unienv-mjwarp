"""MjWarpWorld and the generic nodes on a plane, a free box and a one-joint arm with a position actuator (needs a
GPU)."""

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from unienv_interface.world.env_composer import WorldEnv

from mjwarp_adaptor import (
    ActuatorControlNode,
    BodyPoseNode,
    ContactForceNode,
    ContactPointsNode,
    FreeBodyNode,
    JointStateNode,
    MjWarpWorld,
)

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


def test_free_body_node_sets_pose_and_force():
    world = MjWarpWorld(mujoco.MjModel.from_xml_string(MODEL), 4)
    box = FreeBodyNode(world, "box", "box")
    pose = torch.tensor([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], device=world.device).repeat(2, 1)
    worlds = torch.tensor([0, 2], device=world.device)
    world.reset()
    world.tensor("qvel")[:] = 1.0
    box.set_pose(pose, worlds)
    world.after_reset()
    obs = box.get_observation()
    assert torch.allclose(obs["body_pose"][worlds], pose) and (obs["body_velocity"][worlds] == 0).all()
    assert torch.allclose(obs["body_pose"][1, :3], torch.tensor([0.0, 0.0, 0.1], device=world.device))  # untouched
    weight = float(world.mj_model.body("box").mass[0]) * 9.81
    force = torch.zeros(4, 3, device=world.device)
    force[worlds, 2] = 2 * weight  # twice the weight: accelerates upward at g
    box.set_force(force)
    for _ in range(10):
        world.step()
    vz = box.get_observation()["body_velocity"][:, 2]
    expected = 9.81 * 10 * world.world_timestep
    assert torch.allclose(vz[worlds], torch.full((2,), expected, device=world.device), rtol=0.05), vz


def test_contact_points_of_a_resting_box():
    world = MjWarpWorld(mujoco.MjModel.from_xml_string(MODEL), 4)
    for _ in range(500):
        world.step()
    world_id, point, frame = ContactPointsNode(world, "box_contacts", "box", max_distance=1e-3).contacts()
    assert set(world_id.tolist()) == {0, 1, 2, 3}
    assert torch.allclose(point[:, 2], torch.full_like(point[:, 2], -0.02), atol=2e-3), point  # the bottom face
    down = torch.tensor([0.0, 0.0, -1.0], device=world.device)
    assert torch.allclose(frame[:, 0], down.expand_as(frame[:, 0]), atol=1e-3), frame[:, 0]  # out of the box
    assert len(ContactPointsNode(world, "none", "box", max_distance=-1.0).contacts()[0]) == 0


def test_control_timesteps_of_any_whole_number_of_physics_steps():
    model = mujoco.MjModel.from_xml_string(MODEL)
    model.opt.timestep = 1 / 240
    world = MjWarpWorld(model, 2)
    assert all(world.is_control_timestep_compatible(k * world.world_timestep) for k in range(1, 25))
    assert all(world.is_control_timestep_compatible(k / 240) for k in range(1, 25))
    assert not world.is_control_timestep_compatible(2.5 * world.world_timestep)
    assert not world.is_control_timestep_compatible(0.4 * world.world_timestep)
    env = WorldEnv(world, ActuatorControlNode(world, "arm", ["hinge"], control_timestep=6 / 240))
    env.reset()
    _, _, _, _, info = env.step(torch.zeros(2, 1, device=world.device))
    assert abs(float(info["control_step_elapsed_time"]) - 6 / 240) < 1e-9


def test_worlds_on_their_own_streams_match_the_default_stream():
    """Two worlds on their own streams, stepped interleaved after torch writes their controls, end where a world on
    the default stream does. Torch runs on a non-blocking stream of its own, so only wait() and join() order its
    writes and reads with the worlds' steps (the legacy default stream would order them implicitly)."""
    model = mujoco.MjModel.from_xml_string(MODEL)
    ref = MjWarpWorld(model, 4)
    worlds = [MjWarpWorld(model, 4, stream=wp.Stream(wp.get_device())) for _ in range(2)]
    ref.after_reset()
    for i in range(200):  # torch on its default stream: ordered with the default-stream graphs implicitly
        ref.tensor("ctrl")[:] = 0.001 * i
        ref.step()
    qpos, qvel = ref.tensor("qpos").clone(), ref.tensor("qvel").clone()
    with torch.cuda.stream(torch.cuda.Stream()):
        for w in worlds:
            w.wait()
            w.after_reset()
        for i in range(200):
            for w in worlds:
                w.tensor("ctrl")[:] = 0.001 * i
            for w in worlds:  # before the steps read the controls
                w.wait()
            for w in worlds:
                w.step()
            for w in worlds:  # before torch writes the next controls
                w.join()
        for w in worlds:
            assert torch.allclose(w.tensor("qpos"), qpos, atol=1e-6)
            assert torch.allclose(w.tensor("qvel"), qvel, atol=1e-5)
