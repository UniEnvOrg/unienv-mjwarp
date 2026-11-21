"""The hands, the hand + object scene and HandNode (the hand models come from tactile_tracker2's setup scripts; the
HandNode test needs a GPU)."""

import mujoco
import numpy as np
import pytest
import torch
import trimesh

from unienv_mjwarp import FreeBodyNode, MjWarpWorld
from unienv_mjwarp_collection import WRIST_JOINTS, HandNode, SimConfig, build_model, load_hand
from unienv_mjwarp_collection.hand_object_scene import HAND_GROUP, OBJECT_GROUP


@pytest.fixture(scope="module", params=["mano", "wuji"])
def scene(request, tmp_path_factory):
    hand = load_hand(request.param)
    piece = tmp_path_factory.mktemp("object") / "box.obj"
    trimesh.creation.box(extents=(0.04, 0.04, 0.04)).export(piece)
    return hand, build_model(hand, [piece], 1.0, 0.1, SimConfig())


def test_scene_has_a_wrist_a_free_object_and_only_hand_object_collisions(scene):
    hand, m = scene
    assert all(m.joint(j).id >= 0 for j in WRIST_JOINTS) and m.nu == len(hand.joints) + 6
    obj = m.body("object").id
    assert m.jnt_type[m.body_jntadr[obj]] == mujoco.mjtJoint.mjJNT_FREE
    assert np.isclose(m.body_subtreemass[obj], 0.1)
    colliding = (m.geom_contype | m.geom_conaffinity) > 0
    on_object = m.geom_bodyid == obj
    assert (m.geom_contype[colliding & on_object] == OBJECT_GROUP).all()
    assert (m.geom_conaffinity[colliding & on_object] == HAND_GROUP).all()
    assert (m.geom_contype[colliding & ~on_object] == HAND_GROUP).all()
    assert (m.geom_conaffinity[colliding & ~on_object] == OBJECT_GROUP).all()


def test_hand_node_holds_the_state_it_sets(scene):
    hand, m = scene
    cfg = SimConfig()
    world = MjWarpWorld(m, 2, cfg.contacts_per_world, cfg.rows_per_world)
    node, obj = HandNode(world, "hand", hand), FreeBodyNode(world, "object", "object")
    q = torch.zeros(2, len(hand.joints) + 6, device=world.device)
    q[:, : len(hand.joints)] = torch.as_tensor(hand.limits.mean(1), device=world.device)  # mid range
    world.reset()
    node.set_state(q)
    obj.set_pose(torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=world.device).repeat(2, 1))  # out of reach
    world.after_reset()
    for _ in range(20):
        world.step()
    obs = node.get_observation()
    assert torch.allclose(obs["joint_position"], q[:, : len(hand.joints)], atol=1e-3), obs["joint_position"]
    assert obs["joint_velocity"].abs().max() < 1e-2
