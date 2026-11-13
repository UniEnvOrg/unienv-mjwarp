"""MjWarpWorld on a plane and a free box (needs a GPU)."""

import mujoco
import mujoco_warp as mjw
import torch

from unienv_mjwarp import MjWarpWorld

MODEL = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 .01"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.02"/>
    </body>
  </worldbody>
</mujoco>"""


def test_reset_changes_only_the_chosen_worlds_and_recomputes_positions():
    world = MjWarpWorld(mujoco.MjModel.from_xml_string(MODEL), 4)
    box = world.model.nbody - 1
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
