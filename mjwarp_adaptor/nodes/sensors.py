"""Contact sensing."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import warp as wp

from ..world import MjWarpWorld
from .base import MjWarpNode, unbounded_obs_space


class ContactForceNode(MjWarpNode):
    """Observation: ``net_force`` (N, bodies, 3), the total contact force on each named body, world frame, from the
    contacts of the last step or after_reset."""

    def __init__(self, world: MjWarpWorld, name: str, bodies: Sequence[str]):
        super().__init__(world, name)
        self.num_bodies = len(bodies)
        slot = torch.full((self.model.nbody,), self.num_bodies)  # bodies not named: an extra slot, dropped
        slot[[self.model.body(b).id for b in bodies]] = torch.arange(self.num_bodies)
        self.geom_slot = slot[torch.as_tensor(self.model.geom_bodyid)].to(world.device)
        self.observation_space = unbounded_obs_space(world, net_force=(self.num_bodies, 3))

    def get_observation(self) -> dict[str, torch.Tensor]:
        w = self.world
        c = w.contacts(world_frame=True)
        f = c.force[:, :3]  # on the contact's second geom (the normal points 0 -> 1)
        geom, world_id = c.geom.long(), c.world.long()
        out = torch.zeros(w.batch_size, self.num_bodies + 1, 3, device=w.device)
        out.index_put_((world_id, self.geom_slot[geom[:, 1]]), f, accumulate=True)
        out.index_put_((world_id, self.geom_slot[geom[:, 0]]), -f, accumulate=True)
        return {"net_force": out[:, : self.num_bodies]}


class ContactPointsNode(MjWarpNode):
    """The contacts between one body and anything else, from the last step or after_reset. Their number changes, so
    they are read with contacts(), not as an observation."""

    def __init__(self, world: MjWarpWorld, name: str, body: str, max_distance: float = 0.0):
        """max_distance: contacts with a larger distance between the surfaces are left out (touching: 0 or less)."""
        super().__init__(world, name)
        self.body, self.max_distance = self.model.body(body).id, max_distance
        self.geom_on_body = torch.as_tensor(self.model.geom_bodyid == self.body, device=world.device)
        self.xmat = world.tensor("xmat").view(world.batch_size, self.model.nbody, 3, 3)

    def contacts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """World (M,) of each contact, and in the body's frame its point (M, 3) and frame (M, 3, 3: rows the normal,
        pointing out of the body, and two tangents)."""
        w, d = self.world, self.world.data
        n = int(d.nacon.numpy()[0])
        if n > d.naconmax:
            raise RuntimeError(f"{n} contacts > the buffer's {d.naconmax}: raise MjWarpWorld's contacts_per_world")
        geom = wp.to_torch(d.contact.geom)[:n].long()
        second = self.geom_on_body[geom[:, 1]]
        keep = (self.geom_on_body[geom[:, 0]] ^ second) & (wp.to_torch(d.contact.dist)[:n] < self.max_distance)
        world_id = wp.to_torch(d.contact.worldid)[:n].long()[keep]
        # the contact normal points from geom 0 to geom 1
        frame = wp.to_torch(d.contact.frame).view(-1, 3, 3)[:n][keep] * torch.where(second[keep], -1.0, 1.0)[:, None, None]
        R, p = self.xmat[world_id, self.body], w.tensor("xpos")[world_id, self.body]
        point = (R.transpose(-1, -2) @ (wp.to_torch(d.contact.pos)[:n][keep] - p)[..., None])[..., 0]
        return world_id, point, frame @ R
