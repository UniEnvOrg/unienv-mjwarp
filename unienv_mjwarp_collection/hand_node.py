"""The hand of a hand_object_scene model as a node."""

from __future__ import annotations

import torch
from unienv_mjwarp import MjWarpWorld
from unienv_mjwarp.nodes.base import MjWarpNode, unbounded_obs_space
from unienv_mjwarp.nodes.space_metadata import joint_state_metadata

from .hand_object_scene import SceneIndex
from .hands import Hand


class HandNode(MjWarpNode):
    """Observation: ``joint_position`` and ``joint_velocity`` (N, J) of the finger joints, in the hand's joint order.
    The joints are driven by position targets (J + 6: fingers, then the wrist's 3 slides and 3 hinges), held until
    changed."""

    def __init__(self, world: MjWarpWorld, name: str, hand: Hand):
        super().__init__(world, name)
        ix = SceneIndex.build(self.model, hand, hand.links)
        g = lambda x: torch.as_tensor(x, device=world.device)  # noqa: E731
        self.J = len(hand.joints)
        self.qpos_adr, self.dof_adr, self.act = g(ix.qpos_adr), g(ix.dof_adr), g(ix.act)
        md = joint_state_metadata(self.model, ix.joint[: self.J], hand.joints)
        self.observation_space = unbounded_obs_space(world, md, joint_position=(self.J,), joint_velocity=(self.J,))

    def get_observation(self) -> dict[str, torch.Tensor]:
        return {"joint_position": self.world.tensor("qpos")[:, self.qpos_adr[: self.J]],
                "joint_velocity": self.world.tensor("qvel")[:, self.dof_adr[: self.J]]}  # fmt: skip

    def set_state(self, q: torch.Tensor, worlds: torch.Tensor | None = None) -> None:
        """The start state of ``worlds`` (indices; None: all): joints q (n, J + 6), zero velocities, targets at q.
        Positions and contacts are recomputed by the world's after_reset()."""
        w = torch.arange(self.world.batch_size, device=self.world.device) if worlds is None else worlds
        self.world.tensor("qpos")[w[:, None], self.qpos_adr] = q
        self.world.tensor("qvel")[w[:, None], self.dof_adr] = 0.0
        self.world.tensor("ctrl")[w[:, None], self.act] = q

    def set_targets(self, q: torch.Tensor) -> None:
        """Position targets (N, J + 6) of every world."""
        self.world.tensor("ctrl")[:, self.act] = q
