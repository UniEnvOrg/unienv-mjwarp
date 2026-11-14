"""Joints, actuators and bodies of the model."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
import torch
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace

from ..world import MjWarpWorld
from .base import MjWarpNode, unbounded_obs_space


class JointStateNode(MjWarpNode):
    """Observation: ``joint_position`` and ``joint_velocity`` (N, n) of the named hinge and slide joints."""

    def __init__(self, world: MjWarpWorld, name: str, joints: Sequence[str]):
        super().__init__(world, name)
        ids = [self.model.joint(j).id for j in joints]
        one_dof = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}  # jnt_type holds plain ints
        if any(int(self.model.jnt_type[i]) not in one_dof for i in ids):
            raise ValueError("JointStateNode reads hinge and slide joints only")
        self.qpos_adr = torch.as_tensor(self.model.jnt_qposadr[ids], device=world.device)
        self.dof_adr = torch.as_tensor(self.model.jnt_dofadr[ids], device=world.device)
        self.observation_space = unbounded_obs_space(world, joint_position=(len(ids),), joint_velocity=(len(ids),))

    def get_observation(self) -> dict[str, torch.Tensor]:
        return {"joint_position": self.world.tensor("qpos")[:, self.qpos_adr],  # index tensors copy
                "joint_velocity": self.world.tensor("qvel")[:, self.dof_adr]}


class ActuatorControlNode(MjWarpNode):
    """Action: the controls (N, n) of the named actuators (a position actuator's control is its target), within their
    ``ctrlrange`` where limited. Without an action the last one is kept; a reset world's control becomes what the
    reset left in ``ctrl`` (zero after MjWarpWorld.reset, or what a task node wrote)."""

    reset_priorities = {0}  # after the robots and tasks (50-80) have written their reset state
    pre_environment_step_priorities = {50}

    def __init__(self, world: MjWarpWorld, name: str, actuators: Sequence[str], control_timestep: float | None = None):
        super().__init__(world, name)
        ids = [self.model.actuator(a).id for a in actuators]
        limited = self.model.actuator_ctrllimited[ids].astype(bool)
        lo = np.where(limited, self.model.actuator_ctrlrange[ids, 0], -np.inf)
        hi = np.where(limited, self.model.actuator_ctrlrange[ids, 1], np.inf)
        bound = lambda x: torch.as_tensor(x[None], dtype=torch.float32, device=world.device)  # noqa: E731 (1, n)
        self.control_timestep = control_timestep
        self.ids = torch.as_tensor(ids, device=world.device)
        self.action_space = BoxSpace(PyTorchComputeBackend, bound(lo), bound(hi), torch.float32, device=world.device,
                                     shape=(world.batch_size, len(ids)))
        self.control = world.tensor("ctrl")[:, self.ids]

    def set_next_action(self, action: torch.Tensor | None) -> None:
        if action is not None:
            self.control = action.to(self.control, copy=True)  # reset() writes into it, never into the caller's

    def reset(self, *, priority: int = 0, seed: int | None = None, mask: torch.Tensor | None = None, **kwargs) -> None:
        worlds = slice(None) if mask is None else mask
        self.control[worlds] = self.world.tensor("ctrl")[worlds][:, self.ids]

    def pre_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self.world.tensor("ctrl")[:, self.ids] = self.control


class BodyPoseNode(MjWarpNode):
    """Observation: ``body_pose`` (N, 7: position, wxyz quaternion) of the named body, world frame. As MuJoCo computes
    it: after a step it is the pose before that step's integration."""

    def __init__(self, world: MjWarpWorld, name: str, body: str):
        super().__init__(world, name)
        self.body = self.model.body(body).id
        self.observation_space = unbounded_obs_space(world, body_pose=(7,))

    def get_observation(self) -> dict[str, torch.Tensor]:
        w = self.world
        return {"body_pose": torch.cat([w.tensor("xpos")[:, self.body], w.tensor("xquat")[:, self.body]], -1)}
