"""MjWarpWorld: UniEnv's World over MuJoCo Warp, one MuJoCo model in nworld copies on the GPU.

One World.step() is one physics step. A reset is three calls, in UniEnv's order:

    world.reset(mask=...)     # the chosen worlds back to the model's default state
    ...                       # the caller writes its start state through world.tensor(...)
    world.after_reset()       # positions and contacts recomputed once, for all worlds

After step(), body positions and contacts are those from before the last integration (MuJoCo's convention);
after_reset() refreshes them.

The step and the recomputation are each recorded once as a CUDA graph and replayed. Launched kernel by kernel from
Python instead, MuJoCo Warp's forward takes about 6 times longer and allocates GPU memory on every call.
"""

from __future__ import annotations

from typing import Any, Callable

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.world import World


class MjWarpWorld(World):
    backend = PyTorchComputeBackend

    def __init__(self, model: mujoco.MjModel, nworld: int, contacts_per_world: int | None = None,
                 rows_per_world: int | None = None):  # fmt: skip
        """
        Args:
            model: the scene.
            nworld: the number of copies (the world's batch size).
            contacts_per_world, rows_per_world: MuJoCo Warp's contact and constraint buffer sizes (None: its defaults).
        """
        # freed GPU arrays stay in Warp's pool (up to 10% of the GPU) instead of going back to the driver
        wp.set_mempool_release_threshold(wp.get_device(), 0.1)
        self.model = mjw.put_model(model)
        # every world at the model's default joint positions; contacts are found by the first after_reset()
        self.data = mjw.make_data(model, nworld=nworld, nconmax=contacts_per_world, njmax=rows_per_world)
        self.device = wp.device_to_torch(wp.get_device())
        self.world_timestep = float(model.opt.timestep)
        self.batch_size = nworld
        self._qpos0 = torch.as_tensor(model.qpos0, device=self.device, dtype=torch.float32)
        self._tensors: dict[str, torch.Tensor] = {}
        self._graphs: dict[Callable, wp.Graph] = {}

    def tensor(self, name: str) -> torch.Tensor:
        """A simulation array by MuJoCo's name (``qpos``, ``qvel``, ``ctrl``, ``xpos``, ``xfrc_applied``, ...) as a
        torch tensor, (nworld, ...): a view, so writing to it writes to the simulation."""
        if name not in self._tensors:
            self._tensors[name] = wp.to_torch(getattr(self.data, name))
        return self._tensors[name]

    def _replay(self, fn: Callable) -> None:
        if fn not in self._graphs:
            with wp.ScopedCapture() as capture:
                fn(self.model, self.data)
            self._graphs[fn] = capture.graph
        wp.capture_launch(self._graphs[fn])

    def step(self) -> float:
        """One physics step of every world; returns the timestep."""
        self._replay(mjw.step)
        return self.world_timestep

    def reset(self, *, seed: int | None = None, mask: torch.Tensor | None = None, **kwargs: Any) -> None:
        """The worlds in ``mask`` (bool, (nworld,); None: all) get the model's default joint positions and zero
        velocities, controls and applied forces. Nothing is recomputed: after_reset() does that."""
        worlds = slice(None) if mask is None else mask
        self.tensor("qpos")[worlds] = self._qpos0
        for name in ("qvel", "ctrl", "xfrc_applied"):
            self.tensor(name)[worlds] = 0.0

    def after_reset(self, *, seed: int | None = None, mask: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Recomputes positions and contacts from the current state (MuJoCo's forward), for all worlds."""
        self._replay(mjw.forward)
