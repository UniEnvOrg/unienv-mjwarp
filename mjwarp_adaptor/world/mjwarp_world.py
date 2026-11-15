"""MjWarpWorld: UniEnv's World over MuJoCo Warp, one MuJoCo model in nworld copies on the GPU.

One World.step() is one physics step. A reset is three calls, in UniEnv's order:

    world.reset(mask=...)     # the chosen worlds back to the model's default state
    ...                       # the caller writes its start state through world.tensor(...)
    world.after_reset()       # positions and contacts recomputed once, for all worlds

After step(), body positions and contacts are those from before the last integration (MuJoCo's convention);
after_reset() refreshes them. A WorldEnv's first reset goes through reload / after_reload, which do the same.
``seed`` is accepted and unused: the world draws no random numbers.

The step and the recomputation are each recorded once as a CUDA graph and replayed. Launched kernel by kernel from
Python instead, MuJoCo Warp's forward takes about 6 times longer and allocates GPU memory on every call.

By default the graphs run on Warp's current stream, so several worlds (e.g. one per scene) run one after another on
the GPU. MjWarpWorlds gives each world its own CUDA stream and steps them together, so the GPU overlaps them. A world on
its own stream still orders itself with torch: step() and after_reset() make the stream wait for the work queued on
torch's current stream so far, and torch's stream wait for the launch (on the GPU; the host does not wait), so torch
reads and writes world.tensor(...) as without a stream.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.world import World


class Contacts(NamedTuple):
    """Contacts (C of them), all worlds' in one list: views of MuJoCo Warp's contact buffers."""

    geom: torch.Tensor  # (C, 2) int: the geom pair; the normal points from geom 0 to geom 1
    world: torch.Tensor  # (C,) int: the world
    frame: torch.Tensor  # (C, 3, 3): rows normal, tangent, tangent; world frame
    pos: torch.Tensor  # (C, 3): world frame
    force: torch.Tensor  # (C, 6): force then torque on geom 1, in the contact frame or the world frame


class MjWarpWorld(World):
    backend = PyTorchComputeBackend

    def __init__(self, model: mujoco.MjModel, nworld: int, contacts_per_world: int | None = None,
                 rows_per_world: int | None = None, stream: wp.Stream | None = None):  # fmt: skip
        """
        Args:
            model: the scene.
            nworld: the number of copies (the world's batch size).
            contacts_per_world, rows_per_world: MuJoCo Warp's contact and constraint buffer sizes (None: its defaults).
            stream: the CUDA stream the step and the recomputation run on (None: Warp's current stream; MjWarpWorlds
                gives each of its worlds one).
        """
        self.stream = stream
        # freed GPU arrays stay in Warp's pool (up to 10% of the GPU) instead of going back to the driver
        wp.set_mempool_release_threshold(wp.get_device(), 0.1)
        self.mj_model = model  # for name lookups
        self.model = mjw.put_model(model)
        # every world at the model's default joint positions; contacts are found by the first after_reset()
        self.data = mjw.make_data(model, nworld=nworld, nconmax=contacts_per_world, njmax=rows_per_world)
        self.device = wp.device_to_torch(wp.get_device())
        self.world_timestep = float(model.opt.timestep)
        self.batch_size = nworld
        self._qpos0 = torch.as_tensor(model.qpos0, device=self.device, dtype=torch.float32)
        self._tensors: dict[str, torch.Tensor] = {}
        self._graphs: dict[Callable, wp.Graph] = {}
        self._contact_ids = self._contact_force = None  # contact_force()'s buffers, made on its first call

    def is_control_timestep_compatible(self, control_timestep: float | None) -> bool:
        """A whole number of physics steps, within rounding (the base class's exact remainder test rejects e.g.
        6 / 240 s on a 1 / 240 s timestep)."""
        if control_timestep is None:
            return True
        steps = control_timestep / self.world_timestep
        return round(steps) >= 1 and abs(steps - round(steps)) < 1e-9

    def tensor(self, name: str) -> torch.Tensor:
        """A simulation array by MuJoCo Warp's name (``qpos``, ``qvel``, ``ctrl``, ``xpos``, ``xfrc_applied``, ...;
        (nworld, ...)) or a contact buffer's (``contact.geom``, ``contact.pos``, ...; (naconmax, ...), all worlds'
        contacts, the first contact_count() valid) as a torch tensor: a view, made once, so writing to it writes to
        the simulation."""
        if name not in self._tensors:
            array = self.data
            for part in name.split("."):
                array = getattr(array, part)
            self._tensors[name] = wp.to_torch(array)
        return self._tensors[name]

    def contact_count(self) -> int:
        """The number of contacts in the contact buffers (all worlds), read back to the host."""
        n = int(self.tensor("nacon")[0])
        if n > self.data.naconmax:
            raise RuntimeError(f"{n} contacts > the buffer's {self.data.naconmax}: raise contacts_per_world")
        return n

    def contact_force(self, world_frame: bool = False) -> torch.Tensor:
        """(naconmax, 6) force then torque of every contact on its second geom (MuJoCo Warp's contact_force), in the
        contact's frame or the world frame; the first contact_count() rows are valid. A view of one buffer, so the next
        call overwrites it. Runs on torch's current stream."""
        if self._contact_force is None:
            n = self.data.naconmax
            self._contact_ids = wp.array(np.arange(n, dtype=np.int32), dtype=int)
            self._contact_force = wp.zeros(n, dtype=wp.spatial_vector)
        with wp.ScopedStream(wp.stream_from_torch(self.device)):
            mjw.contact_force(self.model, self.data, self._contact_ids, world_frame, self._contact_force)
        return wp.to_torch(self._contact_force)

    def contacts(self, world_frame: bool = False) -> Contacts:
        """The current contacts of all worlds (see Contacts), views of the contact buffers sliced to contact_count();
        the force is MuJoCo Warp's contact_force, in the contact's frame or the world frame."""
        n = self.contact_count()
        t = self.tensor
        return Contacts(t("contact.geom")[:n], t("contact.worldid")[:n], t("contact.frame").view(-1, 3, 3)[:n],
                        t("contact.pos")[:n], self.contact_force(world_frame)[:n])  # fmt: skip

    def _replay(self, fn: Callable) -> None:
        if fn not in self._graphs:
            with wp.ScopedStream(self.stream), wp.ScopedCapture() as capture:  # ScopedStream(None): no change
                fn(self.model, self.data)
            self._graphs[fn] = capture.graph
        wp.capture_launch(self._graphs[fn], stream=self.stream)

    def _wait(self) -> None:
        """This world's stream waits, on the GPU, for the work queued on torch's current stream so far."""
        if self.stream is not None:
            self.stream.wait_stream(wp.stream_from_torch(self.device))

    def _join(self) -> None:
        """Torch's current stream waits, on the GPU, for the work launched on this world's stream so far."""
        if self.stream is not None:
            wp.stream_from_torch(self.device).wait_stream(self.stream)

    def _ordered(self, fn: Callable) -> None:
        self._wait()
        self._replay(fn)
        self._join()

    def step(self) -> float:
        """One physics step of every world; returns the timestep."""
        self._ordered(mjw.step)
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
        self._ordered(mjw.forward)

    after_reload = after_reset  # the model is fixed, so a reload is a reset
