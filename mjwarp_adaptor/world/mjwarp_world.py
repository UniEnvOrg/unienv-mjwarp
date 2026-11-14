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
the GPU. Give each its own stream (``MjWarpWorld(..., stream=wp.Stream(wp.get_device()))``) and they can overlap.
The caller then orders torch's work and the worlds' with wait() and join() (both GPU-side; the host does not wait):

    for w in worlds:
        w.wait()    # the world's stream waits for torch's writes queued so far
    for w in worlds:
        w.step()    # or after_reset()
    for w in worlds:
        w.join()    # torch waits for the step before it reads or writes the world's arrays again

All the wait() calls go before any launch: torch's default stream is CUDA's legacy stream, and anything queued on it
(such as the event wait() records) waits for every stream's work so far, so a wait() between two launches would make
the second world wait for the first.
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
                 rows_per_world: int | None = None, stream: wp.Stream | None = None):  # fmt: skip
        """
        Args:
            model: the scene.
            nworld: the number of copies (the world's batch size).
            contacts_per_world, rows_per_world: MuJoCo Warp's contact and constraint buffer sizes (None: its defaults).
            stream: the CUDA stream the step and the recomputation run on (None: Warp's current stream). See wait()
                and join().
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

    def is_control_timestep_compatible(self, control_timestep: float | None) -> bool:
        """A whole number of physics steps, within rounding (the base class's exact remainder test rejects e.g.
        6 / 240 s on a 1 / 240 s timestep)."""
        if control_timestep is None:
            return True
        steps = control_timestep / self.world_timestep
        return round(steps) >= 1 and abs(steps - round(steps)) < 1e-9

    def tensor(self, name: str) -> torch.Tensor:
        """A simulation array by MuJoCo's name (``qpos``, ``qvel``, ``ctrl``, ``xpos``, ``xfrc_applied``, ...) as a
        torch tensor, (nworld, ...): a view, so writing to it writes to the simulation."""
        if name not in self._tensors:
            self._tensors[name] = wp.to_torch(getattr(self.data, name))
        return self._tensors[name]

    def _replay(self, fn: Callable) -> None:
        if fn not in self._graphs:
            with wp.ScopedStream(self.stream), wp.ScopedCapture() as capture:  # ScopedStream(None): no change
                fn(self.model, self.data)
            self._graphs[fn] = capture.graph
        wp.capture_launch(self._graphs[fn], stream=self.stream)

    def wait(self) -> None:
        """Makes this world's stream wait, on the GPU, for the work queued on torch's current stream so far (e.g. writes
        through world.tensor(...)); call it before step() or after_reset(). Without a stream it does nothing."""
        if self.stream is not None:
            self.stream.wait_stream(wp.stream_from_torch(self.device))

    def join(self) -> None:
        """Makes torch's current stream wait, on the GPU (the host does not wait), for the step and recomputation
        launched on this world's stream so far; call it before torch reads or writes world.tensor(...) again (a write
        could otherwise land before a pending step has read the old value). Without a stream it does nothing."""
        if self.stream is not None:
            wp.stream_from_torch(self.device).wait_stream(self.stream)

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

    after_reload = after_reset  # the model is fixed, so a reload is a reset
