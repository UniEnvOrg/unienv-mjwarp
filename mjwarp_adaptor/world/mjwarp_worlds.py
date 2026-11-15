"""MjWarpWorlds: several MjWarpWorlds (e.g. one per scene) stepped together, each on its own CUDA stream, so the GPU
overlaps them; their contacts read out in one list.

A world on its own stream orders itself with torch in step() and after_reset() (MjWarpWorld's docstring). Done world by
world, that serialises them: torch's default stream is CUDA's legacy stream, and the event a world's wait records on it
waits for every stream's work so far, including the previous world's launch. So the group makes every world wait first,
then launches them all, then makes torch wait for each. Torch reads and writes every world's arrays right after each
call, as with one world.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import mujoco_warp as mjw
import torch
import warp as wp

from .mjwarp_world import Contacts, MjWarpWorld


class MjWarpWorlds:
    def __init__(self, worlds: Sequence[MjWarpWorld]):
        """worlds: a world made without a stream gets its own; two worlds may not share one."""
        self.worlds = list(worlds)
        for w in self.worlds:
            if w.stream is None:
                w.stream = wp.Stream(wp.get_device())
        if len({id(w.stream) for w in self.worlds}) < len(self.worlds):
            raise ValueError("two worlds share a stream: they would run one after the other")
        self.device = self.worlds[0].device
        # contacts(): geom ids and world ids numbered across the worlds, in order
        self._first_geom = [0]
        self._first_world = [0]
        for w in self.worlds[:-1]:
            self._first_geom.append(self._first_geom[-1] + w.mj_model.ngeom)
            self._first_world.append(self._first_world[-1] + w.batch_size)

    def _run(self, worlds: Sequence[MjWarpWorld], launch: Callable[[MjWarpWorld], None], n: int = 1) -> None:
        for w in worlds:  # all before any launch (the module docstring)
            w._wait()
        for _ in range(n):
            for w in worlds:
                launch(w)
        for w in worlds:
            w._join()

    def step(self, n: int = 1) -> None:
        """n physics steps of every world, the worlds interleaved step by step."""
        self._run(self.worlds, lambda w: w._replay(mjw.step), n)

    def after_reset(self, which: Sequence[int] | None = None) -> None:
        """MjWarpWorld.after_reset of the worlds at the indices ``which`` (None: all)."""
        worlds = self.worlds if which is None else [self.worlds[i] for i in which]
        self._run(worlds, lambda w: w._replay(mjw.forward))

    def contact_counts(self) -> list[int]:
        """MjWarpWorld.contact_count of every world, read back to the host once for all of them."""
        counts = torch.cat([w.tensor("nacon") for w in self.worlds]).tolist()
        for w, n in zip(self.worlds, counts):
            if n > w.data.naconmax:
                raise RuntimeError(f"{n} contacts > the buffer's {w.data.naconmax}: raise contacts_per_world")
        return counts

    def contacts(self, world_frame: bool = False) -> Contacts:
        """MjWarpWorld.contacts of every world in one list (one host readback of the counts), with geom ids offset by
        the geoms of the worlds before (world i's geom g is sum(ngeom of worlds < i) + g) and world ids by their
        batch sizes."""
        parts = []
        for w, n, g0, w0 in zip(self.worlds, self.contact_counts(), self._first_geom, self._first_world):
            t = w.tensor
            parts.append(Contacts(t("contact.geom")[:n] + g0, t("contact.worldid")[:n] + w0,
                                  t("contact.frame").view(-1, 3, 3)[:n], t("contact.pos")[:n],
                                  w.contact_force(world_frame)[:n]))  # fmt: skip
        return Contacts(*(torch.cat(field) for field in zip(*parts)))
