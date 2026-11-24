"""Nodes over named parts of an MjWarpWorld's model (joints, actuators, bodies).

Lifecycle priorities follow UniEnv's bands (higher runs first): robots and objects 50-80, tasks 0-50, sensors below
0. A node only takes part in the phases whose priority sets are non-empty.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import WorldNode

from ..world import MjWarpWorld


class MjWarpNode(WorldNode):
    """A node of an MjWarpWorld; ``model`` is the world's MuJoCo model, for looking up names."""

    def __init__(self, world: MjWarpWorld, name: str):
        self.world, self.name = world, name
        self.model = world.mj_model


def unbounded_obs_space(world: MjWarpWorld, metadata: Mapping[str, Mapping[str, Any]] | None = None,
                        **shapes: tuple[int, ...]) -> DictSpace:  # fmt: skip
    """A dict of unbounded float boxes, one per key, each (nworld, *shape); ``metadata``: the space metadata of some
    of the keys (see space_metadata)."""
    metadata = metadata or {}
    if unknown := set(metadata) - set(shapes):
        raise KeyError(f"metadata for keys without a shape: {sorted(unknown)}")
    boxes = {k: BoxSpace(PyTorchComputeBackend, -torch.inf, torch.inf, torch.float32, device=world.device,
                         shape=(world.batch_size, *s), metadata=metadata.get(k)) for k, s in shapes.items()}
    return DictSpace(PyTorchComputeBackend, boxes, device=world.device)
