"""Physics settings, in one place. Saved with every output and checked when the output is read back, so data made
with one setting can't be used with another by mistake."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class SimConfig:
    physics_hz: int = 240
    control_hz: int = 120
    gravity: bool = False
    friction: float = 1.0  # sliding friction coefficient, hand-object contacts
    torsional_friction: float = 0.02  # m, MuJoCo's torsional coefficient (Dexonomy's grasp test)
    rolling_friction: float = 0.0001  # m
    condim: int = 4  # contact dimensions: normal, 2 sliding, torsional
    cone: str = "elliptic"  # elliptic | pyramidal
    impratio: float = 10.0
    noslip_iterations: int = 2  # MuJoCo's no-slip solver, as in Dexonomy's grasp test
    # contact softness per side, mixed equally by MuJoCo: the hand stiff as in Dexonomy, the object at MuJoCo's default
    hand_solref: tuple[float, float] = (0.005, 1.0)
    hand_solimp: tuple[float, float, float, float, float] = (0.5, 0.99, 0.0001, 0.5, 2.0)
    object_solref: tuple[float, float] = (0.02, 1.0)
    object_solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    contacts_per_world: int = 128  # MuJoCo Warp buffers
    rows_per_world: int = 1024

    def __post_init__(self):
        if self.physics_hz % self.control_hz:
            raise ValueError(f"physics_hz {self.physics_hz} is not a multiple of control_hz {self.control_hz}")
        if self.cone not in ("elliptic", "pyramidal"):
            raise ValueError(f"cone {self.cone!r}")

    @property
    def substeps(self) -> int:
        return self.physics_hz // self.control_hz

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown SimConfig fields {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def check_same(self, saved: "SimConfig | dict", what: str = "data") -> None:
        """Raise if `saved` (the settings some data were made with) differ from these."""
        saved = saved if isinstance(saved, SimConfig) else SimConfig.from_dict(saved)
        diff = {
            f.name: (getattr(saved, f.name), getattr(self, f.name))
            for f in fields(self)
            if getattr(saved, f.name) != getattr(self, f.name)
        }
        if diff:
            raise ValueError(f"{what} made with different physics (saved, now): {diff}")
