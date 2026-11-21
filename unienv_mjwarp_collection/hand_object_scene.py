"""Hand + object -> one MuJoCo model, used as is by the CPU simulator and MuJoCo Warp.

Hand: its MJCF, with a wrist on the palm body (slides x, y, z then hinges x, y, z, i.e. intrinsic XYZ angles) and PD
position actuators on the wrist joints (Hand.wrist_*_gains). Finger PD targets are not limited to the joint ranges: a
joint at its limit keeps pushing toward a target past it, up to its torque limit, as a servo does (Dexonomy's squeeze
targets often lie past a knuckle's limit; clipping them unbalances the grasp's forces).
Object: convex mesh pieces times a scale, a body of the given mass (uniform density): free,
at the world origin, or fixed to the world at a given pose (for pushing on it; contacts against it then see only the
hand's inertia). Only hand-object contacts are on (no hand self-contacts), all with the friction, contact dimension and
softness of the SimConfig.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .hands import Hand
from .sim_config import SimConfig

WRIST_JOINTS = ("wrist_x", "wrist_y", "wrist_z", "wrist_rx", "wrist_ry", "wrist_rz")
HAND_GROUP, OBJECT_GROUP = 2, 1  # contype / conaffinity bits: hand geoms touch object geoms only


def wrist_q(palm_pose: np.ndarray) -> np.ndarray:
    """Palm poses (..., 7: position, wxyz) -> wrist joint values (..., 6): the position, then intrinsic x, y, z angles
    (the wrist's slides and hinges)."""
    angles = Rotation.from_quat(palm_pose[..., 3:].reshape(-1, 4), scalar_first=True).as_euler("XYZ")
    return np.concatenate([palm_pose[..., :3], angles.reshape(*palm_pose.shape[:-1], 3)], -1)


def build_model(hand: Hand, pieces: Sequence[Path], scale: float, mass: float, cfg: SimConfig,
                fixed_pose: np.ndarray | None = None) -> mujoco.MjModel:  # fmt: skip
    """pieces: the object's convex mesh files. fixed_pose (7,: position, wxyz): the object fixed to the world there
    (no joint); None: free."""
    spec = mujoco.MjSpec.from_file(str(hand.mjcf))
    o = spec.option
    o.timestep = 1.0 / cfg.physics_hz
    o.gravity = [0.0, 0.0, -9.81 if cfg.gravity else 0.0]
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if cfg.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.impratio = cfg.impratio
    o.noslip_iterations = cfg.noslip_iterations

    for a in spec.actuators:
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    palm = spec.body(hand.palm)
    if list(palm.joints):
        raise ValueError(f"{hand.mjcf}: the palm body already has joints")
    for i, name in enumerate(WRIST_JOINTS):
        kind = mujoco.mjtJoint.mjJNT_SLIDE if i < 3 else mujoco.mjtJoint.mjJNT_HINGE
        palm.add_joint(name=name, type=kind, axis=np.eye(3)[i % 3])
        kp, kd = hand.wrist_slide_gains if i < 3 else hand.wrist_hinge_gains
        spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT).set_to_position(kp=kp, kv=kd)
    for b in spec.bodies:  # the hand's links feel no gravity (it is held by the wrist drives); added before the object
        b.gravcomp = 1.0

    body = spec.worldbody.add_body(name="object")
    if fixed_pose is None:
        body.add_freejoint(name="object")
    else:
        body.pos, body.quat = fixed_pose[:3], fixed_pose[3:]
    volume = sum(trimesh.load(p, force="mesh").convex_hull.volume for p in pieces) * scale**3
    for i, p in enumerate(pieces):
        spec.add_mesh(name=f"object_piece{i}", file=str(p), scale=[scale] * 3)
        body.add_geom(
            name=f"object_piece{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"object_piece{i}", density=mass / volume
        )

    for g in spec.geoms:
        if g.contype == 0 and g.conaffinity == 0:  # visual-only geoms stay that way
            continue
        on_object = g.name.startswith("object_piece")
        g.contype, g.conaffinity = (OBJECT_GROUP, HAND_GROUP) if on_object else (HAND_GROUP, OBJECT_GROUP)
        g.condim = cfg.condim
        g.friction = [cfg.friction, cfg.torsional_friction, cfg.rolling_friction]
        g.solref = cfg.object_solref if on_object else cfg.hand_solref
        g.solimp = cfg.object_solimp if on_object else cfg.hand_solimp
    return spec.compile()


@dataclass(frozen=True)
class SceneIndex:
    """Where a scene model keeps the hand's joints (fingers then wrist), their actuators and the object, and which
    glove link (index into the glove layout's links, -1 for none) each body and geom belongs to."""

    qpos_adr: np.ndarray  # (J + 6,)
    dof_adr: np.ndarray  # (J + 6,)
    joint: np.ndarray  # (J + 6,) joint ids
    act: np.ndarray  # (J + 6,) actuator ids
    obj_qpos: int  # -1 for a fixed object
    obj_dof: int  # -1 for a fixed object
    obj_body: int
    link_body: np.ndarray  # (L,) body id of each glove link
    geom_link: np.ndarray  # (ngeom,)
    geom_on_object: np.ndarray  # (ngeom,) bool

    @classmethod
    def build(cls, m: mujoco.MjModel, hand: Hand, glove_links: tuple[str, ...]) -> "SceneIndex":
        joint = np.array([m.joint(n).id for n in hand.joints + WRIST_JOINTS])
        joint_actuator = {int(j): a for a, j in enumerate(m.actuator_trnid[:, 0])}
        link_body = np.array([m.body(n).id for n in glove_links])
        body_link = {b: i for i, b in enumerate(link_body)}
        obj_body = m.body("object").id
        free = m.body_jntnum[obj_body] > 0
        return cls(
            qpos_adr=m.jnt_qposadr[joint],
            dof_adr=m.jnt_dofadr[joint],
            joint=joint,
            act=np.array([joint_actuator[j] for j in joint]),
            obj_qpos=int(m.jnt_qposadr[m.joint("object").id]) if free else -1,
            obj_dof=int(m.jnt_dofadr[m.joint("object").id]) if free else -1,
            obj_body=obj_body,
            link_body=link_body,
            geom_link=np.array([body_link.get(b, -1) for b in m.geom_bodyid]),
            geom_on_object=m.geom_bodyid == obj_body,
        )
