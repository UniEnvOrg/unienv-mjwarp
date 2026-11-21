"""A hand: its MuJoCo model, glove layout and finger groups (assets/hands/<name>.yaml).

The hand MJCF is the single source of the finger joints (one position actuator per joint, in actuator order), their
limits, PD gains and torque limits. The wrist (3 slides + 3 hinges) is added by hand_object_scene, with the gains
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml

HANDS_DIR = Path(__file__).resolve().parent / "assets" / "hands"


@dataclass(frozen=True)
class Hand:
    name: str
    mjcf: Path
    glove: Path
    palm: str  # palm body name
    fingers: dict[str, tuple[str, ...]]  # finger -> its bodies, base to tip (thumb, index, middle, ring, pinky)
    wrist_slide_gains: tuple[float, float]  # kp N/m, kd N s/m
    wrist_hinge_gains: tuple[float, float]  # kp N m/rad, kd N m s/rad
    dexonomy: str  # the hand's name in Dexonomy
    joints: tuple[str, ...]  # finger joints, actuator order
    limits: np.ndarray  # (J, 2) rad
    kp: np.ndarray  # (J,) N m/rad
    kd: np.ndarray  # (J,) N m s/rad
    tau_max: np.ndarray  # (J,) N m

    @property
    def links(self) -> tuple[str, ...]:
        """Bodies that can touch objects: the palm, then each finger's bodies."""
        return (self.palm,) + tuple(b for bodies in self.fingers.values() for b in bodies)


def load_hand(name: str) -> Hand:
    cfg = yaml.safe_load((HANDS_DIR / f"{name}.yaml").read_text())
    mjcf = Path(cfg["mjcf"]).expanduser()
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    if m.nu == 0:
        raise ValueError(f"{mjcf} has no actuators")
    joint_ids = m.actuator_trnid[:, 0]
    joints = tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids)
    if not (m.actuator_forcelimited.all() and m.jnt_limited[joint_ids].all()):
        raise ValueError(f"{mjcf}: every actuator needs a force range and every joint a range")
    hand = Hand(
        name=name,
        mjcf=mjcf,
        glove=Path(cfg["glove"]).expanduser(),
        palm=cfg["palm"],
        fingers={f: tuple(b) for f, b in cfg["fingers"].items()},
        wrist_slide_gains=tuple(cfg["wrist_slide_gains"]),
        wrist_hinge_gains=tuple(cfg["wrist_hinge_gains"]),
        dexonomy=cfg["dexonomy"],
        joints=joints,
        limits=m.jnt_range[joint_ids].copy(),
        kp=m.actuator_gainprm[:, 0].copy(),
        kd=-m.actuator_biasprm[:, 2].copy(),
        tau_max=m.actuator_forcerange[:, 1].copy(),
    )
    missing = [b for b in hand.links if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) < 0]
    if missing:
        raise ValueError(f"{name}.yaml names bodies not in {mjcf}: {missing}")
    return hand


def collision_meshes(m: mujoco.MjModel) -> dict[str, trimesh.Trimesh]:
    """Each body's collision geometry as one triangle mesh, in the body's frame (meshes as they are; capsules,
    cylinders, boxes and spheres tessellated)."""
    G = mujoco.mjtGeom
    out = {}
    for g in range(m.ngeom):
        if not (m.geom_contype[g] | m.geom_conaffinity[g]):
            continue
        t, size = m.geom_type[g], m.geom_size[g]
        if t == G.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            local = trimesh.Trimesh(
                m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]],
                m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]],
            )
        elif t == G.mjGEOM_CAPSULE:
            local = trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[24, 24])
        elif t == G.mjGEOM_CYLINDER:
            local = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
        elif t == G.mjGEOM_BOX:
            local = trimesh.creation.box(extents=2 * size)
        elif t == G.mjGEOM_SPHERE:
            local = trimesh.creation.icosphere(subdivisions=3, radius=size[0])
        else:
            raise ValueError(f"geom {g}: type {G(t).name} not handled")
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        mesh = trimesh.Trimesh(local.vertices @ R.reshape(3, 3).T + m.geom_pos[g], local.faces, process=True)
        name = m.body(m.geom_bodyid[g]).name
        out[name] = trimesh.util.concatenate([out[name], mesh]) if name in out else mesh
    return out


def palm_side(m: mujoco.MjModel, body: int) -> np.ndarray:
    """Body frame direction the body's mesh centre moves when the flexing joints from the palm up to it increase."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    c0 = d.xpos[body] + d.xmat[body].reshape(3, 3) @ m.body_ipos[body]
    b = body
    while b > 0:
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            lo, hi = m.jnt_range[j]
            if abs(lo + hi) > 0.1:  # not a symmetric (abduction) joint
                d.qpos[m.jnt_qposadr[j]] += 1e-3
        b = m.body_parentid[b]
    mujoco.mj_kinematics(m, d)
    c1 = d.xpos[body] + d.xmat[body].reshape(3, 3) @ m.body_ipos[body]
    v = d.xmat[body].reshape(3, 3).T @ (c1 - c0)  # moved since the joints rotate it: back in the body's frame at rest
    return v / np.linalg.norm(v)


def link_axes(mesh: trimesh.Trimesh, side: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A link's axes in its body frame: along (its mesh's long axis, pointing away from the joint at the body
    origin), side (`side`, e.g. palm_side, made perpendicular to it) and lateral (along x side)."""
    v = mesh.vertices
    along = np.linalg.svd(v - v.mean(0))[2][0]
    along *= np.sign(along @ v.mean(0))
    side = side - (side @ along) * along
    side /= np.linalg.norm(side)
    return along, side, np.cross(along, side)
