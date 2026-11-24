"""Space metadata of the nodes' observations and actions, in UniEnv's vocabulary
(``unienv_interface.space.space_utils.metadata_utils``), derived from the MuJoCo model.

Units are only given where the model determines them: per joint from ``jnt_type`` (hinge: rad, rad/s, N*m; slide: m,
m/s, N), and for actuators only when every one of them is a plain position, velocity or motor actuator on a hinge or
slide joint (see ``actuator_control``); otherwise ``control`` and ``units`` are left out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np
from unienv_interface.space.space_utils.metadata_utils import action_metadata, metadata, proprio_metadata

__all__ = ["JOINT_UNITS", "XYZ_NAMES", "QUAT_WXYZ_NAMES", "POSE_NAMES", "POSE_UNITS", "VELOCITY_NAMES",
           "VELOCITY_UNITS", "joint_units", "joint_state_metadata", "actuator_control", "actuator_action_metadata",
           "body_pose_metadata", "free_body_velocity_metadata", "contact_force_metadata"]  # fmt: skip

_HINGE, _SLIDE = int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
JOINT_UNITS: dict[int, dict[str, str]] = {
    _HINGE: {"position": "rad", "velocity": "rad/s", "force": "N*m"},
    _SLIDE: {"position": "m", "velocity": "m/s", "force": "N"},
}
"""``jnt_type -> quantity -> unit`` of the one-DOF joints."""

XYZ_NAMES = ["x", "y", "z"]
QUAT_WXYZ_NAMES = ["qw", "qx", "qy", "qz"]
POSE_NAMES = XYZ_NAMES + QUAT_WXYZ_NAMES
POSE_UNITS = ["m"] * 3 + ["normalized"] * 4
VELOCITY_NAMES = ["vx", "vy", "vz", "wx", "wy", "wz"]
VELOCITY_UNITS = ["m/s"] * 3 + ["rad/s"] * 3


def joint_units(model: mujoco.MjModel, joint_ids: Sequence[int], quantity: str) -> list[str] | None:
    """The unit of ``quantity`` (``"position"``, ``"velocity"`` or ``"force"``) of each joint; None unless all are
    hinge or slide joints."""
    types = [int(model.jnt_type[j]) for j in joint_ids]
    if any(t not in JOINT_UNITS for t in types):
        return None
    return [JOINT_UNITS[t][quantity] for t in types]


def joint_state_metadata(model: mujoco.MjModel, joint_ids: Sequence[int], names: Sequence[str],
                         **extra: Any) -> dict[str, dict[str, Any]]:  # fmt: skip
    """``kind="proprio"`` metadata of ``joint_position`` and ``joint_velocity`` of one-DOF joints."""
    return {f"joint_{q}": proprio_metadata(units=joint_units(model, joint_ids, q), names=list(names), **extra)
            for q in ("position", "velocity")}  # fmt: skip


def _close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-9, atol=0.0))


def actuator_control(model: mujoco.MjModel, actuator_id: int) -> str | None:
    """What an actuator's control is, if it is one of the plain cases, else None.

    ``"joint_position"`` (a position actuator: force = kp (ctrl - q) - kv qdot), ``"joint_velocity"`` (a velocity
    actuator: force = kv (ctrl - qdot)) or ``"joint_torque"`` (a motor: force = ctrl) on a hinge or slide joint, with
    gear 1, a fixed gain, no or a first-order filter's activation dynamics (the control is still the setpoint) and a
    single control. Anything else (gears, tendons, sites, integrators, muscles, user gains, ...) is None.
    """
    m, a = model, actuator_id
    if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
        return None
    if int(m.jnt_type[m.actuator_trnid[a, 0]]) not in JOINT_UNITS or not _close(m.actuator_gear[a, 0], 1.0):
        return None
    if hasattr(m, "actuator_ctrlnum") and int(m.actuator_ctrlnum[a]) != 1:
        return None
    dyn = {int(mujoco.mjtDyn.mjDYN_NONE), int(mujoco.mjtDyn.mjDYN_FILTER), int(mujoco.mjtDyn.mjDYN_FILTEREXACT)}
    if int(m.actuator_dyntype[a]) not in dyn or int(m.actuator_gaintype[a]) != int(mujoco.mjtGain.mjGAIN_FIXED):
        return None
    gain, bias, biastype = float(m.actuator_gainprm[a, 0]), m.actuator_biasprm[a, :3], int(m.actuator_biastype[a])
    if biastype == int(mujoco.mjtBias.mjBIAS_NONE):
        return "joint_torque" if _close(gain, 1.0) else None
    if biastype != int(mujoco.mjtBias.mjBIAS_AFFINE) or gain <= 0 or not _close(bias[0], 0.0):
        return None
    if _close(bias[1], -gain) and bias[2] <= 0:
        return "joint_position"
    if _close(bias[1], 0.0) and _close(bias[2], -gain):
        return "joint_velocity"
    return None


_CONTROL_QUANTITY = {"joint_position": "position", "joint_velocity": "velocity", "joint_torque": "force"}


def actuator_action_metadata(model: mujoco.MjModel, actuator_ids: Sequence[int], names: Sequence[str],
                             **extra: Any) -> dict[str, Any]:  # fmt: skip
    """``kind="action"`` metadata of absolute actuator controls; ``control`` and per-actuator ``units`` only when
    every actuator has the same ``actuator_control``."""
    controls = {actuator_control(model, a) for a in actuator_ids}
    control = controls.pop() if len(controls) == 1 else None
    units = None
    if control is not None:
        units = joint_units(model, [int(model.actuator_trnid[a, 0]) for a in actuator_ids], _CONTROL_QUANTITY[control])
    return action_metadata(control=control, mode="absolute", units=units, names=list(names), **extra)


def body_pose_metadata(**extra: Any) -> dict[str, Any]:
    """``kind="pose"`` metadata of a world frame body pose (position, wxyz quaternion), e.g. ``xpos`` + ``xquat``."""
    return metadata("pose", frame="world", rotation="quat_wxyz", units=POSE_UNITS, names=POSE_NAMES, **extra)


def free_body_velocity_metadata(**extra: Any) -> dict[str, Any]:
    """``kind="proprio"`` metadata of a free joint's velocity (``qvel``): linear in the world frame, then angular in
    the body's frame, so no single ``frame``."""
    return proprio_metadata(
        units=VELOCITY_UNITS, names=VELOCITY_NAMES,
        label="free joint velocity: linear (world frame), then angular (body frame)", **extra,
    )


def contact_force_metadata(bodies: Sequence[str], **extra: Any) -> dict[str, Any]:
    """``kind="force"`` metadata of world frame forces (..., bodies, 3); the body of each row is in ``bodies``."""
    return metadata("force", units="N", frame="world", names=["fx", "fy", "fz"], bodies=list(bodies), **extra)
