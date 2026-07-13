"""Pinocchio IK helpers for arm EEF pose feasibility checks.

Requires pinocchio + casadi in the project conda env (`physgnosis`).
Point ARM_IK_DIR / ARM_URDF / ARM_ROS_SRC at your robot's Pinocchio scripts + URDF
(reference lab used a Piper Pinocchio tree).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        # Back-compat aliases from older configs
        aliases = {
            "ARM_IK_DIR": "PIPER_PINOCCHIO_DIR",
            "ARM_URDF": "PIPER_URDF",
            "ARM_ROS_SRC": "PIPER_ROS_SRC",
        }
        alt = aliases.get(name)
        if alt:
            val = os.environ.get(alt, "").strip()
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is required for IK. "
            "Set it in configs/paths.env to your arm's Pinocchio/URDF paths."
        )
    return val


_PIK_DIR = os.environ.get("ARM_IK_DIR") or os.environ.get("PIPER_PINOCCHIO_DIR") or ""
_DEFAULT_URDF = os.environ.get("ARM_URDF") or os.environ.get("PIPER_URDF") or ""
_DEFAULT_PKG = os.environ.get("ARM_ROS_SRC") or os.environ.get("PIPER_ROS_SRC") or ""


@dataclass(frozen=True)
class PlaceIkOptions:
    """Limits for IK-based lift/transport pose adjustment before place."""

    # At most this much below --place-lift-offset-m (e.g. 0.20 -> min 0.17 when 0.03).
    max_lift_reduction_m: float = 0.03
    lift_step_m: float = 0.01
    # Step-3 transport Z lowering (legacy) is off by default — it caused table collisions.
    allow_transport_z_lower: bool = False
    max_transport_z_lower_m: float = 0.02
    transport_z_step_m: float = 0.01
    # Never place transport below grasp_z + this margin (m).
    transport_min_above_grasp_m: float = 0.05
    # At most this much below --place-lower-before-release-m when lower IK fails.
    max_lower_reduction_m: float = 0.05
    lower_step_m: float = 0.01

    @classmethod
    def strict(cls) -> PlaceIkOptions:
        """No lift/transport/lower adjustment; fail if pose is IK-infeasible."""
        return cls(max_lift_reduction_m=0.0, allow_transport_z_lower=False, max_lower_reduction_m=0.0)


def build_ik_solver(*, urdf_path: str = "", package_dir: str = "", visualize: bool = False):
    pik_dir = _PIK_DIR
    urdf_path = urdf_path or _DEFAULT_URDF
    package_dir = package_dir or _DEFAULT_PKG
    if not pik_dir or not urdf_path or not package_dir:
        raise RuntimeError(
            "ARM_IK_DIR, ARM_URDF, and ARM_ROS_SRC must be set in configs/paths.env "
            "to your arm's Pinocchio scripts and URDF (reference lab used a Piper tree)."
        )
    if pik_dir not in sys.path:
        sys.path.insert(0, pik_dir)
    from ik import Arm_IK

    return Arm_IK(
        urdf_path=urdf_path,
        package_dir=package_dir,
        enable_visualization=visualize,
    )


@dataclass(frozen=True)
class PlacePosePlan:
    """Resolved place EEF poses, IK compromises, and cached joint solutions."""

    lifted_xyzrpy: np.ndarray
    transport_xyzrpy: np.ndarray
    lower_xyzrpy: np.ndarray
    actual_lift_m: float
    actual_lower_m: float
    transport_z_lower_m: float = 0.0
    transport_rpy_blend: float = 0.0
    compromises: tuple[str, ...] = field(default_factory=tuple)
    joint_lift: np.ndarray | None = None
    joint_transport: np.ndarray | None = None
    joint_lower: np.ndarray | None = None


_BASE_UP_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
# AnyGrasp TCP +Z (wrist camera) = Piper flange -X in arm base frame.
_R_TCP_FLANGE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
# TCP origin in flange frame (from run_grasp T_TCP_FLANGE; ~9 cm along flange +Z).
_TCP_OFFSET_IN_FLANGE = np.array([0.0, 0.0, 0.09], dtype=np.float64)


def _rpy_rad_to_R(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(x) for x in np.asarray(rpy, dtype=np.float64).reshape(3))
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def tcp_xyz_from_flange_xyzrpy(xyzrpy: Sequence[float]) -> np.ndarray:
    """AnyGrasp TCP position in arm base frame from a Piper flange xyzrpy pose."""
    v = np.asarray(xyzrpy, dtype=np.float64).reshape(6)
    return v[:3] + _rpy_rad_to_R(v[3:]) @ _TCP_OFFSET_IN_FLANGE


def flange_xyz_from_tcp_xy(
    tcp_x: float,
    tcp_y: float,
    flange_rpy: Sequence[float],
    *,
    flange_z: float,
) -> np.ndarray:
    """Flange XYZ in arm base so TCP xy matches (tcp_x, tcp_y) at the given flange z."""
    R = _rpy_rad_to_R(flange_rpy)
    flange_xyz = np.array([tcp_x, tcp_y, float(flange_z)], dtype=np.float64) - R @ _TCP_OFFSET_IN_FLANGE
    flange_xyz[2] = float(flange_z)
    return flange_xyz


def flange_camera_up_dot(xyzrpy: Sequence[float]) -> float:
    """Dot(wrist_camera_axis, arm_base_up). Positive => camera points upward."""
    v = np.asarray(xyzrpy, dtype=np.float64).reshape(6)
    roll, pitch, yaw = (float(x) for x in v[3:])
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    r_flange = Rz @ Ry @ Rx
    cam = r_flange @ _R_TCP_FLANGE.T @ np.array([0.0, 0.0, 1.0])
    n = float(np.linalg.norm(cam))
    if n < 1e-12:
        return 1.0
    return float(np.dot(cam / n, _BASE_UP_AXIS))


def canonicalize_flange_xyzrpy_camera_up(xyzrpy: np.ndarray) -> tuple[np.ndarray, bool]:
    """Flip 180° about approach if the symmetric parallel-jaw branch has camera more upward.

    Roll is realized jointly by wrist joints (j4, j5, …); fix the flange pose here and
    let IK distribute the rotation across the wrist chain.
    """
    v = np.asarray(xyzrpy, dtype=np.float64).reshape(6)
    up0 = flange_camera_up_dot(v)
    if up0 >= 1.0 - 1e-9:
        return v, False

    roll, pitch, yaw = (float(x) for x in v[3:])
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    r_flange = Rz @ Ry @ Rx
    r_tcp = r_flange @ _R_TCP_FLANGE.T
    r_tcp2 = r_tcp @ np.diag([1.0, -1.0, -1.0])
    r_flange2 = r_tcp2 @ _R_TCP_FLANGE
    cam = r_tcp2 @ np.array([0.0, 0.0, 1.0])
    up1 = float(np.dot(cam / max(float(np.linalg.norm(cam)), 1e-12), _BASE_UP_AXIS))
    if up1 > up0 + 1e-9:
        sy2 = np.sqrt(r_flange2[0, 0] ** 2 + r_flange2[1, 0] ** 2)
        if sy2 >= 1e-6:
            roll2 = np.arctan2(r_flange2[2, 1], r_flange2[2, 2])
            pitch2 = np.arctan2(-r_flange2[2, 0], sy2)
            yaw2 = np.arctan2(r_flange2[1, 0], r_flange2[0, 0])
        else:
            roll2 = np.arctan2(-r_flange2[1, 2], r_flange2[1, 1])
            pitch2 = np.arctan2(-r_flange2[2, 0], sy2)
            yaw2 = 0.0
        out = v.copy()
        out[3:6] = [roll2, pitch2, yaw2]
        return out, True
    return v, False


def solve_xyzrpy_ik(
    xyzrpy: Sequence[float],
    *,
    gripper: float,
    arm_ik,
) -> tuple[np.ndarray | None, bool]:
    from eef_to_joint_npy import xyzrpy_to_joint7

    pose, _ = canonicalize_flange_xyzrpy_camera_up(
        np.asarray(xyzrpy, dtype=np.float64).reshape(6),
    )
    joints7, ok = xyzrpy_to_joint7(pose, gripper=gripper, arm_ik=arm_ik)
    if not ok or joints7 is None:
        return None, False
    return joints7, True


def check_xyzrpy_ik(
    xyzrpy: Sequence[float],
    *,
    gripper: float,
    arm_ik,
) -> bool:
    _joints7, ok = solve_xyzrpy_ik(xyzrpy, gripper=gripper, arm_ik=arm_ik)
    return ok


def blend_rpy(a: Sequence[float], b: Sequence[float], t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    av = np.asarray(a, dtype=np.float64).reshape(3)
    bv = np.asarray(b, dtype=np.float64).reshape(3)
    return (1.0 - t) * av + t * bv


def _lift_xyzrpy(grasp_xyzrpy: np.ndarray, *, offset_m: float) -> np.ndarray:
    grasp = np.asarray(grasp_xyzrpy, dtype=np.float64).reshape(6)
    lift_xyz = grasp[:3].copy()
    lift_xyz[2] += float(offset_m)
    return np.concatenate([lift_xyz, grasp[3:]])


def find_feasible_lift(
    grasp_xyzrpy: np.ndarray,
    *,
    requested_offset_m: float,
    gripper: float,
    arm_ik,
    options: PlaceIkOptions | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (lifted_xyzrpy, actual_offset_m, joint7), reducing lift until IK succeeds."""
    opts = options or PlaceIkOptions()
    grasp = np.asarray(grasp_xyzrpy, dtype=np.float64).reshape(6)
    requested = float(requested_offset_m)
    min_offset = max(0.0, requested - float(opts.max_lift_reduction_m))
    offset = requested
    step_m = float(opts.lift_step_m)
    while offset + 1e-9 >= min_offset:
        lifted = _lift_xyzrpy(grasp, offset_m=offset)
        joints7, ok = solve_xyzrpy_ik(lifted, gripper=gripper, arm_ik=arm_ik)
        if ok and joints7 is not None:
            return lifted, offset, joints7
        offset -= step_m
    raise RuntimeError(
        f"lift IK failed at offset>={min_offset:.3f}m "
        f"(requested {requested:.3f}m, max reduction {opts.max_lift_reduction_m:.3f}m); "
        f"grasp z={grasp[2]:.3f}. Try a closer place target, smaller lift, or --strict-place-ik off with "
        f"a larger --place-ik-max-lift-reduction-m."
    )


def find_feasible_transport(
    grasp_xyzrpy: np.ndarray,
    *,
    place_x: float,
    place_y: float,
    lifted_xyzrpy: np.ndarray,
    gripper: float,
    arm_ik,
    carry_rpy_ref: Sequence[float] | None = None,
    options: PlaceIkOptions | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Find transport pose with IK; return (transport_xyzrpy, z_lowered_m, rpy_blend, joint7)."""
    opts = options or PlaceIkOptions()
    grasp = np.asarray(grasp_xyzrpy, dtype=np.float64).reshape(6)
    lifted = np.asarray(lifted_xyzrpy, dtype=np.float64).reshape(6)
    z = float(lifted[2])
    z_floor = float(grasp[2]) + float(opts.transport_min_above_grasp_m)
    carry_ref = np.asarray(carry_rpy_ref if carry_rpy_ref is not None else grasp[3:], dtype=np.float64)

    def _try(z_try: float, blend: float) -> tuple[np.ndarray, np.ndarray] | None:
        if z_try + 1e-9 < z_floor:
            return None
        rpy = blend_rpy(grasp[3:], carry_ref, blend) if blend > 0.0 else np.asarray(grasp[3:], dtype=np.float64)
        flange_xyz = flange_xyz_from_tcp_xy(place_x, place_y, rpy, flange_z=z_try)
        candidate = np.array([*flange_xyz, *rpy], dtype=np.float64)
        candidate, _ = canonicalize_flange_xyzrpy_camera_up(candidate)
        if flange_camera_up_dot(candidate) < -1e-6:
            return None
        joints7, ok = solve_xyzrpy_ik(candidate, gripper=gripper, arm_ik=arm_ik)
        if ok and joints7 is not None:
            return candidate, joints7
        return None

    result = _try(z, 0.0)
    if result is not None:
        candidate, joints7 = result
        return candidate, 0.0, 0.0, joints7

    for blend in (0.2, 0.4, 0.6, 0.8, 1.0):
        result = _try(z, blend)
        if result is not None:
            candidate, joints7 = result
            return candidate, 0.0, blend, joints7

    if opts.allow_transport_z_lower and opts.max_transport_z_lower_m > 0.0:
        step = float(opts.transport_z_step_m)
        max_dz = float(opts.max_transport_z_lower_m)
        dz = step
        while dz <= max_dz + 1e-9:
            z_try = z - dz
            for blend in (0.0, 0.4, 0.8, 1.0):
                result = _try(z_try, blend)
                if result is not None:
                    candidate, joints7 = result
                    return candidate, dz, blend, joints7
            dz += step

    raise RuntimeError(
        f"transport IK failed for TCP place=({place_x:.3f}, {place_y:.3f}, flange_z={z:.3f}); "
        "try a closer place target, increase --place-lift-offset-m, or enable "
        "--place-ik-allow-transport-z-lower with a small --place-ik-max-transport-z-lower-m"
    )


def find_feasible_lower(
    transport_xyzrpy: np.ndarray,
    *,
    requested_lower_m: float,
    gripper: float,
    arm_ik,
    options: PlaceIkOptions | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (lower_xyzrpy, actual_lower_m, joint7), reducing lower distance until IK succeeds."""
    requested = float(requested_lower_m)
    transport = np.asarray(transport_xyzrpy, dtype=np.float64).reshape(6)
    if requested <= 0.0:
        joints7, ok = solve_xyzrpy_ik(transport, gripper=gripper, arm_ik=arm_ik)
        if not ok or joints7 is None:
            raise RuntimeError(
                f"transport IK failed at ({transport[0]:.3f}, {transport[1]:.3f}, z={transport[2]:.3f})"
            )
        return transport, 0.0, joints7

    opts = options or PlaceIkOptions()
    min_lower = max(0.0, requested - float(opts.max_lower_reduction_m))
    lower_m = requested
    step_m = float(opts.lower_step_m)
    while lower_m + 1e-9 >= min_lower:
        lower_xyz = transport[:3].copy()
        lower_xyz[2] -= lower_m
        candidate = np.concatenate([lower_xyz, transport[3:]])
        joints7, ok = solve_xyzrpy_ik(candidate, gripper=gripper, arm_ik=arm_ik)
        if ok and joints7 is not None:
            return candidate, lower_m, joints7
        lower_m -= step_m

    raise RuntimeError(
        f"lower IK failed for place=({transport[0]:.3f}, {transport[1]:.3f}, "
        f"transport_z={transport[2]:.3f}, requested lower={requested:.3f}m, "
        f"min tried={min_lower:.3f}m); try a closer place target, smaller "
        f"--place-lower-before-release-m, or increase --place-ik-max-lower-reduction-m"
    )


def validate_grasp_sequence_ik(
    poses: Sequence[tuple[str, np.ndarray, float]],
    *,
    arm_ik,
) -> list[tuple[str, np.ndarray, float]]:
    """Check IK for named poses; return list of failures [(name, xyzrpy, gripper)]."""
    failures: list[tuple[str, np.ndarray, float]] = []
    for name, xyzrpy, gripper in poses:
        if not check_xyzrpy_ik(xyzrpy, gripper=gripper, arm_ik=arm_ik):
            failures.append((name, np.asarray(xyzrpy, dtype=np.float64), float(gripper)))
    return failures
