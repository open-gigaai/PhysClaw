from __future__ import annotations

import os
import sys

if '--quiet' in sys.argv:
    os.environ.setdefault('OMP_NUM_THREADS', '16')
    import logging
    import warnings

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.ERROR)

import argparse
import contextlib
import logging
import warnings
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from PIL import Image

AnyGrasp: Any = None
GraspGroup: Any = None


def _import_anygrasp_deps() -> None:
    global AnyGrasp, GraspGroup
    if AnyGrasp is not None:
        return
    from gsnet import AnyGrasp as _AnyGrasp
    from graspnetAPI import GraspGroup as _GraspGroup

    AnyGrasp = _AnyGrasp
    GraspGroup = _GraspGroup

_VERBOSE = True


def _log(*args, **kwargs) -> None:
    if _VERBOSE:
        print(*args, **kwargs)


@contextlib.contextmanager
def _silence_stdio() -> Iterator[None]:
    """Silence stdout/stderr at the OS fd level (covers C++ extensions)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)


def _configure_quiet_mode(quiet: bool, *, debug: bool = False) -> None:
    global _VERBOSE
    _VERBOSE = debug or not quiet
    if not quiet or debug:
        return
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    os.environ.setdefault('OMP_NUM_THREADS', '16')

# =========================
# Per-arm hand-eye calibration (cam2base / cam2gripper samples)
# =========================
def _rot_y_deg(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _apply_cam2base_correction(
    R: np.ndarray,
    t: np.ndarray,
    *,
    rot_y_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a small rotation correction in the arm base frame."""
    if rot_y_deg == 0.0:
        return R, t
    R_delta = _rot_y_deg(rot_y_deg)
    return R_delta @ R, R_delta @ t


_LEFT_CAM2BASE_R = np.array(
    [[-0.01349245, -0.83721614, 0.54670567],
     [-0.99983488, 0.00464049, -0.0175691],
     [0.01217215, -0.54685245, -0.83714051]],
    dtype=np.float64,
)
_LEFT_CAM2BASE_T = np.array([-0.00679678, -0.29738849, 0.64202349], dtype=np.float64)
_LEFT_CAM2BASE_R, _LEFT_CAM2BASE_T = _apply_cam2base_correction(
    _LEFT_CAM2BASE_R,
    _LEFT_CAM2BASE_T,
    rot_y_deg=4.0,
)

_ARM_CONFIGS: Dict[str, Dict[str, Any]] = {
    'left': {
        'label': 'left_arm_base',
        'cam2base_samples': ((_LEFT_CAM2BASE_R, _LEFT_CAM2BASE_T),)
    },
    'right': {
        'label': 'right_arm_base',
        'cam2base_samples': ((
            np.array([[-0.1189392, -0.86399853, 0.48924432],
                      [-0.99216738, 0.12236648, -0.02510633],
                      [-0.03817527, -0.48839838, -0.87178533]], dtype=np.float64),
            np.array([0.03467391, 0.27043642, 0.64287032], dtype=np.float64)
        ),)
    }
}

# AnyGrasp TCP -> flange (constant for all scenes in the reference setup).
# Replace via CAMERA_EXTRINSICS_JSON "T_tcp_flange" when your gripper frame differs.
# AnyGrasp TCP: x=approach, y=gripper opening, z=gripper height.
# Reference Piper PosCmd flange: approach is +Z.
# p_tcp = T_TCP_FLANGE @ p_flange
# R maps AnyGrasp +X (approach) to flange +Z; Y opening axes align; det(R)=+1.
T_TCP_FLANGE = np.array(
    [[0, 0, 1, -0.09],
     [0, 1, 0, 0.0],
     [-1, 0, 0, 0.0],
     [0, 0, 0, 1.0]],
    dtype=np.float64,
)

PREGRASP_OFFSET_M = 0.10

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EEF_POSE_XYZRPY_NPY = os.path.join(_SCRIPT_DIR, 'eef_pose_xyzrpy.npy')
_GSNET_DIR = os.environ.get('GSNET_DIR', '').strip()
if _GSNET_DIR and _GSNET_DIR not in sys.path:
    sys.path.insert(0, _GSNET_DIR)
ANYGRASP_CHECKPOINT = os.environ.get(
    'ANYGRASP_CHECKPOINT',
    os.path.join(_SCRIPT_DIR, 'log', 'checkpoint_detection.tar'),
)

# Camera intrinsics (front camera). Override via CAM_FX/CAM_FY/CAM_CX/CAM_CY / DEPTH_SCALE.
def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


# Defaults are empty placeholders; set CAM_* in configs/paths.env for your camera.
# Reference lab (Piper + RealSense) used approx fx=fy=488, cx=324, cy=211.
CAM_FX = _float_env("CAM_FX", 0.0)
CAM_FY = _float_env("CAM_FY", 0.0)
CAM_CX = _float_env("CAM_CX", 0.0)
CAM_CY = _float_env("CAM_CY", 0.0)
DEPTH_SCALE = _float_env("DEPTH_SCALE", 1000.0)


def _load_extrinsics_from_env() -> None:
    """Optionally override cam2base / T_tcp_flange from CAMERA_EXTRINSICS_JSON."""
    global T_TCP_FLANGE, _ARM_CONFIGS
    path = os.environ.get("CAMERA_EXTRINSICS_JSON", "").strip()
    if not path:
        return
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "T_tcp_flange" in data:
        T_TCP_FLANGE = np.asarray(data["T_tcp_flange"], dtype=np.float64)
    for arm in ("left", "right"):
        if arm not in data:
            continue
        block = data[arm]
        R = np.asarray(block["R"], dtype=np.float64)
        t = np.asarray(block["t"], dtype=np.float64).reshape(3)
        _ARM_CONFIGS[arm] = {
            "label": block.get("label", f"{arm}_arm_base"),
            "cam2base_samples": ((R, t),),
        }


_load_extrinsics_from_env()

# Workspace limits for grasp filtering [xmin, xmax, ymin, ymax, zmin, zmax]
DEFAULT_WORKSPACE_LIMS = [-0.4, 0.4, -0.4, 0.4, 0.41, 1.03]

# Approach-axis filter: AnyGrasp TCP +X == Piper flange +Z in base; pregrasp along that axis.
# Prefer oblique rays from an elevated reference point (default z=0.5 at arm base xy)
# toward each grasp — not straight down (IK-unfriendly), not near-horizontal (side reach).
DEFAULT_APPROACH_REF_Z = 2
DEFAULT_APPROACH_REF_XY = (0.0, 0.0)   # active arm base origin in its own base frame
DEFAULT_APPROACH_IDEAL_TILT_DEG = 35.0   # preferred depression below horizontal
DEFAULT_APPROACH_TILT_SIGMA_DEG = 18.0
DEFAULT_APPROACH_MIN_ALIGN = 0.25        # soft gate: dot(approach, desired)
DEFAULT_APPROACH_SCORE_WEIGHT = 0.65

# Piper flange RPY (rad) for forward reach in arm base: x fwd, y left, z up.
# roll≈-π & yaw≈-π: gripper faces forward; pitch≈1.57: approach horizontal (not straight down).
DEFAULT_FORWARD_FLANGE_RPY = np.array([-np.pi, 1.5, -np.pi], dtype=np.float64)
# Min |flange roll - ref_roll| (rad) when camera axis is horizontal (tie-break only).
DEFAULT_ROLL_CANON_THRESHOLD_RAD = np.pi / 2.0
# Arm base +Z; camera is on TCP +Z (= flange -X). Prefer this half-space for upright grasps.
_BASE_UP_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# 180° about TCP +X (approach): parallel-jaw equivalent; flips TCP ±Y/±Z (camera up/down).
_R_FLIP_TCP_APPROACH = np.diag([1.0, -1.0, -1.0])

# GraspNet viz (plot_gripper_pro_max): fingers reach up to +depth along TCP +X from translation.
# Execution TCP is shifted by the same amount so flange pose matches the mesh fingertip region.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
    parser.add_argument(
        '--top_down_grasp',
        action='store_true',
        help='Use a slightly steeper oblique target (ideal tilt 45 deg instead of 35).',
    )
    parser.add_argument(
        '--no_approach_filter',
        action='store_true',
        help='Disable approach-orientation filter; pick highest AnyGrasp score only.',
    )
    parser.add_argument(
        '--no_grasp_depth_offset',
        action='store_true',
        help=(
            'Do not shift execution TCP along approach by g.depth; '
            'default applies GraspNet depth so robot pose matches viz fingertips.'
        ),
    )
    parser.add_argument(
        '--approach_ref_z',
        type=float,
        default=DEFAULT_APPROACH_REF_Z,
        help='Reference retreat height (m) above base xy; oblique ray starts near (0,0,ref_z).',
    )
    parser.add_argument(
        '--approach_ideal_tilt_deg',
        type=float,
        default=None,
        help=(
            'Preferred approach depression below horizontal (deg). '
            f'Default {DEFAULT_APPROACH_IDEAL_TILT_DEG}; with --top_down_grasp uses 45.'
        ),
    )
    parser.add_argument(
        '--approach_score_weight',
        type=float,
        default=DEFAULT_APPROACH_SCORE_WEIGHT,
        help='Blend weight for AnyGrasp score vs approach orientation (0–1).',
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress intermediate logs and third-party warnings; print a short summary only',
    )
    parser.add_argument(
        '--gui_viz',
        action='store_true',
        help='Open interactive GUI viewer with layer toggles (Open3D gui framework)',
    )
    parser.add_argument('--arm', choices=('left', 'right'), default='left', help='Which arm to grasp with (selects hand-eye calibration)')
    parser.add_argument('--data_dir', type=str, default='./example_data', help='Directory containing color.png and depth.png')
    parser.add_argument(
        '--no_roll_canonicalize',
        action='store_true',
        help='Disable 180° TCP flip that keeps the wrist camera (TCP +Z) pointing upward.',
    )
    parser.add_argument(
        '--forward_rpy',
        type=float,
        nargs=3,
        metavar=('roll', 'pitch', 'yaw'),
        default=None,
        help=(
            'Reference forward flange RPY in rad for roll canonicalization. '
            f'Default {DEFAULT_FORWARD_FLANGE_RPY.tolist()}.'
        ),
    )
    parser.add_argument(
        '--roll_canon_threshold_deg',
        type=float,
        default=float(np.rad2deg(DEFAULT_ROLL_CANON_THRESHOLD_RAD)),
        help=(
            'When TCP +Z is nearly horizontal after camera-up flip, only flip if '
            '|flange roll - forward_rpy[0]| exceeds this (deg) and flip reduces roll error.'
        ),
    )
    parser.add_argument(
        '--forward_roll',
        type=float,
        default=None,
        help=(
            'Reference flange roll (rad) for canonicalization; overrides forward_rpy[0]. '
            f'Default {DEFAULT_FORWARD_FLANGE_RPY[0]:.4f}.'
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    cfgs = build_parser().parse_args(argv)
    cfgs.checkpoint_path = ANYGRASP_CHECKPOINT
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    return cfgs


def _average_rotation(Rs: Sequence[np.ndarray]) -> np.ndarray:
    """Average proper rotation matrices via SVD orthogonalization."""
    R_mean = np.mean(np.stack(Rs, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _build_T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def build_T_base_cam_from_cam2base_samples(
    samples: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Fuse repeated cam2base (cam2gripper) calibrations into one base_T_cam."""
    Rs = [R for R, _ in samples]
    ts = [t for _, t in samples]
    return _build_T_from_Rt(_average_rotation(Rs), np.mean(np.stack(ts, axis=0), axis=0))


def get_T_base_cam(arm: str) -> np.ndarray:
    if arm not in _ARM_CONFIGS:
        raise ValueError(f'Unknown arm {arm!r}, expected one of {list(_ARM_CONFIGS)}')
    return build_T_base_cam_from_cam2base_samples(_ARM_CONFIGS[arm]['cam2base_samples'])


def load_rgb_depth(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    return colors, depths


def build_point_cloud_from_rgbd(
    colors: np.ndarray,
    depths: np.ndarray,
    *,
    fx: float = CAM_FX,
    fy: float = CAM_FY,
    cx: float = CAM_CX,
    cy: float = CAM_CY,
    scale: float = DEPTH_SCALE,
    pixel_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project RGB-D to colored points in camera frame.

    pixel_mask: optional (H, W) bool; True keeps the pixel.
    """
    h, w = depths.shape[:2]
    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        if pixel_mask.shape != (h, w):
            raise ValueError(f'pixel_mask shape {pixel_mask.shape} != depth shape {(h, w)}')

    xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
    points_z = depths.astype(np.float64) / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    valid = (points_z > 0) & (points_z < 1)
    if pixel_mask is not None:
        valid &= pixel_mask

    points = np.stack([points_x, points_y, points_z], axis=-1)[valid].astype(np.float32)
    point_colors = colors[valid].astype(np.float32)
    return points, point_colors


def _R_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def project_cam_point_to_pixel(
    point_cam: np.ndarray,
    *,
    fx: float = CAM_FX,
    fy: float = CAM_FY,
    cx: float = CAM_CX,
    cy: float = CAM_CY,
) -> Optional[Tuple[int, int]]:
    x, y, z = np.asarray(point_cam, dtype=np.float64).reshape(3)
    if z <= 1e-6:
        return None
    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))
    return u, v


def filter_grasps_by_pixel_mask(
    gg: GraspGroup,
    pixel_mask: np.ndarray,
    *,
    fx: float = CAM_FX,
    fy: float = CAM_FY,
    cx: float = CAM_CX,
    cy: float = CAM_CY,
) -> GraspGroup:
    """Keep grasps whose center projects into pixel_mask foreground."""
    h, w = pixel_mask.shape[:2]
    keep = []
    for i in range(len(gg)):
        px = project_cam_point_to_pixel(gg[i].translation, fx=fx, fy=fy, cx=cx, cy=cy)
        if px is None:
            continue
        u, v = px
        if 0 <= u < w and 0 <= v < h and pixel_mask[v, u]:
            keep.append(i)
    _log(f'Mask grasp filter: kept {len(keep)}/{len(gg)} grasps')
    if not keep:
        return GraspGroup()
    return gg[keep]


def _grasp_approach_axis_base(g, T_base_cam: np.ndarray) -> np.ndarray:
    """AnyGrasp approach (TCP +X) in the active arm's base frame."""
    R_base_tcp = T_base_cam[:3, :3] @ g.rotation_matrix
    v = R_base_tcp[:, 0]
    return v / (np.linalg.norm(v) + 1e-12)


def _grasp_position_base(g, T_base_cam: np.ndarray) -> np.ndarray:
    p_cam = np.asarray(g.translation, dtype=np.float64).reshape(3)
    return T_base_cam[:3, :3] @ p_cam + T_base_cam[:3, 3]


def _grasp_tcp_translation_cam(g, *, apply_grasp_depth: bool = True) -> np.ndarray:
    """AnyGrasp TCP origin in camera frame, optionally shifted by g.depth.

    GraspNet defines translation as the gripper frame origin. In plot_gripper_pro_max
    the finger bodies extend up to ``depth`` meters along TCP +X (approach). Without
    the shift, execution stops at the frame origin while the viz mesh reaches farther
    into the scene — use apply_grasp_depth=True (default) to align them.
    """
    p = np.asarray(g.translation, dtype=np.float64).reshape(3)
    if not apply_grasp_depth:
        return p
    depth_m = float(g.depth)
    if depth_m <= 0.0:
        return p
    approach = np.asarray(g.rotation_matrix[:, 0], dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(approach))
    if n < 1e-12:
        return p
    return p + (approach / n) * depth_m


def _approach_depression_deg(approach: np.ndarray) -> float:
    """Angle below the base xy plane (deg). Positive = pointing downward."""
    horiz = float(np.hypot(approach[0], approach[1]))
    return float(np.degrees(np.arctan2(max(0.0, -approach[2]), horiz + 1e-12)))


def _tilt_preference_score(depression_deg: float, *, ideal_deg: float, sigma_deg: float) -> float:
    """Peak at ideal oblique tilt; penalize near-vertical and near-horizontal."""
    if depression_deg <= 0.0:
        return 0.0
    delta = (depression_deg - ideal_deg) / max(sigma_deg, 1e-6)
    return float(np.exp(-0.5 * delta * delta))


def _desired_approach_axis_base(
    grasp_pos: np.ndarray,
    *,
    ref_xy: Tuple[float, float] = DEFAULT_APPROACH_REF_XY,
    ref_z: float = DEFAULT_APPROACH_REF_Z,
) -> Optional[np.ndarray]:
    """Unit vector from elevated reference point toward grasp (TCP +X target)."""
    ref = np.array([ref_xy[0], ref_xy[1], ref_z], dtype=np.float64)
    delta = np.asarray(grasp_pos, dtype=np.float64).reshape(3) - ref
    n = np.linalg.norm(delta)
    if n < 1e-6:
        return None
    return delta / n


def score_grasp_approach_orientation(
    g,
    T_base_cam: np.ndarray,
    *,
    ref_xy: Tuple[float, float] = DEFAULT_APPROACH_REF_XY,
    ref_z: float = DEFAULT_APPROACH_REF_Z,
    ideal_tilt_deg: float = DEFAULT_APPROACH_IDEAL_TILT_DEG,
    tilt_sigma_deg: float = DEFAULT_APPROACH_TILT_SIGMA_DEG,
) -> Tuple[float, float, float, np.ndarray, Optional[np.ndarray]]:
    """Score oblique approach: elevated base-side ref point -> grasp contact.

    Returns (orientation_score, align, depression_deg, approach_axis, desired_axis).
    """
    approach = _grasp_approach_axis_base(g, T_base_cam)
    grasp_pos = _grasp_position_base(g, T_base_cam)
    desired = _desired_approach_axis_base(grasp_pos, ref_xy=ref_xy, ref_z=ref_z)

    if desired is None:
        return 0.0, 0.0, 0.0, approach, None

    align = float(max(0.0, min(1.0, np.dot(approach, desired))))
    depression = _approach_depression_deg(approach)
    tilt = _tilt_preference_score(depression, ideal_deg=ideal_tilt_deg, sigma_deg=tilt_sigma_deg)
    orient = 0.65 * align + 0.35 * tilt
    return orient, align, depression, approach, desired


def filter_grasps_by_approach_orientation(
    gg: GraspGroup,
    T_base_cam: np.ndarray,
    *,
    ref_xy: Tuple[float, float] = DEFAULT_APPROACH_REF_XY,
    ref_z: float = DEFAULT_APPROACH_REF_Z,
    ideal_tilt_deg: float = DEFAULT_APPROACH_IDEAL_TILT_DEG,
    tilt_sigma_deg: float = DEFAULT_APPROACH_TILT_SIGMA_DEG,
    min_align: float = DEFAULT_APPROACH_MIN_ALIGN,
    score_weight: float = DEFAULT_APPROACH_SCORE_WEIGHT,
    fallback_if_empty: bool = True,
) -> GraspGroup:
    """Re-rank grasps by AnyGrasp score + oblique approach from ref_z toward contact."""
    if len(gg) == 0:
        return gg

    score_weight = float(max(0.0, min(1.0, score_weight)))
    min_align = float(max(0.0, min(1.0, min_align)))

    scored = []
    for i in range(len(gg)):
        g = gg[i]
        orient, align, depression, approach, desired = score_grasp_approach_orientation(
            g,
            T_base_cam,
            ref_xy=ref_xy,
            ref_z=ref_z,
            ideal_tilt_deg=ideal_tilt_deg,
            tilt_sigma_deg=tilt_sigma_deg,
        )
        combined = score_weight * float(g.score) + (1.0 - score_weight) * orient
        scored.append((i, combined, orient, align, depression, approach, desired))

    passed = [item for item in scored if item[3] >= min_align]
    pool = passed if passed else (scored if fallback_if_empty else [])
    if not pool:
        print('Approach filter: no grasps left')
        return GraspGroup()

    if not passed and fallback_if_empty:
        _log(
            f'Approach filter: 0/{len(gg)} pass min_align={min_align:.2f}; '
            'falling back to orientation-ranked pool'
        )
    else:
        _log(
            f'Approach filter: kept {len(passed)}/{len(gg)} grasps '
            f'(ref=({ref_xy[0]:.2f},{ref_xy[1]:.2f},{ref_z:.2f}), '
            f'ideal_tilt={ideal_tilt_deg:.0f} deg, min_align={min_align:.2f})'
        )

    pool.sort(key=lambda item: item[1], reverse=True)
    best_i, best_combined, best_orient, best_align, best_dep, best_approach, best_desired = pool[0]
    best_g = gg[best_i]
    _log(
        f'  Best after approach filter: score={best_g.score:.3f} orient={best_orient:.3f} '
        f'align={best_align:.3f} tilt={best_dep:.1f} deg '
        f'approach={best_approach} desired={best_desired}'
    )

    ranked_indices = [item[0] for item in pool]
    return gg[ranked_indices]


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _roll_error_rad(roll: float, ref_roll: float) -> float:
    return abs(_wrap_to_pi(roll - ref_roll))


def _flange_R_from_grasp(
    g,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
) -> np.ndarray:
    _, T_base_flange, _ = _grasp_cam_to_base_flange(g, T_base_cam, T_tcp_flange)
    return T_base_flange[:3, :3]


def _flange_rpy_from_grasp(
    g,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
) -> np.ndarray:
    return _R_to_rpy_rad(_flange_R_from_grasp(g, T_base_cam, T_tcp_flange))


def _tcp_camera_axis_base(
    g,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
) -> np.ndarray:
    """Wrist-camera axis in base frame: AnyGrasp TCP +Z (= flange -X)."""
    T_base_tcp, _, _ = _grasp_cam_to_base_flange(g, T_base_cam, T_tcp_flange)
    axis = T_base_tcp[:3, 2]
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return _BASE_UP_AXIS.copy()
    return axis / n


def _flip_grasp_about_tcp_approach(g) -> None:
    """In-place: 180° about grasp TCP +X (parallel-jaw symmetric flip)."""
    g.rotation_matrix = g.rotation_matrix @ _R_FLIP_TCP_APPROACH


def canonicalize_grasp_forward_roll(
    g,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
    *,
    ref_roll: float,
    roll_threshold_rad: float = DEFAULT_ROLL_CANON_THRESHOLD_RAD,
) -> bool:
    """Pick the parallel-jaw branch with wrist camera (TCP +Z) pointing more upward.

    Always compares the grasp with its 180°-about-approach symmetric mate and keeps
    the branch with larger dot(camera, base_up). ``ref_roll`` / ``roll_threshold_rad``
    are kept for API compatibility but no longer drive the decision.
    """
    del ref_roll, roll_threshold_rad
    up0 = float(np.dot(_tcp_camera_axis_base(g, T_base_cam, T_tcp_flange), _BASE_UP_AXIS))
    if up0 >= 1.0 - 1e-9:
        return False

    R_before = g.rotation_matrix.copy()
    _flip_grasp_about_tcp_approach(g)
    up1 = float(np.dot(_tcp_camera_axis_base(g, T_base_cam, T_tcp_flange), _BASE_UP_AXIS))
    if up1 > up0 + 1e-9:
        return True

    g.rotation_matrix = R_before
    return False


def canonicalize_grasps_forward_roll(
    gg: GraspGroup,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
    *,
    ref_roll: float,
    roll_threshold_rad: float = DEFAULT_ROLL_CANON_THRESHOLD_RAD,
) -> GraspGroup:
    if len(gg) == 0:
        return gg
    flipped = 0
    for i in range(len(gg)):
        g = gg[i]
        if canonicalize_grasp_forward_roll(
            g,
            T_base_cam,
            T_tcp_flange,
            ref_roll=ref_roll,
            roll_threshold_rad=roll_threshold_rad,
        ):
            gg.grasp_group_array[i] = g.grasp_array
            flipped += 1
    _log(
        f'Camera-up canonicalize: flipped {flipped}/{len(gg)} grasps '
        f'(TCP+Z toward base+Z; roll tie-break ref={ref_roll:.3f} rad, '
        f'threshold={np.rad2deg(roll_threshold_rad):.0f} deg)'
    )
    return gg


def _R_to_rpy_rad(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _rpy_rad_to_R(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in np.asarray(rpy, dtype=np.float64).reshape(3)]
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _xyzrpy_to_T(xyzrpy: np.ndarray) -> np.ndarray:
    v = np.asarray(xyzrpy, dtype=np.float64).reshape(6)
    return _build_T_from_Rt(_rpy_rad_to_R(v[3:]), v[:3])


def _load_eef_pose_npy(path: str) -> Tuple[np.ndarray, float]:
    """Load (7,) [x,y,z,roll,pitch,yaw, grasp_width] from run_grasp output."""
    v = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if v.shape != (7,):
        raise ValueError(f'{path}: expected shape (7,) [x,y,z,roll,pitch,yaw,grasp_width], got {v.shape}')
    return v[:6], float(v[6])


def _as_T(name: str, T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f'{name} must be 4x4, got {T.shape}')
    return T


def _save_vec_npy(path: str, v: np.ndarray) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    np.save(path, v.astype(np.float64))


def _coord_frame_at(T: np.ndarray, size: float = 0.08) -> o3d.geometry.TriangleMesh:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size))
    frame.transform(_as_T('T', T))
    return frame


def _T_other_arm_base_in_active_base(active_arm: str, other_arm: str) -> np.ndarray:
    """Pose of other_arm's base origin/axes expressed in active_arm's base frame."""
    if active_arm == other_arm:
        return np.eye(4, dtype=np.float64)
    T_active_cam = get_T_base_cam(active_arm)
    T_other_cam = get_T_base_cam(other_arm)
    return T_active_cam @ np.linalg.inv(T_other_cam)


def _both_arm_base_frames_in_active_base(
    active_arm: str,
    *,
    active_size: float = 0.15,
    other_size: float = 0.12,
) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    """Return (left_base_frame, right_base_frame) triads in active_arm's base frame."""
    left_T = _T_other_arm_base_in_active_base(active_arm, 'left')
    right_T = _T_other_arm_base_in_active_base(active_arm, 'right')
    left_size = active_size if active_arm == 'left' else other_size
    right_size = active_size if active_arm == 'right' else other_size
    left_frame = _coord_frame_at(left_T, size=left_size)
    right_frame = _coord_frame_at(right_T, size=right_size)
    if active_arm == 'left':
        left_frame.paint_uniform_color([0.2, 0.55, 1.0])
    else:
        left_frame.paint_uniform_color([0.35, 0.65, 0.95])
    if active_arm == 'right':
        right_frame.paint_uniform_color([1.0, 0.35, 0.2])
    else:
        right_frame.paint_uniform_color([0.95, 0.55, 0.35])
    return left_frame, right_frame


def _base_xy_plane_at(
    T: np.ndarray,
    *,
    size: float = 1.2,
    color: Sequence[float] = (0.95, 0.85, 0.2),
) -> o3d.geometry.TriangleMesh:
    """Thin sheet in local xy (z=0), posed by T in the parent frame."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=float(size), height=float(size), depth=0.002)
    mesh.translate([-size / 2.0, -size / 2.0, -0.001])
    mesh.transform(_as_T('T', T))
    mesh.paint_uniform_color(np.asarray(color, dtype=np.float64))
    return mesh


def _base_xy_reference_plane(size: float = 1.2, z: float = 0.0) -> o3d.geometry.TriangleMesh:
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = float(z)
    return _base_xy_plane_at(T, size=size)


def diagnose_table_vs_base_z(points_base: np.ndarray) -> Tuple[float, np.ndarray]:
    pts = np.asarray(points_base, dtype=np.float64)
    if pts.shape[0] < 50:
        return float('nan'), np.array([0.0, 0.0, 1.0])
    z = pts[:, 2]
    z_cut = np.percentile(z, 25) + 0.03
    table_pts = pts[z <= z_cut]
    if table_pts.shape[0] < 50:
        table_pts = pts
    center = table_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(table_pts - center, full_matrices=False)
    normal = vh[2]
    if normal[2] < 0.0:
        normal = -normal
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    tilt_deg = float(np.rad2deg(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    return tilt_deg, normal


def _sphere_at(center: np.ndarray, radius: float = 0.012, color=(1.0, 0.2, 0.2)) -> o3d.geometry.TriangleMesh:
    s = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
    s.translate(np.asarray(center, dtype=np.float64).reshape(3))
    s.paint_uniform_color(np.asarray(color, dtype=np.float64))
    return s


def _approach_arrow_at(
    T: np.ndarray,
    *,
    length: float = 0.12,
    color: Sequence[float] = (0.95, 0.15, 0.55),
) -> o3d.geometry.TriangleMesh:
    """Arrow along frame +Z (Piper flange approach when T is T_base_flange)."""
    length = float(length)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003,
        cone_radius=0.007,
        cylinder_height=length * 0.72,
        cone_height=length * 0.28,
    )
    arrow.transform(_as_T('T', T))
    arrow.paint_uniform_color(np.asarray(color, dtype=np.float64))
    arrow.compute_vertex_normals()
    return arrow


def _best_grasp_orientation_geoms(
    g,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
    *,
    apply_grasp_depth: bool = True,
) -> Tuple[
    o3d.geometry.TriangleMesh,
    o3d.geometry.TriangleMesh,
    o3d.geometry.TriangleMesh,
    o3d.geometry.TriangleMesh,
]:
    """TCP triad, approach arrow, flange triad, and contact sphere for the best grasp."""
    T_base_tcp, T_base_flange, _ = _grasp_cam_to_base_flange(
        g,
        T_base_cam,
        T_tcp_flange,
        apply_grasp_depth=apply_grasp_depth,
    )
    tcp_frame = _coord_frame_at(T_base_tcp, size=0.06)
    approach_arrow = _approach_arrow_at(T_base_flange)
    flange_frame = _coord_frame_at(T_base_flange, size=0.10)
    contact = _sphere_at(T_base_tcp[:3, 3], radius=0.008, color=(1.0, 0.85, 0.1))
    return tcp_frame, approach_arrow, flange_frame, contact


def _grasp_cam_to_base_flange(
    g,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
    *,
    apply_grasp_depth: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_cam_tcp = np.eye(4, dtype=np.float64)
    T_cam_tcp[:3, :3] = g.rotation_matrix
    T_cam_tcp[:3, 3] = _grasp_tcp_translation_cam(g, apply_grasp_depth=apply_grasp_depth)
    T_base_tcp = T_base_cam @ T_cam_tcp
    T_base_flange = T_base_tcp @ T_tcp_flange
    approach_axis_b = T_base_tcp[:3, :3][:, 0]
    return T_base_tcp, T_base_flange, approach_axis_b


_CAM_DISPLAY_FLIP = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _make_o3d_point_cloud(points: np.ndarray, point_colors: np.ndarray) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(point_colors, dtype=np.float64))
    return pc


def _visualize_grasps_camera_frame(
    gg_show: GraspGroup,
    cloud: o3d.geometry.PointCloud,
    window_title: str,
) -> None:
    """Show grasps with full-scene point cloud in camera frame (Z flipped for display)."""
    if len(gg_show) == 0:
        print(f'  Debug skip "{window_title}": no grasps')
        return
    cloud_cam = o3d.geometry.PointCloud(cloud)
    cloud_cam.transform(_CAM_DISPLAY_FLIP)
    grippers = gg_show.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(_CAM_DISPLAY_FLIP)
    print(f'  Debug: {window_title} ({len(grippers)} grasps, {len(cloud_cam.points)} points)')
    o3d.visualization.draw_geometries([*grippers, cloud_cam], window_name=window_title)


def _grasp_to_xyzrpy(
    g,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
    *,
    apply_grasp_depth: bool = True,
) -> np.ndarray:
    _, T_base_flange, _ = _grasp_cam_to_base_flange(
        g,
        T_base_cam,
        T_tcp_flange,
        apply_grasp_depth=apply_grasp_depth,
    )
    rpy_flange = _R_to_rpy_rad(T_base_flange[:3, :3])
    return np.concatenate([T_base_flange[:3, 3], rpy_flange])


def visualize_in_base_frame(
    cloud_cam: o3d.geometry.PointCloud,
    gg_pick: GraspGroup,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
    *,
    arm: str = 'left',
    max_grasps: int = 5,
    show_saved_npy: bool = True,
    window_title: Optional[str] = None,
    apply_grasp_depth: bool = True,
) -> None:
    T_base_cam = _as_T('T_base_cam', T_base_cam)
    T_tcp_flange = _as_T('T_tcp_flange', T_tcp_flange)
    if window_title is None:
        window_title = f'Base frame ({arm}): point cloud + flange/TCP'

    cloud_base = o3d.geometry.PointCloud(cloud_cam)
    cloud_base.transform(T_base_cam)
    pts_base = np.asarray(cloud_base.points)
    table_tilt_deg, table_normal = diagnose_table_vs_base_z(pts_base)

    left_base_frame, right_base_frame = _both_arm_base_frames_in_active_base(arm)
    geoms = [cloud_base, _base_xy_reference_plane(), left_base_frame, right_base_frame]
    print('\n=== Base-frame extrinsic sanity check ===')
    print(f'  Table plane tilt vs base +Z: {table_tilt_deg:.1f} deg  (expect <~5 deg if T_BASE_CAM is good)')
    print(f'  Fitted table normal (base): {table_normal}')
    if table_tilt_deg > 8.0:
        print('  => Large tilt: T_BASE_CAM likely wrong (cam2gripper used as base, or bad hand-eye).')
    else:
        print('  => Tilt OK: global base frame vs scene is plausible; check Piper rpy / T_TCP_FLANGE if arm misses.')

    n_show = min(int(max_grasps), len(gg_pick))
    for i in range(n_show):
        g = gg_pick[i]
        T_base_tcp, T_base_flange, approach_b = _grasp_cam_to_base_flange(
            g,
            T_base_cam,
            T_tcp_flange,
            apply_grasp_depth=apply_grasp_depth,
        )
        tb_flange = T_base_flange[:3, 3]
        tb_pre = tb_flange - approach_b * float(PREGRASP_OFFSET_M)

        if i == 0:
            tcp_gripper = g.to_open3d_geometry()
            tcp_gripper.transform(T_base_cam)
            geoms.append(tcp_gripper)
            tcp_frame, approach_arrow, flange_frame, contact = _best_grasp_orientation_geoms(
                g,
                T_base_cam,
                T_tcp_flange,
                apply_grasp_depth=apply_grasp_depth,
            )
            geoms.extend([tcp_frame, approach_arrow, flange_frame, contact])
            geoms.append(_sphere_at(tb_pre, radius=0.015, color=(1.0, 0.15, 0.15)))
            print('\n=== Base-frame visualization (best grasp) ===')
            print(f'  grasp_depth(m)= {float(g.depth):.4f}  apply_grasp_depth={apply_grasp_depth}')
            print('  T_base_flange.t =', tb_flange)
            print('  pregrasp_flange.t =', tb_pre)
        else:
            frame = _coord_frame_at(T_base_flange, size=0.05)
            frame.paint_uniform_color([0.55, 0.55, 0.55])
            geoms.append(frame)

    if show_saved_npy and os.path.isfile(EEF_POSE_XYZRPY_NPY):
        saved_xyzrpy, _ = _load_eef_pose_npy(EEF_POSE_XYZRPY_NPY)
        T_saved = _as_T('eef_pose_xyzrpy.npy', _xyzrpy_to_T(saved_xyzrpy))
        saved_frame = _coord_frame_at(T_saved, size=0.08)
        geoms.append(saved_frame)
        if n_show > 0:
            _, T_best_flange, _ = _grasp_cam_to_base_flange(
                gg_pick[0],
                T_base_cam,
                T_tcp_flange,
                apply_grasp_depth=apply_grasp_depth,
            )
            pos_err_mm = 1e3 * np.linalg.norm(T_saved[:3, 3] - T_best_flange[:3, 3])
            ang_err_deg = np.rad2deg(
                np.arccos(np.clip((np.trace(T_saved[:3, :3].T @ T_best_flange[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
            )
            print(f'  Saved npy vs live best flange: |dt|={pos_err_mm:.1f} mm, angle={ang_err_deg:.1f} deg')

    print(
        f'  Legend: yellow sheet=base z=0 (table level); '
        f'blue triad=left base; orange triad=right base (active arm={arm}, larger triad); '
        f'green mesh=AnyGrasp TCP gripper; RGB@contact=AnyGrasp TCP (red X=API approach, green Y=jaw open); '
        f'magenta arrow=flange +Z approach (same as Piper); larger RGB triad=flange; gold=contact; red=pregrasp.'
    )
    print('  Close window to continue.')
    o3d.visualization.draw_geometries(geoms, window_name=window_title)


# =============================================================================
# GUI visualization (Open3D gui) — isolated from draw_geometries paths above
# =============================================================================

_GUI_LAYER_POINT_CLOUD = 'layer_point_cloud'
_GUI_LAYER_BASE_FRAME_LEFT = 'layer_base_frame_left'
_GUI_LAYER_BASE_FRAME_RIGHT = 'layer_base_frame_right'
_GUI_LAYER_BASE_XY_LEFT = 'layer_base_xy_left'
_GUI_LAYER_BASE_XY_RIGHT = 'layer_base_xy_right'
_GUI_LAYER_GRASPS_NMS = 'layer_grasps_nms'
_GUI_LAYER_GRASPS_NMS_MASK = 'layer_grasps_nms_mask'
_GUI_LAYER_GRASP_BEST = 'layer_grasp_best'
_GUI_LAYER_GRASP_BEST_TCP_FRAME = 'layer_grasp_best_tcp_frame'
_GUI_LAYER_GRASP_BEST_APPROACH = 'layer_grasp_best_approach'
_GUI_LAYER_GRASP_BEST_FLANGE = 'layer_grasp_best_flange'
_GUI_LAYER_GRASP_BEST_CONTACT = 'layer_grasp_best_contact'


def _setup_gui_cjk_font() -> None:
    """Prefer a CJK-capable font for checkbox labels on Linux."""
    try:
        font = gui.FontDescription()
        font.add_typeface_for_language('NotoSansCJK', 'zh')
        gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)
    except Exception:
        pass


def _combine_triangle_meshes(meshes: Sequence[o3d.geometry.TriangleMesh]) -> Optional[o3d.geometry.TriangleMesh]:
    if not meshes:
        return None
    combined = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        combined += mesh
    if len(combined.vertices) == 0:
        return None
    combined.compute_vertex_normals()
    return combined


def _grasp_group_to_base_mesh(
    gg: GraspGroup,
    T_base_cam: np.ndarray,
    *,
    uniform_color: Optional[Sequence[float]] = None,
) -> Optional[o3d.geometry.TriangleMesh]:
    if len(gg) == 0:
        return None
    T_base_cam = _as_T('T_base_cam', T_base_cam)
    grippers = []
    for gripper in gg.to_open3d_geometry_list():
        mesh = o3d.geometry.TriangleMesh(gripper)
        mesh.transform(T_base_cam)
        if uniform_color is not None:
            mesh.paint_uniform_color(np.asarray(uniform_color, dtype=np.float64))
        grippers.append(mesh)
    return _combine_triangle_meshes(grippers)


class _GraspGuiVisualizer:
    """Checkbox-driven base-frame viewer; independent from legacy draw_geometries."""

    _PANEL_WIDTH_EM = 22.0

    def __init__(
        self,
        layers: Sequence[Tuple[str, str, Optional[o3d.geometry.Geometry], bool]],
        *,
        window_title: str,
    ) -> None:
        self._layers = {key: (label, geom, default_visible) for key, label, geom, default_visible in layers}
        self._app = gui.Application.instance
        self._app.initialize()
        _setup_gui_cjk_font()

        self._window = self._app.create_window(window_title, 1400, 900)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self._window.renderer)

        self._pc_mat = rendering.MaterialRecord()
        self._pc_mat.shader = 'defaultUnlit'
        self._pc_mat.point_size = 2.0

        self._mesh_mat = rendering.MaterialRecord()
        self._mesh_mat.shader = 'defaultLit'

        self._panel = gui.Vert(0.5 * self._window.theme.font_size, gui.Margins(0.5 * self._window.theme.font_size))
        self._panel.add_child(gui.Label('Display layers'))
        self._panel.add_fixed(int(0.5 * self._window.theme.font_size))

        for key, (label, geom, default_visible) in self._layers.items():
            if geom is None:
                cb = gui.Checkbox(label)
                cb.enabled = False
                cb.tooltip = 'No data available'
            else:
                mat = self._pc_mat if isinstance(geom, o3d.geometry.PointCloud) else self._mesh_mat
                self._scene.scene.add_geometry(key, geom, mat)
                self._scene.scene.show_geometry(key, default_visible)
                cb = gui.Checkbox(label)
                cb.checked = default_visible
                cb.set_on_checked(lambda checked, layer_key=key: self._on_layer_toggle(layer_key, checked))
            self._panel.add_child(cb)

        self._window.set_on_layout(self._on_layout)
        self._window.add_child(self._scene)
        self._window.add_child(self._panel)

        bounds = self._scene.scene.bounding_box
        if bounds.get_extent()[0] > 0:
            self._scene.setup_camera(60.0, bounds, bounds.get_center())

        print('  GUI viewer: use checkboxes on the left to toggle layers. Close window to continue.')
        self._app.run()

    def _on_layer_toggle(self, layer_key: str, visible: bool) -> None:
        if layer_key in self._layers and self._layers[layer_key][1] is not None:
            self._scene.scene.show_geometry(layer_key, visible)

    def _on_layout(self, layout_context: gui.LayoutContext) -> None:
        content = self._window.content_rect
        em = layout_context.theme.font_size
        panel_w = int(self._PANEL_WIDTH_EM * em)
        self._panel.frame = gui.Rect(content.x, content.y, panel_w, content.height)
        self._scene.frame = gui.Rect(content.x + panel_w, content.y, content.width - panel_w, content.height)


def visualize_grasps_gui(
    cloud_cam: o3d.geometry.PointCloud,
    gg_all: GraspGroup,
    gg_masked: Optional[GraspGroup],
    gg_best: GraspGroup,
    T_base_cam: np.ndarray,
    T_tcp_flange: np.ndarray,
    *,
    arm: str = 'left',
    show_saved_triad: bool = True,
    window_title: Optional[str] = None,
    apply_grasp_depth: bool = True,
) -> None:
    """Interactive base-frame viewer with checkbox layers (Open3D gui framework)."""
    if cloud_cam is None:
        raise ValueError('cloud_cam is None; pass a valid PointCloud (e.g. from _make_o3d_point_cloud).')
    T_base_cam = _as_T('T_base_cam', T_base_cam)
    T_tcp_flange = _as_T('T_tcp_flange', T_tcp_flange)
    if window_title is None:
        window_title = f'Grasp GUI ({arm}) — base frame'

    cloud_base = o3d.geometry.PointCloud(cloud_cam)
    cloud_base.transform(T_base_cam)

    flange_triad = None
    tcp_frame = None
    approach_arrow = None
    contact_sphere = None
    if len(gg_best) > 0:
        tcp_frame, approach_arrow, flange_triad, contact_sphere = _best_grasp_orientation_geoms(
            gg_best[0],
            T_base_cam,
            T_tcp_flange,
            apply_grasp_depth=apply_grasp_depth,
        )
    elif show_saved_triad and os.path.isfile(EEF_POSE_XYZRPY_NPY):
        saved_xyzrpy, _ = _load_eef_pose_npy(EEF_POSE_XYZRPY_NPY)
        T_saved = _as_T('eef_pose_xyzrpy.npy', _xyzrpy_to_T(saved_xyzrpy))
        flange_triad = _coord_frame_at(T_saved, size=0.10)

    mask_label = 'Grasps after NMS+Mask'
    if gg_masked is None:
        mask_label = 'Grasps after NMS+Mask (no mask)'

    left_base_frame, right_base_frame = _both_arm_base_frames_in_active_base(arm)
    left_T = _T_other_arm_base_in_active_base(arm, 'left')
    right_T = _T_other_arm_base_in_active_base(arm, 'right')
    left_xy_plane = _base_xy_plane_at(left_T, color=(0.2, 0.55, 1.0))
    right_xy_plane = _base_xy_plane_at(right_T, color=(1.0, 0.35, 0.2))
    layers = [
        (_GUI_LAYER_POINT_CLOUD, 'Point cloud', cloud_base, True),
        (_GUI_LAYER_BASE_FRAME_LEFT, 'Left Base frame', left_base_frame, True),
        (_GUI_LAYER_BASE_FRAME_RIGHT, 'Right Base frame', right_base_frame, True),
        (_GUI_LAYER_BASE_XY_LEFT, 'Left Base xy plane', left_xy_plane, False),
        (_GUI_LAYER_BASE_XY_RIGHT, 'Right Base xy plane', right_xy_plane, False),
        (_GUI_LAYER_GRASPS_NMS, 'All grasps after NMS', _grasp_group_to_base_mesh(gg_all, T_base_cam), True),
        (
            _GUI_LAYER_GRASPS_NMS_MASK,
            mask_label,
            _grasp_group_to_base_mesh(gg_masked, T_base_cam) if gg_masked is not None else None,
            gg_masked is not None and len(gg_masked) > 0,
        ),
        (
            _GUI_LAYER_GRASP_BEST,
            'Best grasp (green mesh)',
            _grasp_group_to_base_mesh(gg_best, T_base_cam, uniform_color=(0.1, 0.85, 0.2)),
            len(gg_best) > 0,
        ),
        (
            _GUI_LAYER_GRASP_BEST_TCP_FRAME,
            'AnyGrasp TCP (red X=API approach, green Y=jaw open)',
            tcp_frame,
            tcp_frame is not None,
        ),
        (
            _GUI_LAYER_GRASP_BEST_APPROACH,
            'Approach arrow (Flange blue Z / Piper)',
            approach_arrow,
            approach_arrow is not None,
        ),
        (
            _GUI_LAYER_GRASP_BEST_FLANGE,
            'Flange frame (blue Z=approach, eef_pose)',
            flange_triad,
            flange_triad is not None,
        ),
        (
            _GUI_LAYER_GRASP_BEST_CONTACT,
            'Contact point (TCP origin)',
            contact_sphere,
            contact_sphere is not None,
        ),
    ]
    _GraspGuiVisualizer(layers, window_title=window_title)


def run_grasp(
    data_dir: str,
    cfgs: argparse.Namespace,
    *,
    pixel_mask: Optional[np.ndarray] = None,
    apply_object_mask: Optional[bool] = None,
) -> None:
    """Run AnyGrasp on RGB-D in data_dir and save EEF poses for the selected arm.

    When pixel_mask is given, AnyGrasp still runs on the full depth-valid point
    cloud (collision detection keeps table/scene geometry), then grasps whose TCP
    center projects into the mask foreground are kept (grasp-first).
    """
    _configure_quiet_mode(cfgs.quiet, debug=cfgs.debug)
    arm = cfgs.arm
    arm_label = _ARM_CONFIGS[arm]['label']

    T_base_cam = _as_T('T_BASE_CAM', get_T_base_cam(arm))
    T_tcp_flange = _as_T('T_TCP_FLANGE', T_TCP_FLANGE)
    apply_grasp_depth = not cfgs.no_grasp_depth_offset
    _log(f'Arm: {arm} ({arm_label})')
    _log(f'Grasp depth offset along approach: {"on" if apply_grasp_depth else "off"}')

    colors, depths = load_rgb_depth(data_dir)
    lims = list(DEFAULT_WORKSPACE_LIMS)
    points, point_colors = build_point_cloud_from_rgbd(colors, depths)
    if points.shape[0] == 0:
        raise RuntimeError('No valid points after depth filtering.')

    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        h, w = depths.shape[:2]
        if pixel_mask.shape != (h, w):
            raise ValueError(f'pixel_mask shape {pixel_mask.shape} != depth shape {(h, w)}')
        n_fg = int(np.count_nonzero(pixel_mask))
        _log(
            f'Object mask: {n_fg}/{h * w} foreground pixels '
            f'({100.0 * n_fg / (h * w):.1f}%) — full point cloud for inference'
        )

    _log(points.min(axis=0), points.max(axis=0))

    if apply_object_mask is None:
        apply_object_mask = pixel_mask is None

    def _load_and_infer():
        _import_anygrasp_deps()
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
        return anygrasp.get_grasp(
            points,
            point_colors,
            lims=lims,
            apply_object_mask=apply_object_mask,
            dense_grasp=False,
            collision_detection=True,
        )

    if cfgs.quiet and not cfgs.debug:
        with _silence_stdio():
            gg, cloud = _load_and_infer()
    else:
        gg, cloud = _load_and_infer()

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg_all = gg.nms().sort_by_score()
    gg_masked = None
    if pixel_mask is not None:
        gg_object = filter_grasps_by_pixel_mask(gg, pixel_mask)
        if len(gg_object) == 0:
            print('No grasps remain after mask filter.')
        gg_masked = gg_object.nms().sort_by_score()
        gg = gg_masked
    else:
        gg = gg_all

    if not cfgs.no_roll_canonicalize:
        ref_rpy = (
            np.asarray(cfgs.forward_rpy, dtype=np.float64)
            if cfgs.forward_rpy is not None
            else DEFAULT_FORWARD_FLANGE_RPY
        )
        ref_roll = (
            float(cfgs.forward_roll)
            if cfgs.forward_roll is not None
            else float(ref_rpy[0])
        )
        roll_thresh = float(np.deg2rad(cfgs.roll_canon_threshold_deg))
        gg_all = canonicalize_grasps_forward_roll(
            gg_all,
            T_base_cam,
            T_tcp_flange,
            ref_roll=ref_roll,
            roll_threshold_rad=roll_thresh,
        )
        if gg_masked is not None:
            gg_masked = canonicalize_grasps_forward_roll(
                gg_masked,
                T_base_cam,
                T_tcp_flange,
                ref_roll=ref_roll,
                roll_threshold_rad=roll_thresh,
            )
            gg = gg_masked
        else:
            gg = gg_all

    if not cfgs.no_approach_filter:
        ideal_tilt = (
            cfgs.approach_ideal_tilt_deg
            if cfgs.approach_ideal_tilt_deg is not None
            else (45.0 if cfgs.top_down_grasp else DEFAULT_APPROACH_IDEAL_TILT_DEG)
        )
        gg = filter_grasps_by_approach_orientation(
            gg,
            T_base_cam,
            ref_z=cfgs.approach_ref_z,
            ideal_tilt_deg=ideal_tilt,
            score_weight=cfgs.approach_score_weight,
        )

    gg_pick = gg[0:20]
    if len(gg_pick) > 0:
        _log(f'Keep {len(gg_pick)} grasps.')
    else:
        print('No grasps remain after filtering.')

    if len(gg_pick) > 0:
        # print(f'\n=== Top grasps / EEF poses in {arm} base frame ===')
        # for i in range(min(5, len(gg_pick))):
        #     g = gg_pick[i]
        #     T_cam_tcp = np.eye(4, dtype=np.float64)
        #     T_cam_tcp[:3, :3] = g.rotation_matrix
        #     T_cam_tcp[:3, 3] = g.translation
        #     T_base_tcp = T_base_cam @ T_cam_tcp
        #     T_base_flange = T_base_tcp @ T_tcp_flange

        #     Rb_flange = T_base_flange[:3, :3]
        #     tb_flange = T_base_flange[:3, 3]
        #     qb_flange_wxyz = _R_to_quat_wxyz(Rb_flange)
        #     rpy_flange = _R_to_rpy_rad(Rb_flange)
        #     approach_axis_b = T_base_tcp[:3, :3][:, 0]
        #     tb_pre_flange = tb_flange - approach_axis_b * float(PREGRASP_OFFSET_M)

        #     print(f'[{i}] score={g.score:.3f} width={g.width:.3f} depth={g.depth:.3f}')
        #     print('    T_base_flange.t =', tb_flange)
        #     print('    T_base_flange.q(wxyz) =', qb_flange_wxyz)
        #     print('    T_base_flange.rpy(rad) =', rpy_flange)
        #     print('    pregrasp_flange.t =', tb_pre_flange)

        best = gg_pick[0]
        xyzrpy_grasp = _grasp_to_xyzrpy(
            best,
            T_base_cam,
            T_tcp_flange,
            apply_grasp_depth=apply_grasp_depth,
        )
        from piper_pose_ik import canonicalize_flange_xyzrpy_camera_up

        xyzrpy_grasp, flipped_save = canonicalize_flange_xyzrpy_camera_up(xyzrpy_grasp)
        if flipped_save:
            _log('Saved grasp: flipped 180° about approach so wrist camera points up in arm base')
        gripper_open_m = float(np.clip(best.width, 0.0, cfgs.max_gripper_width))
        pose_out = np.concatenate([xyzrpy_grasp, [gripper_open_m]])
        _save_vec_npy(EEF_POSE_XYZRPY_NPY, pose_out)

        if cfgs.quiet:
            print(
                f'AnyGrasp ({arm}): {len(gg_pick)} grasps -> eef_pose_xyzrpy.npy  '
                f'width={gripper_open_m:.4f}m  depth={float(best.depth):.4f}m  score={best.score:.3f}'
            )
        else:
            print('\nSaved eef_pose_xyzrpy.npy')
            print(f'  xyzrpy(rad)= {xyzrpy_grasp}')
            print(f'  grasp_width(m)= {gripper_open_m:.4f}  (AnyGrasp jaw opening at contact)')
            print(
                f'  grasp_depth(m)= {float(best.depth):.4f}  '
                f'({"applied" if apply_grasp_depth else "ignored"} along approach for execution TCP)'
            )

    if cfgs.gui_viz:
        gg_best = gg_pick[0:1] if len(gg_pick) > 0 else GraspGroup()
        if cloud is not None:
            cloud_gui = cloud
        else:
            cloud_gui = _make_o3d_point_cloud(points, point_colors)
        visualize_grasps_gui(
            cloud_gui,
            gg_all,
            gg_masked,
            gg_best,
            T_base_cam,
            T_tcp_flange,
            arm=arm,
            show_saved_triad=True,
            apply_grasp_depth=apply_grasp_depth,
        )

    if cfgs.debug:
        cloud_viz = _make_o3d_point_cloud(points, point_colors)
        cloud_label = 'all points'
        _visualize_grasps_camera_frame(
            gg_all,
            cloud_viz,
            f'Camera frame: {cloud_label}, all grasps (NMS)',
        )
        _visualize_grasps_camera_frame(
            gg_pick[0:1],
            cloud_viz,
            f'Camera frame: {cloud_label}, best grasp',
        )
        if len(gg_pick) > 0:
            visualize_in_base_frame(
                cloud,
                gg_pick,
                T_base_cam,
                T_tcp_flange,
                arm=arm,
                max_grasps=5,
                show_saved_npy=True,
                apply_grasp_depth=apply_grasp_depth,
            )


if __name__ == '__main__':
    _cfgs = parse_args()
    run_grasp(_cfgs.data_dir, _cfgs)
