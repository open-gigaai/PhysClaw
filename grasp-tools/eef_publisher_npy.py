"""Publish EEF7 from run_grasp.py npy output.

Reads `eef_pose_xyzrpy.npy` by default: shape (7,) =
[x, y, z, roll, pitch, yaw, grasp_width] in radians/meters at the GRASP (contact)
flange pose in piper_left base. Approach uses AnyGrasp width + margin; close uses
width * close_scale by default for a tighter grasp.

If the arm misses in space but base-frame viz looks right on the object, check Piper
rpy convention (--override-rpy). If the yellow z=0 sheet and table in the cloud are not
parallel, fix T_BASE_CAM (hand-eye / cam2base chain) before tuning T_TCP_FLANGE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from tf.transformations import euler_from_quaternion


_DEFAULT_XYZRPY_NPY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eef_pose_xyzrpy.npy",
)

# Safe idle / homing pose for the non-commanded arm (from eef_publisher.py examples).
_DEFAULT_IDLE7 = [0.06, 0.0, 0.21, -3.14, 1.5, -3.14, 0.1]
_DEFAULT_HOME7 = list(_DEFAULT_IDLE7)

@dataclass(frozen=True)
class EEF7:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    gripper: float

    @staticmethod
    def from_seq(seq: Sequence[float], *, name: str = "eef") -> "EEF7":
        if len(seq) != 7:
            raise ValueError(f"{name} must have 7 floats: [x,y,z,roll,pitch,yaw,gripper], got len={len(seq)}")
        return EEF7(
            x=float(seq[0]),
            y=float(seq[1]),
            z=float(seq[2]),
            roll=float(seq[3]),
            pitch=float(seq[4]),
            yaw=float(seq[5]),
            gripper=float(seq[6]),
        )

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw, self.gripper]


def load_grasp_pose_npy(path: str) -> tuple[np.ndarray, float]:
    """Load run_grasp output: (7,) [x,y,z,roll,pitch,yaw, grasp_width] in rad/m."""
    v = np.load(path)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.shape != (7,):
        raise ValueError(
            f"{path}: expected shape (7,) [x,y,z,roll,pitch,yaw,grasp_width], got {v.shape}"
        )
    return v[:6], float(v[6])


def xyzrpy_to_eef7(xyzrpy: Sequence[float], *, gripper: float = 0.1) -> list[float]:
    if len(xyzrpy) != 6:
        raise ValueError(f"xyzrpy must have 6 floats, got len={len(xyzrpy)}")
    return [float(xyzrpy[i]) for i in range(6)] + [float(gripper)]


def resolve_grasp_gripper_widths(
    *,
    grasp_width_m: float,
    margin_m: float,
    close_scale: float,
    close_override: float | None,
) -> tuple[float, float, str]:
    """Return (approach_open_m, grasp_close_m) Piper gripper commands (0=closed, 0.1=open).

    AnyGrasp width is treated as clearance for approach; close uses width * close_scale
    for a tighter final grasp unless --gripper-close overrides.
    """
    raw = float(grasp_width_m)
    grasp_m = float(np.clip(raw, 0.0, 0.1))
    src = f"eef_pose_xyzrpy.npy ({raw:.4f}m)"
    open_m = float(np.clip(grasp_m + margin_m, 0.0, 0.1))
    if close_override is not None:
        close_m = float(np.clip(close_override, 0.0, 0.1))
    else:
        close_m = float(np.clip(grasp_m * float(close_scale), 0.0, 0.1))
    return open_m, close_m, src


class EndPoseTracker:
    """Subscribe to Piper puppet end-pose feedback for reach verification."""

    def __init__(self, arm: str) -> None:
        self._pose: PoseStamped | None = None
        topic = f"/puppet/end_pose_{arm}" if arm in ("left", "right") else "/puppet/end_pose_left"
        rospy.Subscriber(topic, PoseStamped, self._on_pose, queue_size=1)
        rospy.sleep(0.05)

    def _on_pose(self, msg: PoseStamped) -> None:
        self._pose = msg

    @staticmethod
    def _xyzrpy_from_pose(pose) -> np.ndarray:
        p = pose.position
        q = pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=np.float64)

    def pose_error(self, target_xyzrpy: Sequence[float]) -> tuple[float, float]:
        if self._pose is None:
            return float("inf"), float("inf")
        cur = self._xyzrpy_from_pose(self._pose.pose)
        tgt = np.asarray(target_xyzrpy, dtype=np.float64).reshape(6)
        pos_err = float(np.linalg.norm(cur[:3] - tgt[:3]))
        rot_err = float(np.linalg.norm(cur[3:] - tgt[3:]))
        return pos_err, rot_err

    def wait_reached(
        self,
        target_xyzrpy: Sequence[float],
        *,
        tol_pos_m: float,
        tol_rot_rad: float,
        timeout_s: float,
        rate_hz: int = 30,
    ) -> bool:
        rate = rospy.Rate(rate_hz)
        deadline = rospy.Time.now() + rospy.Duration(float(timeout_s))
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            pos_err, rot_err = self.pose_error(target_xyzrpy)
            if pos_err <= tol_pos_m and rot_err <= tol_rot_rad:
                return True
            rate.sleep()
        pos_err, rot_err = self.pose_error(target_xyzrpy)
        rospy.logwarn(
            "Pose not reached within %.1fs (pos_err=%.3fm rot_err=%.3frad)",
            timeout_s,
            pos_err,
            rot_err,
        )
        return False


class EEFPublisher:
    """Publish EEF commands as `piper_msgs/PosCmd`."""

    def __init__(
        self,
        *,
        node_name: str = "eef_publisher",
        left_topic: str = "/piper_left/pos_cmd",
        right_topic: str = "/piper_right/pos_cmd",
        queue_size: int = 10,
        anonymous: bool = True,
        disable_signals: bool = False,
    ) -> None:
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=anonymous, disable_signals=disable_signals)
        self._left_pub = rospy.Publisher(left_topic, PosCmd, queue_size=queue_size)
        self._right_pub = rospy.Publisher(right_topic, PosCmd, queue_size=queue_size)
        self._subscriber_checked = False

    def _warn_if_no_subscribers(self) -> None:
        """Warn once if PosCmd topics have no subscribers after a short discovery wait."""
        if self._subscriber_checked:
            return
        self._subscriber_checked = True

        deadline = rospy.Time.now() + rospy.Duration(1.0)
        rate = rospy.Rate(50)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self._left_pub.get_num_connections() > 0 and self._right_pub.get_num_connections() > 0:
                return
            rate.sleep()

        left_n = self._left_pub.get_num_connections()
        right_n = self._right_pub.get_num_connections()
        if left_n == 0 or right_n == 0:
            rospy.logwarn(
                "No subscriber on PosCmd topics (left=%d right=%d). Is piper running with mode:=1?",
                left_n,
                right_n,
            )

    @staticmethod
    def _to_msg(eef: EEF7) -> PosCmd:
        msg = PosCmd()
        msg.x = eef.x
        msg.y = eef.y
        msg.z = eef.z
        msg.roll = eef.roll
        msg.pitch = eef.pitch
        msg.yaw = eef.yaw
        msg.gripper = eef.gripper
        return msg

    def publish(self, left7: Sequence[float], right7: Sequence[float]) -> None:
        left = left7 if isinstance(left7, EEF7) else EEF7.from_seq(left7, name="left7")
        right = right7 if isinstance(right7, EEF7) else EEF7.from_seq(right7, name="right7")

        self._warn_if_no_subscribers()

        self._left_pub.publish(self._to_msg(left))
        self._right_pub.publish(self._to_msg(right))

    def publish_for(
        self,
        left7: Sequence[float],
        right7: Sequence[float],
        *,
        duration_s: float = 2.0,
        rate_hz: int = 30,
        pose_tracker: EndPoseTracker | None = None,
        wait_target_xyzrpy: Sequence[float] | None = None,
        wait_tol_pos_m: float = 0.025,
        wait_tol_rot_rad: float = 0.2,
        abort_on_pose_fail: bool = False,
        phase_name: str = "phase",
    ) -> None:
        rate = rospy.Rate(rate_hz)
        n = max(1, int(round(float(duration_s) * float(rate_hz))))
        for _ in range(n):
            if rospy.is_shutdown():
                return
            self.publish(left7, right7)
            rate.sleep()

        if pose_tracker is None or wait_target_xyzrpy is None:
            return
        extra_wait = max(0.5, float(duration_s) * 0.5)
        reached = pose_tracker.wait_reached(
            wait_target_xyzrpy,
            tol_pos_m=wait_tol_pos_m,
            tol_rot_rad=wait_tol_rot_rad,
            timeout_s=extra_wait,
            rate_hz=rate_hz,
        )
        if not reached and abort_on_pose_fail:
            raise RuntimeError(f"{phase_name}: arm did not reach target pose (IK likely failed)")

    def publish_sequence(
        self,
        phases: Sequence[tuple[Sequence[float], Sequence[float], float]],
        *,
        rate_hz: int = 30,
        phase_names: Sequence[str] | None = None,
        pose_tracker: EndPoseTracker | None = None,
        wait_arm: str | None = None,
        wait_tol_pos_m: float = 0.025,
        wait_tol_rot_rad: float = 0.2,
        abort_on_pose_fail: bool = False,
    ) -> None:
        for i, (left7, right7, duration_s) in enumerate(phases):
            if rospy.is_shutdown():
                return
            name = phase_names[i] if phase_names is not None and i < len(phase_names) else f"phase_{i}"
            wait_xyz = None
            if pose_tracker is not None and wait_arm is not None:
                target7 = left7 if wait_arm == "left" else right7 if wait_arm == "right" else left7
                wait_xyz = target7[:6]
            self.publish_for(
                left7,
                right7,
                duration_s=duration_s,
                rate_hz=rate_hz,
                pose_tracker=pose_tracker,
                wait_target_xyzrpy=wait_xyz,
                wait_tol_pos_m=wait_tol_pos_m,
                wait_tol_rot_rad=wait_tol_rot_rad,
                abort_on_pose_fail=abort_on_pose_fail,
                phase_name=name,
            )

    def publish_waypoints(
        self,
        left_waypoints: Sequence[Sequence[float]],
        right_waypoints: Sequence[Sequence[float]],
        *,
        duration_s: float,
        segment_weights: Sequence[float] | None = None,
        rate_hz: int = 30,
    ) -> None:
        """Linearly interpolate through waypoints (one continuous motion, no per-waypoint hold)."""
        if len(left_waypoints) != len(right_waypoints) or len(left_waypoints) < 2:
            raise ValueError("left/right waypoints must have the same length (>= 2)")
        weights = _normalize_segment_weights(segment_weights, n_segments=len(left_waypoints) - 1)

        rate = rospy.Rate(rate_hz)
        n = max(1, int(round(float(duration_s) * float(rate_hz))))
        for i in range(n):
            if rospy.is_shutdown():
                return
            u = i / max(n - 1, 1)
            left7 = _interp_along_waypoints(left_waypoints, weights, u)
            right7 = _interp_along_waypoints(right_waypoints, weights, u)
            self.publish(left7, right7)
            rate.sleep()


def _lerp7(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    t = float(np.clip(t, 0.0, 1.0))
    return [float(a[i]) + t * (float(b[i]) - float(a[i])) for i in range(7)]


def _normalize_segment_weights(
    segment_weights: Sequence[float] | None,
    *,
    n_segments: int,
) -> list[float]:
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")
    if segment_weights is None:
        return [1.0 / n_segments] * n_segments
    if len(segment_weights) != n_segments:
        raise ValueError(f"segment_weights length {len(segment_weights)} != n_segments {n_segments}")
    total = float(sum(segment_weights))
    if total <= 0.0:
        raise ValueError("segment_weights must sum to a positive value")
    return [float(w) / total for w in segment_weights]


def _interp_along_waypoints(
    waypoints: Sequence[Sequence[float]],
    segment_weights: Sequence[float],
    u: float,
) -> list[float]:
    u = float(np.clip(u, 0.0, 1.0))
    cum = 0.0
    for seg_i, weight in enumerate(segment_weights):
        seg_end = cum + weight
        if u <= seg_end or seg_i == len(segment_weights) - 1:
            local_t = (u - cum) / weight if weight > 0.0 else 1.0
            return _lerp7(waypoints[seg_i], waypoints[seg_i + 1], local_t)
        cum = seg_end
    return _lerp7(waypoints[-2], waypoints[-1], 1.0)


def _parse_7floats(arg: Sequence[str], *, name: str) -> list[float]:
    if len(arg) != 7:
        raise SystemExit(f"{name} expects 7 floats, got {len(arg)}")
    return [float(x) for x in arg]


def _assign_arm_poses(
    target7: Sequence[float],
    idle7: Sequence[float],
    arm: str,
) -> tuple[list[float], list[float]]:
    if arm == "left":
        return list(target7), list(idle7)
    if arm == "right":
        return list(idle7), list(target7)
    return list(target7), list(target7)


def _rpy_rad_to_R(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in np.asarray(rpy, dtype=np.float64).reshape(3)]
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def apply_approach_depth_offset(
    grasp_xyzrpy: Sequence[float],
    *,
    offset_m: float,
) -> np.ndarray:
    """Shift grasp contact along flange +Z (approach) in the arm base frame.

    Positive offset moves deeper toward the object; default pipeline offset is 0.
    """
    v = np.asarray(grasp_xyzrpy, dtype=np.float64).reshape(6)
    if float(offset_m) == 0.0:
        return v.copy()
    approach = _rpy_rad_to_R(v[3:])[:, 2]
    shifted = v.copy()
    shifted[:3] += approach * float(offset_m)
    return shifted


def build_grasp_phases(
    grasp_xyzrpy: np.ndarray,
    *,
    arm: str,
    idle7: Sequence[float],
    gripper_open: float,
    gripper_close: float,
    grasp_duration_s: float,
    close_duration_s: float,
) -> list[tuple[list[float], list[float], float]]:
    grasp_open7 = xyzrpy_to_eef7(grasp_xyzrpy, gripper=gripper_open)
    grasp_close7 = xyzrpy_to_eef7(grasp_xyzrpy, gripper=gripper_close)
    return [
        (*_assign_arm_poses(grasp_open7, idle7, arm), grasp_duration_s),
        (*_assign_arm_poses(grasp_close7, idle7, arm), close_duration_s),
    ]


def lift_xyzrpy_from_grasp(
    grasp_xyzrpy: Sequence[float],
    *,
    offset_m: float,
) -> np.ndarray:
    """Lift contact pose straight up (+Z in arm base frame)."""
    v = np.asarray(grasp_xyzrpy, dtype=np.float64).reshape(6)
    lift_xyz = v[:3].copy()
    lift_xyz[2] += float(offset_m)
    return np.concatenate([lift_xyz, v[3:]])


def lower_xyzrpy_from_pose(
    xyzrpy: Sequence[float],
    *,
    offset_m: float,
) -> np.ndarray:
    """Lower pose straight down (-Z in arm base frame)."""
    v = np.asarray(xyzrpy, dtype=np.float64).reshape(6)
    lower_xyz = v[:3].copy()
    lower_xyz[2] -= float(offset_m)
    return np.concatenate([lower_xyz, v[3:]])


def resolve_place_xyzrpy(
    transport_xyzrpy: Sequence[float],
    *,
    lower_before_release_m: float,
) -> np.ndarray:
    """Place pose at transport height, optionally lowered before gripper release."""
    transport = np.asarray(transport_xyzrpy, dtype=np.float64).reshape(6)
    if lower_before_release_m > 0.0:
        return lower_xyzrpy_from_pose(transport, offset_m=lower_before_release_m)
    return transport


def build_place_phases(
    grasp_xyzrpy: np.ndarray,
    *,
    place_x: float,
    place_y: float,
    arm: str,
    idle7: Sequence[float],
    gripper_close: float,
    gripper_open: float,
    lift_offset_m: float,
    lift_duration_s: float,
    transport_duration_s: float,
    release_duration_s: float,
    lower_before_release_m: float = 0.0,
    lower_duration_s: float = 0.4,
    retract_after_release_m: float,
    retract_duration_s: float,
    append_retract: bool = True,
    lifted_xyzrpy: np.ndarray | None = None,
    transport_xyzrpy: np.ndarray | None = None,
    skip_lift_transport: bool = False,
) -> list[tuple[list[float], list[float], float]]:
    """Move grasped object to TCP (place_x, place_y) at lifted height, then place and open gripper.

    After transport: optionally lower, release gripper, lift up, then return home.
    When append_retract is False, retract is omitted here (e.g. merged into return-home).
    Pass lifted_xyzrpy/transport_xyzrpy when IK pre-check adjusted targets.
    Set skip_lift_transport=True when lift+transport run as one waypoint motion.
    """
    grasp = np.asarray(grasp_xyzrpy, dtype=np.float64).reshape(6)
    if lifted_xyzrpy is None:
        lifted = lift_xyzrpy_from_grasp(grasp, offset_m=lift_offset_m)
    else:
        lifted = np.asarray(lifted_xyzrpy, dtype=np.float64).reshape(6)
    if transport_xyzrpy is None:
        transport = place_transport_xyzrpy(
            grasp,
            place_x=place_x,
            place_y=place_y,
            lift_offset_m=lift_offset_m,
        )
    else:
        transport = np.asarray(transport_xyzrpy, dtype=np.float64).reshape(6)

    place_xyzrpy = resolve_place_xyzrpy(transport, lower_before_release_m=lower_before_release_m)
    lift7 = xyzrpy_to_eef7(lifted, gripper=gripper_close)
    transport7 = xyzrpy_to_eef7(transport, gripper=gripper_close)
    lower7 = xyzrpy_to_eef7(place_xyzrpy, gripper=gripper_close)
    release7 = xyzrpy_to_eef7(place_xyzrpy, gripper=gripper_open)

    phases: list[tuple[list[float], list[float], float]] = []
    if not skip_lift_transport:
        phases.extend(
            [
                (*_assign_arm_poses(lift7, idle7, arm), lift_duration_s),
                (*_assign_arm_poses(transport7, idle7, arm), transport_duration_s),
            ]
        )
    if lower_before_release_m > 0.0:
        phases.append((*_assign_arm_poses(lower7, idle7, arm), lower_duration_s))
    phases.append((*_assign_arm_poses(release7, idle7, arm), release_duration_s))
    if append_retract and retract_after_release_m > 0.0:
        retract_xyzrpy = lift_xyzrpy_from_grasp(place_xyzrpy, offset_m=retract_after_release_m)
        retract7 = xyzrpy_to_eef7(retract_xyzrpy, gripper=gripper_open)
        phases.append((*_assign_arm_poses(retract7, idle7, arm), retract_duration_s))
    return phases


def build_lift_transport_waypoints(
    grasp_close7: Sequence[float],
    lifted_xyzrpy: np.ndarray,
    transport_xyzrpy: np.ndarray,
    *,
    arm: str,
    idle7: Sequence[float],
    gripper_close: float,
) -> tuple[list[list[float]], list[list[float]]]:
    """Waypoints for one continuous lift-then-transport motion after grasp close."""
    lift7 = xyzrpy_to_eef7(lifted_xyzrpy, gripper=gripper_close)
    transport7 = xyzrpy_to_eef7(transport_xyzrpy, gripper=gripper_close)
    close7 = list(grasp_close7)
    if arm == "left":
        return [close7, lift7, transport7], [list(idle7), list(idle7), list(idle7)]
    if arm == "right":
        return [list(idle7), list(idle7), list(idle7)], [close7, lift7, transport7]
    return [close7, lift7, transport7], [close7, lift7, transport7]


def build_retract_home_waypoints(
    transport_xyzrpy: np.ndarray,
    *,
    arm: str,
    idle7: Sequence[float],
    home_left7: Sequence[float],
    home_right7: Sequence[float],
    gripper_open: float,
    lower_before_release_m: float = 0.0,
    retract_after_release_m: float,
    retract_fraction: float,
    skip_release_waypoint: bool = False,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    """Waypoints for one continuous lift-then-home motion after gripper release."""
    place_xyzrpy = resolve_place_xyzrpy(
        transport_xyzrpy,
        lower_before_release_m=lower_before_release_m,
    )
    release7 = xyzrpy_to_eef7(place_xyzrpy, gripper=gripper_open)
    retract_xyzrpy = lift_xyzrpy_from_grasp(place_xyzrpy, offset_m=retract_after_release_m)
    retract7 = xyzrpy_to_eef7(retract_xyzrpy, gripper=gripper_open)

    if skip_release_waypoint:
        if arm == "left":
            left_wps = [list(retract7), list(home_left7)]
            right_wps = [list(idle7), list(home_right7)]
        elif arm == "right":
            left_wps = [list(idle7), list(home_left7)]
            right_wps = [list(retract7), list(home_right7)]
        else:
            left_wps = [list(retract7), list(home_left7)]
            right_wps = [list(retract7), list(home_right7)]
        return left_wps, right_wps, [1.0]

    if arm == "left":
        left_wps = [list(release7), list(retract7), list(home_left7)]
        right_wps = [list(idle7), list(idle7), list(home_right7)]
    elif arm == "right":
        left_wps = [list(idle7), list(idle7), list(home_left7)]
        right_wps = [list(release7), list(retract7), list(home_right7)]
    else:
        left_wps = [list(release7), list(retract7), list(home_left7)]
        right_wps = [list(release7), list(retract7), list(home_right7)]

    frac = float(np.clip(retract_fraction, 0.05, 0.5))
    return left_wps, right_wps, [frac, 1.0 - frac]


def place_transport_xyzrpy(
    grasp_xyzrpy: np.ndarray,
    *,
    place_x: float,
    place_y: float,
    lift_offset_m: float,
) -> np.ndarray:
    """Transport pose: TCP xy at (place_x, place_y), flange z at lifted height."""
    from piper_pose_ik import flange_xyz_from_tcp_xy

    grasp = np.asarray(grasp_xyzrpy, dtype=np.float64).reshape(6)
    lifted = lift_xyzrpy_from_grasp(grasp, offset_m=lift_offset_m)
    flange_xyz = flange_xyz_from_tcp_xy(
        place_x,
        place_y,
        grasp[3:],
        flange_z=float(lifted[2]),
    )
    return np.concatenate([flange_xyz, grasp[3:]])


def build_home_phase(
    *,
    home_left7: Sequence[float],
    home_right7: Sequence[float],
    duration_s: float,
) -> tuple[list[float], list[float], float]:
    """Return both arms to idle homing pose (gripper open)."""
    return list(home_left7), list(home_right7), float(duration_s)


def resolve_place_poses_with_ik(
    grasp_xyzrpy: np.ndarray,
    *,
    place_x: float,
    place_y: float,
    lift_offset_m: float,
    lower_before_release_m: float = 0.0,
    gripper_close: float,
    carry_rpy_ref: Sequence[float],
    check_ik: bool,
    arm_ik=None,
    place_ik_options=None,
) -> "PlacePosePlan":
    """Resolve place EEF poses; when check_ik, solve IK once per key pose and cache joints."""
    from piper_pose_ik import (
        PlaceIkOptions,
        PlacePosePlan,
        build_ik_solver,
        find_feasible_lift,
        find_feasible_lower,
        find_feasible_transport,
    )

    lifted = lift_xyzrpy_from_grasp(grasp_xyzrpy, offset_m=lift_offset_m)
    transport = place_transport_xyzrpy(
        grasp_xyzrpy,
        place_x=place_x,
        place_y=place_y,
        lift_offset_m=lift_offset_m,
    )
    lower = resolve_place_xyzrpy(transport, lower_before_release_m=lower_before_release_m)
    compromises: list[str] = []

    if not check_ik:
        return PlacePosePlan(
            lifted_xyzrpy=lifted,
            transport_xyzrpy=transport,
            lower_xyzrpy=lower,
            actual_lift_m=float(lift_offset_m),
            actual_lower_m=float(lower_before_release_m),
        )

    opts = place_ik_options if place_ik_options is not None else PlaceIkOptions()
    try:
        solver = arm_ik if arm_ik is not None else build_ik_solver()
    except ImportError as exc:
        rospy.logwarn("IK pre-check skipped (need conda env tv): %s", exc)
        return PlacePosePlan(
            lifted_xyzrpy=lifted,
            transport_xyzrpy=transport,
            lower_xyzrpy=lower,
            actual_lift_m=float(lift_offset_m),
            actual_lower_m=float(lower_before_release_m),
        )

    lifted, actual_lift, joint_lift = find_feasible_lift(
        grasp_xyzrpy,
        requested_offset_m=lift_offset_m,
        gripper=gripper_close,
        arm_ik=solver,
        options=opts,
    )
    transport, transport_z_lower, transport_rpy_blend, joint_transport = find_feasible_transport(
        grasp_xyzrpy,
        place_x=place_x,
        place_y=place_y,
        lifted_xyzrpy=lifted,
        gripper=gripper_close,
        arm_ik=solver,
        carry_rpy_ref=carry_rpy_ref,
        options=opts,
    )
    lower, actual_lower, joint_lower = find_feasible_lower(
        transport,
        requested_lower_m=lower_before_release_m,
        gripper=gripper_close,
        arm_ik=solver,
        options=opts,
    )

    if actual_lift + 1e-6 < float(lift_offset_m):
        msg = (
            f"lift: reduced offset {lift_offset_m:.3f}m -> {actual_lift:.3f}m "
            f"(max reduction {opts.max_lift_reduction_m:.3f}m)"
        )
        compromises.append(msg)
        rospy.logwarn("Place IK: %s", msg)
    if transport_z_lower > 1e-6:
        msg = (
            f"transport: lowered z by {transport_z_lower:.3f}m "
            f"(lifted z={float(lifted[2]):.3f}m -> transport z={float(transport[2]):.3f}m)"
        )
        compromises.append(msg)
        rospy.logwarn("Place IK: %s", msg)
    if transport_rpy_blend > 1e-6:
        compromises.append(
            f"transport: relaxed orientation (rpy blend {transport_rpy_blend:.1f} toward carry/home pose)"
        )
    if actual_lower + 1e-6 < float(lower_before_release_m):
        msg = (
            f"lower: reduced before-release {lower_before_release_m:.3f}m -> {actual_lower:.3f}m "
            f"(max reduction {opts.max_lower_reduction_m:.3f}m)"
        )
        compromises.append(msg)
        rospy.logwarn("Place IK: %s", msg)

    return PlacePosePlan(
        lifted_xyzrpy=lifted,
        transport_xyzrpy=transport,
        lower_xyzrpy=lower,
        actual_lift_m=actual_lift,
        actual_lower_m=actual_lower,
        transport_z_lower_m=transport_z_lower,
        transport_rpy_blend=transport_rpy_blend,
        compromises=tuple(compromises),
        joint_lift=joint_lift,
        joint_transport=joint_transport,
        joint_lower=joint_lower,
    )


def add_place_ik_cli_args(parser) -> None:
    """CLI knobs for resolve_place_poses_with_ik (shared by joint/EEF publishers)."""
    parser.add_argument(
        "--strict-place-ik",
        action="store_true",
        help="Do not reduce lift/transport height for IK; fail if requested poses are infeasible",
    )
    parser.add_argument(
        "--place-ik-max-lift-reduction-m",
        type=float,
        default=0.03,
        help="Max lift height reduction below --place-lift-offset-m when IK fails (default 0.03)",
    )
    parser.add_argument(
        "--place-ik-allow-transport-z-lower",
        action="store_true",
        help="Allow lowering transport z for IK (legacy; can cause table collision)",
    )
    parser.add_argument(
        "--place-ik-max-transport-z-lower-m",
        type=float,
        default=0.02,
        help="Cap transport z lowering when --place-ik-allow-transport-z-lower (default 0.02)",
    )
    parser.add_argument(
        "--place-ik-transport-min-above-grasp-m",
        type=float,
        default=0.05,
        help="Never IK-adjust transport below grasp_z + this margin (default 0.05)",
    )
    parser.add_argument(
        "--place-ik-max-lower-reduction-m",
        type=float,
        default=0.05,
        help="Max reduction below --place-lower-before-release-m when lower IK fails (default 0.05)",
    )


def place_ik_options_from_args(args) -> "PlaceIkOptions":
    from piper_pose_ik import PlaceIkOptions

    if getattr(args, "strict_place_ik", False):
        return PlaceIkOptions.strict()
    return PlaceIkOptions(
        max_lift_reduction_m=float(args.place_ik_max_lift_reduction_m),
        allow_transport_z_lower=bool(args.place_ik_allow_transport_z_lower),
        max_transport_z_lower_m=float(args.place_ik_max_transport_z_lower_m),
        transport_min_above_grasp_m=float(args.place_ik_transport_min_above_grasp_m),
        max_lower_reduction_m=float(args.place_ik_max_lower_reduction_m),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Publish EEF7 from eef_pose_xyzrpy.npy (run_grasp output) via ROS PosCmd",
    )
    parser.add_argument(
        "--npy",
        type=str,
        default=_DEFAULT_XYZRPY_NPY,
        help="Path to (7,) npy: [x,y,z,roll,pitch,yaw,grasp_width]",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.1,
        help="Gripper command for --reach-only (0=closed, 0.1=open)",
    )
    parser.add_argument(
        "--gripper-open-margin",
        type=float,
        default=0.05,
        help="Extra opening (m) added to AnyGrasp width before approach",
    )
    parser.add_argument(
        "--gripper-close-scale",
        type=float,
        default=0.33,
        help="Close target = AnyGrasp width * this factor. Ignored if --gripper-close set",
    )
    parser.add_argument(
        "--gripper-close",
        type=float,
        default=None,
        help="Override close target (m). Default: AnyGrasp width * --gripper-close-scale",
    )
    parser.add_argument(
        "--override-rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("roll", "pitch", "yaw"),
        help="Use pipeline xyz but replace rpy (rad); for isolating orientation errors",
    )
    parser.add_argument(
        "--approach-depth-offset-m",
        type=float,
        default=0.0,
        help="Extra depth (m) along flange approach (+Z) at grasp contact; use small positive values if grasp is too shallow",
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="left",
        help="Which arm(s) receive the npy pose; the other arm uses --idle",
    )
    parser.add_argument(
        "--idle",
        nargs=7,
        default=None,
        metavar=("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
        help="EEF7 for the non-commanded arm when --arm is left or right",
    )
    parser.add_argument(
        "--reach-only",
        action="store_true",
        help="Only publish contact pose (no grasp / close sequence)",
    )
    parser.add_argument("--grasp-duration-s", type=float, default=2, help="Move to grasp pose with gripper open")
    parser.add_argument("--close-duration-s", type=float, default=0.5)
    parser.add_argument(
        "--place-x",
        type=float,
        default=None,
        help="Place target TCP x (m) in arm base frame; requires --place-y",
    )
    parser.add_argument(
        "--place-y",
        type=float,
        default=None,
        help="Place target TCP y (m) in arm base frame; requires --place-x",
    )
    parser.add_argument(
        "--place-z",
        type=float,
        default=None,
        help="Deprecated, ignored: release opens at lowered place height",
    )
    parser.add_argument("--place-lift-offset-m", type=float, default=0.15, help="Lift straight up (+Z) after grasp before moving to place xy")
    parser.add_argument("--place-lift-duration-s", type=float, default=0.5)
    parser.add_argument("--place-transport-duration-s", type=float, default=0.5)
    parser.add_argument(
        "--place-lower-before-release-m",
        type=float,
        default=0.05,
        help="Lower (-Z) at place target before opening gripper (0 to skip)",
    )
    parser.add_argument("--place-lower-duration-s", type=float, default=0.4)
    parser.add_argument("--place-release-duration-s", type=float, default=0.5, help="Hold at place pose while gripper opens")
    parser.add_argument("--place-retract-after-release-m", type=float, default=0.08, help="Lift (+Z) after release before return home (0 to skip)")
    parser.add_argument(
        "--place-retract-duration-s",
        type=float,
        default=1.0,
        help="Retract hold duration when --no-home-after-place (ignored when returning home)",
    )
    parser.add_argument(
        "--place-retract-home-fraction",
        type=float,
        default=0.2,
        help="Share of --home-duration-s spent lifting before blending into home (default 0.2)",
    )
    parser.add_argument(
        "--no-home-after-place",
        action="store_true",
        help="After place + release, do not return both arms to homing pose",
    )
    parser.add_argument(
        "--home-left",
        nargs=7,
        default=None,
        metavar=("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
        help="Left arm homing EEF7 (default: safe idle pose, gripper open)",
    )
    parser.add_argument(
        "--home-right",
        nargs=7,
        default=None,
        metavar=("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
        help="Right arm homing EEF7 (default: safe idle pose, gripper open)",
    )
    parser.add_argument("--home-duration-s", type=float, default=1.5, help="Return-home duration (includes retract lift when enabled)")
    parser.add_argument("--duration-s", type=float, default=2.0, help="Reach-only publish duration")
    parser.add_argument("--rate-hz", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Publish once and exit")
    parser.add_argument("--left-topic", type=str, default="/piper_left/pos_cmd")
    parser.add_argument("--right-topic", type=str, default="/piper_right/pos_cmd")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print a one-line execution summary instead of full pose details",
    )
    parser.add_argument(
        "--check-ik",
        action="store_true",
        default=True,
        help="Pre-check/adjust lift+transport+lower poses with Pinocchio IK (default: on; needs tv env)",
    )
    parser.add_argument(
        "--no-check-ik",
        action="store_false",
        dest="check_ik",
        help="Skip Pinocchio IK pre-check",
    )
    add_place_ik_cli_args(parser)
    parser.add_argument(
        "--wait-pose",
        action="store_true",
        default=True,
        help="After each phase, verify arm reached target via /puppet/end_pose_* (default: on)",
    )
    parser.add_argument(
        "--no-wait-pose",
        action="store_false",
        dest="wait_pose",
        help="Do not verify pose convergence between phases",
    )
    parser.add_argument(
        "--abort-on-pose-fail",
        action="store_true",
        default=True,
        help="Stop sequence if a phase does not reach target (default: on)",
    )
    parser.add_argument(
        "--no-abort-on-pose-fail",
        action="store_false",
        dest="abort_on_pose_fail",
        help="Continue even when pose verification fails (legacy behavior)",
    )
    parser.add_argument("--pose-tol-pos-m", type=float, default=0.025, help="Position tolerance for --wait-pose")
    parser.add_argument("--pose-tol-rot-rad", type=float, default=0.2, help="RPY tolerance for --wait-pose")
    args = parser.parse_args()

    reach_gripper = args.gripper_open
    do_grasp = not args.reach_only
    do_place = args.place_x is not None or args.place_y is not None
    if do_place and (args.place_x is None or args.place_y is None):
        raise SystemExit("--place-x and --place-y must be given together")
    if do_place and not do_grasp:
        raise SystemExit("--place-x/--place-y require full grasp sequence (omit --reach-only)")

    xyzrpy, grasp_width_m = load_grasp_pose_npy(args.npy)
    if args.override_rpy is not None:
        xyzrpy = np.concatenate([xyzrpy[:3], np.asarray(args.override_rpy, dtype=np.float64)])
    if args.approach_depth_offset_m != 0.0:
        xyzrpy = apply_approach_depth_offset(xyzrpy, offset_m=args.approach_depth_offset_m)

    if do_grasp:
        gripper_open, gripper_close, gripper_src = resolve_grasp_gripper_widths(
            grasp_width_m=grasp_width_m,
            margin_m=args.gripper_open_margin,
            close_scale=args.gripper_close_scale,
            close_override=args.gripper_close,
        )
    idle7 = list(_DEFAULT_IDLE7) if args.idle is None else _parse_7floats(args.idle, name="--idle")
    home_left7 = list(_DEFAULT_HOME7) if args.home_left is None else _parse_7floats(args.home_left, name="--home-left")
    home_right7 = list(_DEFAULT_HOME7) if args.home_right is None else _parse_7floats(args.home_right, name="--home-right")
    home_after_place = do_place and not args.no_home_after_place

    pub = EEFPublisher(left_topic=args.left_topic, right_topic=args.right_topic)
    pose_tracker = EndPoseTracker(args.arm) if args.wait_pose else None
    seq_common = {
        "rate_hz": args.rate_hz,
        "pose_tracker": pose_tracker,
        "wait_arm": args.arm if args.wait_pose else None,
        "wait_tol_pos_m": args.pose_tol_pos_m,
        "wait_tol_rot_rad": args.pose_tol_rot_rad,
        "abort_on_pose_fail": args.abort_on_pose_fail,
    }

    lifted_xyzrpy: np.ndarray | None = None
    transport_xyzrpy: np.ndarray | None = None
    actual_lower_before_release_m = 0.0
    if do_place:
        place_plan = resolve_place_poses_with_ik(
            xyzrpy,
            place_x=float(args.place_x),
            place_y=float(args.place_y),
            lift_offset_m=args.place_lift_offset_m,
            lower_before_release_m=args.place_lower_before_release_m,
            gripper_close=gripper_close if do_grasp else reach_gripper,
            carry_rpy_ref=idle7[3:6],
            check_ik=args.check_ik,
            place_ik_options=place_ik_options_from_args(args),
        )
        lifted_xyzrpy = place_plan.lifted_xyzrpy
        transport_xyzrpy = place_plan.transport_xyzrpy
        actual_lower_before_release_m = place_plan.actual_lower_m

    if args.quiet:
        if do_place:
            lifted_z = float(lifted_xyzrpy[2]) if lifted_xyzrpy is not None else lift_xyzrpy_from_grasp(xyzrpy, offset_m=args.place_lift_offset_m)[2]
            print(
                f"Execute ({args.arm}): grasp -> place ({args.place_x:.2f}, {args.place_y:.2f}, "
                f"z={lifted_z:.2f})  width={grasp_width_m:.4f}m"
            )
        elif do_grasp:
            print(f"Execute ({args.arm}): grasp  width={grasp_width_m:.4f}m")
        else:
            print(f"Execute ({args.arm}): reach  width={reach_gripper:.4f}")
    else:
        print(f"Loaded {args.npy}")
        print("  Pose type: GRASP flange (contact), from run_grasp eef_pose_xyzrpy.npy")
        print(f"  xyzrpy (rad): {xyzrpy.tolist()}")
        if args.approach_depth_offset_m != 0.0:
            print(f"  approach_depth_offset(m): {args.approach_depth_offset_m:.4f}")
        print(f"  grasp_width(m): {grasp_width_m:.4f}")
        if do_grasp:
            print(f"  gripper open:     {gripper_open:.4f}  ({gripper_src} + {args.gripper_open_margin:.4f}m margin)")
            if args.gripper_close is not None:
                close_desc = f"override (--gripper-close {args.gripper_close:.4f})"
            else:
                close_desc = f"{gripper_src} * {args.gripper_close_scale:.2f}"
            print(f"  gripper close:    {gripper_close:.4f}  ({close_desc})")
            seq = f"grasp({args.grasp_duration_s}s) -> close({args.close_duration_s}s)"
            if do_place:
                seq += (
                    f" -> lift({args.place_lift_duration_s}s)"
                    f" -> transport({args.place_transport_duration_s}s)"
                )
                if args.place_lower_before_release_m > 0.0:
                    seq += f" -> lower({args.place_lower_duration_s}s)"
                seq += f" -> release({args.place_release_duration_s}s)"
                if args.place_retract_after_release_m > 0.0:
                    if home_after_place:
                        seq += f" -> retract+home({args.home_duration_s}s)"
                    else:
                        seq += f" -> retract({args.place_retract_duration_s}s)"
                elif home_after_place:
                    seq += f" -> home({args.home_duration_s}s)"
                lifted_z = float(lifted_xyzrpy[2]) if lifted_xyzrpy is not None else lift_xyzrpy_from_grasp(xyzrpy, offset_m=args.place_lift_offset_m)[2]
                print(f"  place target: ({args.place_x:.4f}, {args.place_y:.4f}, z={lifted_z:.4f} lifted) m")
                if args.place_z is not None:
                    print("  note: --place-z is ignored (release at lowered transport height)")
            print(f"  sequence: {seq}")
        else:
            target7 = xyzrpy_to_eef7(xyzrpy, gripper=reach_gripper)
            left7, right7 = _assign_arm_poses(target7, idle7, args.arm)
            print(f"  eef7:         {target7}")
            print(f"  arm={args.arm}  left7={left7}")
            print(f"               right7={right7}")

    if do_grasp:
        phases = build_grasp_phases(
            xyzrpy,
            arm=args.arm,
            idle7=idle7,
            gripper_open=gripper_open,
            gripper_close=gripper_close,
            grasp_duration_s=args.grasp_duration_s,
            close_duration_s=args.close_duration_s,
        )
        if do_place:
            merge_retract_home = home_after_place and args.place_retract_after_release_m > 0.0
            assert lifted_xyzrpy is not None and transport_xyzrpy is not None
            place_phases = build_place_phases(
                xyzrpy,
                place_x=args.place_x,
                place_y=args.place_y,
                arm=args.arm,
                idle7=idle7,
                gripper_close=gripper_close,
                gripper_open=gripper_open,
                lift_offset_m=args.place_lift_offset_m,
                lift_duration_s=args.place_lift_duration_s,
                transport_duration_s=args.place_transport_duration_s,
                release_duration_s=args.place_release_duration_s,
                lower_before_release_m=actual_lower_before_release_m,
                lower_duration_s=args.place_lower_duration_s,
                retract_after_release_m=args.place_retract_after_release_m,
                retract_duration_s=args.place_retract_duration_s,
                append_retract=not merge_retract_home,
                lifted_xyzrpy=lifted_xyzrpy,
                transport_xyzrpy=transport_xyzrpy,
                skip_lift_transport=True,
            )
            grasp_close7 = xyzrpy_to_eef7(xyzrpy, gripper=gripper_close)
            left_lt, right_lt = build_lift_transport_waypoints(
                grasp_close7,
                lifted_xyzrpy,
                transport_xyzrpy,
                arm=args.arm,
                idle7=idle7,
                gripper_close=gripper_close,
            )
            retract_home_waypoints: tuple[list[list[float]], list[list[float]], list[float]] | None = None
            if merge_retract_home:
                retract_home_waypoints = build_retract_home_waypoints(
                    transport_xyzrpy,
                    arm=args.arm,
                    idle7=idle7,
                    home_left7=home_left7,
                    home_right7=home_right7,
                    gripper_open=gripper_open,
                    lower_before_release_m=actual_lower_before_release_m,
                    retract_after_release_m=args.place_retract_after_release_m,
                    retract_fraction=args.place_retract_home_fraction,
                    skip_release_waypoint=True,
                )
            elif home_after_place:
                place_phases.append(
                    build_home_phase(
                        home_left7=home_left7,
                        home_right7=home_right7,
                        duration_s=args.home_duration_s,
                    )
                )
            if args.once:
                for left7, right7, _ in phases:
                    pub.publish(left7, right7)
                pub.publish(left_lt[-1], right_lt[-1])
                for left7, right7, _ in place_phases:
                    pub.publish(left7, right7)
                if retract_home_waypoints is not None:
                    left_wps, right_wps, _ = retract_home_waypoints
                    pub.publish(left_wps[-1], right_wps[-1])
                return

            pub.publish_sequence(phases, phase_names=["grasp", "close"], **seq_common)
            lt_duration = float(args.place_lift_duration_s) + float(args.place_transport_duration_s)
            pub.publish_waypoints(
                left_lt,
                right_lt,
                duration_s=lt_duration,
                segment_weights=[0.4, 0.6],
                rate_hz=args.rate_hz,
            )
            if pose_tracker is not None:
                reached = pose_tracker.wait_reached(
                    transport_xyzrpy,
                    tol_pos_m=args.pose_tol_pos_m,
                    tol_rot_rad=args.pose_tol_rot_rad,
                    timeout_s=max(1.0, lt_duration * 0.6),
                    rate_hz=args.rate_hz,
                )
                if not reached and args.abort_on_pose_fail:
                    raise RuntimeError("lift+transport: arm did not reach place pose")

            place_names = []
            if actual_lower_before_release_m > 0.0:
                place_names.append("lower")
            place_names.append("release")
            if not merge_retract_home and args.place_retract_after_release_m > 0.0:
                place_names.append("retract")
            if home_after_place and not merge_retract_home:
                place_names.append("home")
            pub.publish_sequence(place_phases, phase_names=place_names, **seq_common)
            if retract_home_waypoints is not None:
                left_wps, right_wps, weights = retract_home_waypoints
                pub.publish_waypoints(
                    left_wps,
                    right_wps,
                    duration_s=args.home_duration_s,
                    segment_weights=weights,
                    rate_hz=args.rate_hz,
                )
            return
        if args.once:
            for left7, right7, _ in phases:
                pub.publish(left7, right7)
            return
        pub.publish_sequence(phases, phase_names=["grasp", "close"], **seq_common)
        return

    if args.once:
        pub.publish(left7, right7)
        return
    pub.publish_for(left7, right7, duration_s=args.duration_s, rate_hz=args.rate_hz)


if __name__ == "__main__":
    main()
