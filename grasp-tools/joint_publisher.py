"""Minimal joint-state ROS publisher for dual arms.

Standalone CLI for homing and manual joint moves (used by the recorder pipeline do_home).
Grasp+place pipelines use joint_grasp_publisher_npy.py instead.

Publishes Joint7 = [j0..j5, gripper] in radians to /master/joint_{left,right}
(reference topic naming; override feedback topics via TOPIC_JOINT_LEFT/RIGHT).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

TOPIC_JOINT_LEFT = os.environ.get("TOPIC_JOINT_LEFT", "/puppet/joint_left")
TOPIC_JOINT_RIGHT = os.environ.get("TOPIC_JOINT_RIGHT", "/puppet/joint_right")

MIN_QPOS = [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944]
MAX_QPOS = [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944]


@dataclass(frozen=True)
class Joint7:
    j0: float
    j1: float
    j2: float
    j3: float
    j4: float
    j5: float
    gripper: float

    @staticmethod
    def from_seq(seq: Sequence[float], *, name: str = "joint") -> "Joint7":
        if len(seq) != 7:
            raise ValueError(f"{name} must have 7 floats: [j0..j5, gripper], got len={len(seq)}")
        return Joint7(*(float(x) for x in seq))

    def as_list(self) -> list[float]:
        return [self.j0, self.j1, self.j2, self.j3, self.j4, self.j5, self.gripper]


def clip_arm_joints(joints: Sequence[float]) -> list[float]:
    out = list(joints)
    for i in range(6):
        out[i] = max(MIN_QPOS[i], min(MAX_QPOS[i], out[i]))
    return out


class _PuppetJointFeed:
    """Read current arm joints from TOPIC_JOINT_LEFT/RIGHT for motion interpolation."""

    def __init__(self) -> None:
        self._left: JointState | None = None
        self._right: JointState | None = None
        rospy.Subscriber(TOPIC_JOINT_LEFT, JointState, self._on_left, queue_size=1)
        rospy.Subscriber(TOPIC_JOINT_RIGHT, JointState, self._on_right, queue_size=1)
        rospy.sleep(0.05)

    def _on_left(self, msg: JointState) -> None:
        self._left = msg

    def _on_right(self, msg: JointState) -> None:
        self._right = msg

    def left7(self, fallback: Sequence[float]) -> list[float]:
        if self._left is not None and len(self._left.position) >= 7:
            return [float(x) for x in self._left.position[:7]]
        return list(fallback)

    def right7(self, fallback: Sequence[float]) -> list[float]:
        if self._right is not None and len(self._right.position) >= 7:
            return [float(x) for x in self._right.position[:7]]
        return list(fallback)


class JointPublisher:
    """Publish joint commands as `sensor_msgs/JointState`.

    API:
      pub = JointPublisher()
      pub.publish(left7, right7)
    """

    def __init__(
        self,
        *,
        node_name: str = "joint_publisher",
        left_topic: str = "/master/joint_left",
        right_topic: str = "/master/joint_right",
        queue_size: int = 10,
        anonymous: bool = True,
        disable_signals: bool = False,
        clip_joints: bool = True,
    ) -> None:
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=anonymous, disable_signals=disable_signals)
        self._left_pub = rospy.Publisher(left_topic, JointState, queue_size=queue_size)
        self._right_pub = rospy.Publisher(right_topic, JointState, queue_size=queue_size)
        self._clip_joints = clip_joints
        self._puppet_feed: _PuppetJointFeed | None = None
        self._last_left7: list[float] | None = None
        self._last_right7: list[float] | None = None

    def _ensure_puppet_feed(self) -> _PuppetJointFeed:
        if self._puppet_feed is None:
            self._puppet_feed = _PuppetJointFeed()
        return self._puppet_feed

    def _remember_cmd(self, left7: Sequence[float], right7: Sequence[float]) -> None:
        self._last_left7 = list(left7)
        self._last_right7 = list(right7)

    def _resolve_start_joints(
        self,
        left_end: Sequence[float],
        right_end: Sequence[float],
        *,
        from_left7: Sequence[float] | None,
        from_right7: Sequence[float] | None,
    ) -> tuple[list[float], list[float]]:
        if from_left7 is not None and from_right7 is not None:
            return list(from_left7), list(from_right7)
        if self._last_left7 is not None and self._last_right7 is not None:
            return list(self._last_left7), list(self._last_right7)
        feed = self._ensure_puppet_feed()
        left_start = feed.left7(left_end)
        right_start = feed.right7(right_end)
        if left_start == list(left_end) and right_start == list(right_end):
            rospy.logwarn_throttle(
                5.0,
                "No joint feedback yet; motion may be instant. Are TOPIC_JOINT_LEFT/RIGHT publishing?",
            )
        return left_start, right_start

    @staticmethod
    def _to_msg(joints: Joint7) -> JointState:
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = list(JOINT_NAMES)
        msg.position = joints.as_list()
        return msg

    def _prepare(self, joints7: Sequence[float]) -> Joint7:
        joints = joints7 if isinstance(joints7, Joint7) else Joint7.from_seq(joints7)
        if not self._clip_joints:
            return joints
        return Joint7.from_seq(clip_arm_joints(joints.as_list()))

    def publish(self, left7: Sequence[float], right7: Sequence[float]) -> None:
        left = self._prepare(left7)
        right = self._prepare(right7)

        if self._left_pub.get_num_connections() == 0 or self._right_pub.get_num_connections() == 0:
            rospy.logwarn_throttle(
                5.0,
                "No subscriber on JointState topics (left=%d right=%d). Is the arm driver running?",
                self._left_pub.get_num_connections(),
                self._right_pub.get_num_connections(),
            )

        self._left_pub.publish(self._to_msg(left))
        self._right_pub.publish(self._to_msg(right))
        self._remember_cmd(left.as_list(), right.as_list())

    def publish_for(
        self,
        left7: Sequence[float],
        right7: Sequence[float],
        *,
        duration_s: float = 2.0,
        rate_hz: int = 30,
        from_left7: Sequence[float] | None = None,
        from_right7: Sequence[float] | None = None,
        interpolate: bool = True,
    ) -> None:
        """Publish joint commands over duration_s.

        When interpolate=True (default), linearly ramp from start to target joints
        at rate_hz so duration_s controls actual motion time. Start joints come
        from from_* if given, else the last published command, else /puppet/joint_*.
        """
        left_end = self._prepare(left7).as_list()
        right_end = self._prepare(right7).as_list()
        if interpolate:
            left_start, right_start = self._resolve_start_joints(
                left_end,
                right_end,
                from_left7=from_left7,
                from_right7=from_right7,
            )
        else:
            left_start, right_start = left_end, right_end

        rate = rospy.Rate(rate_hz)
        n = max(1, int(round(float(duration_s) * float(rate_hz))))
        for i in range(n):
            if rospy.is_shutdown():
                return
            t = i / max(n - 1, 1)
            if interpolate:
                left_cmd = _lerp7(left_start, left_end, t)
                right_cmd = _lerp7(right_start, right_end, t)
            else:
                left_cmd, right_cmd = left_end, right_end
            self.publish(left_cmd, right_cmd)
            rate.sleep()

    def publish_sequence(
        self,
        phases: Sequence[tuple[Sequence[float], Sequence[float], float]],
        *,
        rate_hz: int = 30,
    ) -> None:
        for left7, right7, duration_s in phases:
            if rospy.is_shutdown():
                return
            self.publish_for(left7, right7, duration_s=duration_s, rate_hz=rate_hz)

    def publish_waypoints(
        self,
        left_waypoints: Sequence[Sequence[float]],
        right_waypoints: Sequence[Sequence[float]],
        *,
        duration_s: float,
        segment_weights: Sequence[float] | None = None,
        rate_hz: int = 30,
    ) -> None:
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Publish left/right Joint7 (j0..j5 gripper) via ROS JointState"
    )
    parser.add_argument("--left", nargs=7, required=True, metavar=("j0", "j1", "j2", "j3", "j4", "j5", "gripper"))
    parser.add_argument("--right", nargs=7, required=True, metavar=("j0", "j1", "j2", "j3", "j4", "j5", "gripper"))
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Publish once and exit")
    parser.add_argument("--left-topic", type=str, default="/master/joint_left")
    parser.add_argument("--right-topic", type=str, default="/master/joint_right")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip arm joints to Piper limits")
    args = parser.parse_args()

    left7 = _parse_7floats(args.left, name="--left")
    right7 = _parse_7floats(args.right, name="--right")

    pub = JointPublisher(
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        clip_joints=not args.no_clip,
    )
    if args.once:
        pub.publish(left7, right7)
        return
    pub.publish_for(left7, right7, duration_s=args.duration_s, rate_hz=args.rate_hz)


if __name__ == "__main__":
    main()
'''
python3 joint_publisher.py \
  --left  -0.15069  1.20674 -0.93398  0.19989  0.64097  0 0.0 \
  --right -0.12318  1.1905  -0.90064  0.3659   0.56606  0 0.06 \
  --duration-s 3 --rate-hz 30

python3 joint_publisher.py \
    --left 0.0 0 0 0 0.0 0.0 0.0\
    --right 0.0 0 0 0 0.0 0 0.0 \
    --duration-s 2 --rate-hz 30

0.0 1.1 -0.9 0 0.9 0.0 0.0 \

--left  0.0 0.5 -0.5 -0.03 0.0 0.0 0.0 \
--right 0.0 0.5 -0.5 -0.03 0.0 0.0 0.06 \
'''