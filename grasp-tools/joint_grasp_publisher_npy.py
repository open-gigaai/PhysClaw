"""Pipeline entry: grasp+place from AnyGrasp eef_pose_xyzrpy.npy via joint control.

Called by run_multi_pipeline.sh / run_full_pipeline.sh after AnyGrasp.
Homing is handled separately by joint_publisher.py.

Planning lives in joint_grasp_plan.py; this file loads the npy, plans IK once,
then publishes JointState via joint_publisher.JointPublisher.

Example:
  conda activate physgnosis
  python joint_grasp_publisher_npy.py --arm left --npy eef_pose_xyzrpy.npy \\
      --place-x 0.25 --place-y -0.24
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from piper_pose_ik import build_ik_solver
from eef_publisher_npy import (
    _DEFAULT_HOME7,
    _DEFAULT_IDLE7,
    _parse_7floats,
    apply_approach_depth_offset,
    load_grasp_pose_npy,
    resolve_grasp_gripper_widths,
    add_place_ik_cli_args,
    place_ik_options_from_args,
)
from joint_grasp_plan import (
    JointGraspPlacePlan,
    JointPhasePlan,
    JointWaypointPlan,
    format_plan_report,
    plan_joint_grasp_place,
)
from joint_publisher import JointPublisher

_DEFAULT_XYZRPY_NPY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eef_pose_xyzrpy.npy",
)
# Piper homing pose in joint space (j0..j5, gripper); no IK needed.
_DEFAULT_HOME_JOINT7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class JointTracker:
    """Subscribe to puppet joint feedback for reach verification."""

    def __init__(self, arm: str) -> None:
        self._state: JointState | None = None
        topic = f"/puppet/joint_{arm}" if arm in ("left", "right") else "/puppet/joint_left"
        rospy.Subscriber(topic, JointState, self._on_state, queue_size=1)
        rospy.sleep(0.05)

    def _on_state(self, msg: JointState) -> None:
        self._state = msg

    def joint_error(self, target7: Sequence[float]) -> float:
        if self._state is None or len(self._state.position) < 6:
            return float("inf")
        cur = np.asarray(self._state.position[:6], dtype=np.float64)
        tgt = np.asarray(target7[:6], dtype=np.float64)
        return float(np.linalg.norm(cur - tgt))

    def wait_reached(
        self,
        target7: Sequence[float],
        *,
        tol_rad: float,
        timeout_s: float,
        rate_hz: int = 30,
    ) -> bool:
        rate = rospy.Rate(rate_hz)
        deadline = rospy.Time.now() + rospy.Duration(float(timeout_s))
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.joint_error(target7) <= tol_rad:
                return True
            rate.sleep()
        err = self.joint_error(target7)
        rospy.logwarn("Joint target not reached within %.1fs (err=%.3frad)", timeout_s, err)
        return False


def active_joint7(left7: Sequence[float], right7: Sequence[float], arm: str) -> list[float]:
    if arm == "right":
        return list(right7)
    return list(left7)


def publish_pre_grasp_joint_pose(
    pre_grasp_joint7: Sequence[float],
    *,
    arm: str,
    idle_joint7: Sequence[float],
    duration_s: float,
    pub: JointPublisher,
    joint_tracker: JointTracker | None,
    rate_hz: int,
    joint_tol_rad: float,
    abort_on_joint_fail: bool,
) -> None:
    """Move the active arm to a raised joint pose before approaching the grasp target."""
    pre_j = list(pre_grasp_joint7)
    idle_j = list(idle_joint7)
    if arm == "left":
        left_j, right_j = pre_j, idle_j
    elif arm == "right":
        left_j, right_j = idle_j, pre_j
    else:
        left_j, right_j = pre_j, pre_j

    pub.publish_for(left_j, right_j, duration_s=duration_s, rate_hz=rate_hz)
    if joint_tracker is None:
        return
    target = active_joint7(left_j, right_j, arm)
    reached = joint_tracker.wait_reached(
        target,
        tol_rad=joint_tol_rad,
        timeout_s=max(0.5, float(duration_s) * 0.5),
        rate_hz=rate_hz,
    )
    if not reached and abort_on_joint_fail:
        raise RuntimeError("pre_grasp: arm did not reach target joints")


def execute_joint_grasp_place_plan(
    plan: JointGraspPlacePlan,
    *,
    arm: str,
    pub: JointPublisher,
    joint_tracker: JointTracker | None,
    rate_hz: int,
    joint_tol_rad: float,
    abort_on_joint_fail: bool,
) -> None:
    """Publish a cached joint plan without re-solving IK."""

    def _publish_phase(phase: JointPhasePlan) -> None:
        pub.publish_for(phase.left_j, phase.right_j, duration_s=phase.duration_s, rate_hz=rate_hz)
        if joint_tracker is None:
            return
        target = active_joint7(phase.left_j, phase.right_j, arm)
        reached = joint_tracker.wait_reached(
            target,
            tol_rad=joint_tol_rad,
            timeout_s=max(0.5, float(phase.duration_s) * 0.5),
            rate_hz=rate_hz,
        )
        if not reached and abort_on_joint_fail:
            raise RuntimeError(f"{phase.name}: arm did not reach target joints")

    def _publish_waypoints(motion: JointWaypointPlan, *, wait_target_idx: int | None = None) -> None:
        pub.publish_waypoints(
            motion.left_wps,
            motion.right_wps,
            duration_s=motion.duration_s,
            segment_weights=motion.segment_weights,
            rate_hz=rate_hz,
        )
        if joint_tracker is None or wait_target_idx is None:
            return
        target = active_joint7(
            motion.left_wps[wait_target_idx],
            motion.right_wps[wait_target_idx],
            arm,
        )
        reached = joint_tracker.wait_reached(
            target,
            tol_rad=joint_tol_rad,
            timeout_s=max(1.0, float(motion.duration_s) * 0.6),
            rate_hz=rate_hz,
        )
        if not reached and abort_on_joint_fail:
            raise RuntimeError(f"{motion.name}: arm did not reach target joints")

    for phase in plan.grasp_phases:
        _publish_phase(phase)

    if plan.lift_transport is not None:
        _publish_waypoints(plan.lift_transport, wait_target_idx=-1)

    for phase in plan.place_phases:
        _publish_phase(phase)

    if plan.retract_home is not None:
        _publish_waypoints(plan.retract_home)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Grasp+place from eef_pose_xyzrpy.npy via Pinocchio IK and joint control",
    )
    parser.add_argument("--npy", type=str, default=_DEFAULT_XYZRPY_NPY)
    parser.add_argument("--gripper-open-margin", type=float, default=0.05)
    parser.add_argument("--gripper-close-scale", type=float, default=0.33)
    parser.add_argument("--gripper-close", type=float, default=None)
    parser.add_argument("--override-rpy", nargs=3, type=float, default=None, metavar=("roll", "pitch", "yaw"))
    parser.add_argument("--approach-depth-offset-m", type=float, default=0.0)
    parser.add_argument("--arm", choices=("left", "right", "both"), default="left")
    parser.add_argument(
        "--idle-joint",
        nargs=7,
        default=None,
        metavar=("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
        help="Joint7 for the non-commanded arm (default: all zeros)",
    )
    parser.add_argument("--grasp-duration-s", type=float, default=2.0)
    parser.add_argument("--close-duration-s", type=float, default=0.5)
    parser.add_argument(
        "--pre-grasp-joint",
        nargs=7,
        default=None,
        metavar=("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
        help="Move active arm to this joint pose before approaching grasp (avoids collisions)",
    )
    parser.add_argument("--pre-grasp-duration-s", type=float, default=1.0)
    parser.add_argument("--place-x", type=float, default=None)
    parser.add_argument("--place-y", type=float, default=None)
    parser.add_argument("--place-lift-offset-m", type=float, default=0.15)
    parser.add_argument("--place-lift-duration-s", type=float, default=0.5)
    parser.add_argument("--place-transport-duration-s", type=float, default=0.5)
    parser.add_argument("--place-lower-before-release-m", type=float, default=0.05)
    parser.add_argument("--place-lower-duration-s", type=float, default=0.4)
    parser.add_argument("--place-release-duration-s", type=float, default=0.5)
    parser.add_argument("--place-retract-after-release-m", type=float, default=0.08)
    parser.add_argument(
        "--place-retract-duration-s",
        type=float,
        default=1.0,
        help="Lift (+Z) duration after gripper release before return home",
    )
    parser.add_argument("--place-retract-home-fraction", type=float, default=0.2)
    parser.add_argument("--no-home-after-place", action="store_true")
    parser.add_argument(
        "--home-left-joint",
        nargs=7,
        default=None,
        metavar=("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
        help="Left arm homing Joint7 after place (default: all zeros)",
    )
    parser.add_argument(
        "--home-right-joint",
        nargs=7,
        default=None,
        metavar=("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
        help="Right arm homing Joint7 after place (default: all zeros)",
    )
    parser.add_argument("--home-duration-s", type=float, default=1.5)
    parser.add_argument("--rate-hz", type=int, default=30)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--check-ik", action="store_true", default=True)
    parser.add_argument("--no-check-ik", action="store_false", dest="check_ik")
    add_place_ik_cli_args(parser)
    parser.add_argument("--wait-joint", action="store_true", default=True)
    parser.add_argument("--no-wait-joint", action="store_false", dest="wait_joint")
    parser.add_argument(
        "--abort-on-joint-fail",
        action="store_true",
        default=False,
        help="Stop sequence if joint reach check fails (default: warn only, continue)",
    )
    parser.add_argument("--no-abort-on-joint-fail", action="store_false", dest="abort_on_joint_fail")
    parser.add_argument("--joint-tol-rad", type=float, default=0.08)
    args = parser.parse_args()

    pub = JointPublisher()
    arm_ik = build_ik_solver()
    joint_tracker = JointTracker(args.arm) if args.wait_joint else None

    do_place = args.place_x is not None or args.place_y is not None
    if do_place and (args.place_x is None or args.place_y is None):
        raise SystemExit("--place-x and --place-y must be given together")

    xyzrpy, grasp_width_m = load_grasp_pose_npy(args.npy)
    if args.override_rpy is not None:
        xyzrpy = np.concatenate([xyzrpy[:3], np.asarray(args.override_rpy, dtype=np.float64)])
    if args.approach_depth_offset_m != 0.0:
        xyzrpy = apply_approach_depth_offset(xyzrpy, offset_m=args.approach_depth_offset_m)
    from piper_pose_ik import canonicalize_flange_xyzrpy_camera_up, flange_camera_up_dot

    xyzrpy, cam_flipped = canonicalize_flange_xyzrpy_camera_up(xyzrpy)
    if cam_flipped and not args.quiet:
        print(f"  camera-up: flipped grasp 180° about approach (camera·up={flange_camera_up_dot(xyzrpy):.3f})")

    gripper_open, gripper_close, gripper_src = resolve_grasp_gripper_widths(
        grasp_width_m=grasp_width_m,
        margin_m=args.gripper_open_margin,
        close_scale=args.gripper_close_scale,
        close_override=args.gripper_close,
    )

    idle_eef7 = list(_DEFAULT_IDLE7)
    idle_joint7 = (
        list(_DEFAULT_HOME_JOINT7)
        if args.idle_joint is None
        else _parse_7floats(args.idle_joint, name="--idle-joint")
    )
    home_left7 = list(_DEFAULT_HOME7)
    home_right7 = list(_DEFAULT_HOME7)
    home_left_joint7 = (
        list(_DEFAULT_HOME_JOINT7)
        if args.home_left_joint is None
        else _parse_7floats(args.home_left_joint, name="--home-left-joint")
    )
    home_right_joint7 = (
        list(_DEFAULT_HOME_JOINT7)
        if args.home_right_joint is None
        else _parse_7floats(args.home_right_joint, name="--home-right-joint")
    )
    home_after_place = do_place and not args.no_home_after_place

    pre_grasp_joint7 = (
        None
        if args.pre_grasp_joint is None
        else _parse_7floats(args.pre_grasp_joint, name="--pre-grasp-joint")
    )

    def run_pre_grasp_if_configured() -> None:
        if pre_grasp_joint7 is None:
            return
        if not args.quiet:
            print(f"Pre-grasp joint ({args.arm}): {pre_grasp_joint7}")
        publish_pre_grasp_joint_pose(
            pre_grasp_joint7,
            arm=args.arm,
            idle_joint7=idle_joint7,
            duration_s=args.pre_grasp_duration_s,
            pub=pub,
            joint_tracker=joint_tracker,
            rate_hz=args.rate_hz,
            joint_tol_rad=args.joint_tol_rad,
            abort_on_joint_fail=args.abort_on_joint_fail,
        )

    plan = plan_joint_grasp_place(
        xyzrpy=xyzrpy,
        arm=args.arm,
        idle_eef7=idle_eef7,
        idle_joint7=idle_joint7,
        home_left7=home_left7,
        home_right7=home_right7,
        home_left_joint7=home_left_joint7,
        home_right_joint7=home_right_joint7,
        gripper_open=gripper_open,
        gripper_close=gripper_close,
        grasp_duration_s=args.grasp_duration_s,
        close_duration_s=args.close_duration_s,
        do_place=do_place,
        place_x=args.place_x,
        place_y=args.place_y,
        place_lift_offset_m=args.place_lift_offset_m,
        place_lift_duration_s=args.place_lift_duration_s,
        place_transport_duration_s=args.place_transport_duration_s,
        place_lower_before_release_m=args.place_lower_before_release_m,
        place_lower_duration_s=args.place_lower_duration_s,
        place_release_duration_s=args.place_release_duration_s,
        place_retract_after_release_m=args.place_retract_after_release_m,
        place_retract_duration_s=args.place_retract_duration_s,
        place_retract_home_fraction=args.place_retract_home_fraction,
        home_after_place=home_after_place,
        home_duration_s=args.home_duration_s,
        check_ik=args.check_ik,
        arm_ik=arm_ik,
        place_ik_options=place_ik_options_from_args(args),
    )
    print(
        format_plan_report(
            plan,
            arm=args.arm,
            place_x=args.place_x,
            place_y=args.place_y,
            grasp_width_m=grasp_width_m,
            quiet=args.quiet,
        )
    )
    if not args.quiet:
        print(f"Loaded {args.npy}")
        print(f"  grasp_width(m): {grasp_width_m:.4f}")
        print(f"  gripper open:   {gripper_open:.4f}  ({gripper_src} + {args.gripper_open_margin:.4f}m margin)")
        print(f"  gripper close:  {gripper_close:.4f}")

    run_pre_grasp_if_configured()
    execute_joint_grasp_place_plan(
        plan,
        arm=args.arm,
        pub=pub,
        joint_tracker=joint_tracker,
        rate_hz=args.rate_hz,
        joint_tol_rad=args.joint_tol_rad,
        abort_on_joint_fail=args.abort_on_joint_fail,
    )


if __name__ == "__main__":
    main()
