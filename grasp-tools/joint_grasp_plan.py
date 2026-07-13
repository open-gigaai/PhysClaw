"""Plan joint-space grasp+place trajectories (IK solved once, cached for execution).

Library module used by joint_grasp_publisher_npy.py — no CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from eef_publisher_npy import (
    build_lift_transport_waypoints,
    build_place_phases,
    build_retract_home_waypoints,
    lift_xyzrpy_from_grasp,
    resolve_place_poses_with_ik,
    xyzrpy_to_eef7,
)
from piper_pose_ik import PlacePosePlan, solve_xyzrpy_ik


@dataclass
class JointPhasePlan:
    name: str
    left_j: list[float]
    right_j: list[float]
    duration_s: float
    eef_xyzrpy: np.ndarray | None = None


@dataclass
class JointWaypointPlan:
    name: str
    left_wps: list[list[float]]
    right_wps: list[list[float]]
    duration_s: float
    segment_weights: list[float]
    waypoint_names: list[str] = field(default_factory=list)


@dataclass
class JointGraspPlacePlan:
  grasp_phases: list[JointPhasePlan]
  lift_transport: JointWaypointPlan | None = None
  place_phases: list[JointPhasePlan] = field(default_factory=list)
  retract_home: JointWaypointPlan | None = None
  compromises: list[str] = field(default_factory=list)
  path_summary: list[str] = field(default_factory=list)
  grasp_xyzrpy: np.ndarray | None = None
  place_pose_plan: PlacePosePlan | None = None


def _assign_arm_joints(
    active_j7: Sequence[float],
    idle_j7: Sequence[float],
    arm: str,
) -> tuple[list[float], list[float]]:
    active = list(active_j7)
    idle = list(idle_j7)
    if arm == "left":
        return active, idle
    if arm == "right":
        return idle, active
    return active, active


def _solve_eef7_or_raise(
    eef7: Sequence[float],
    *,
    arm_ik,
    phase_name: str,
) -> list[float]:
    joints7, ok = solve_xyzrpy_ik(
        np.asarray(eef7[:6], dtype=np.float64),
        gripper=float(eef7[6]),
        arm_ik=arm_ik,
    )
    if not ok or joints7 is None:
        raise RuntimeError(f"{phase_name}: Pinocchio IK failed for xyzrpy={list(eef7[:6])}")
    return joints7.tolist()


def _eef_phase_to_joint_phase(
    left_eef7: Sequence[float],
    right_eef7: Sequence[float],
    *,
    arm: str,
    arm_ik,
    idle_joint7: Sequence[float],
    phase_name: str,
    duration_s: float,
    cached_active_j7: Sequence[float] | None = None,
    eef_xyzrpy: np.ndarray | None = None,
) -> JointPhasePlan:
    if cached_active_j7 is not None:
        active_j = list(cached_active_j7)
    else:
        if arm == "left":
            active_j = _solve_eef7_or_raise(left_eef7, arm_ik=arm_ik, phase_name=f"{phase_name}/left")
        elif arm == "right":
            active_j = _solve_eef7_or_raise(right_eef7, arm_ik=arm_ik, phase_name=f"{phase_name}/right")
        else:
            _solve_eef7_or_raise(left_eef7, arm_ik=arm_ik, phase_name=f"{phase_name}/left")
            active_j = _solve_eef7_or_raise(right_eef7, arm_ik=arm_ik, phase_name=f"{phase_name}/right")
    left_j, right_j = _assign_arm_joints(active_j, idle_joint7, arm)
    return JointPhasePlan(
        name=phase_name,
        left_j=left_j,
        right_j=right_j,
        duration_s=float(duration_s),
        eef_xyzrpy=None if eef_xyzrpy is None else np.asarray(eef_xyzrpy, dtype=np.float64).reshape(6),
    )


def _home_joint_phase(
    *,
    home_left_joint7: Sequence[float],
    home_right_joint7: Sequence[float],
    duration_s: float,
) -> JointPhasePlan:
    return JointPhasePlan(
        name="home",
        left_j=list(home_left_joint7),
        right_j=list(home_right_joint7),
        duration_s=float(duration_s),
    )


def plan_joint_grasp_place(
    *,
    xyzrpy: np.ndarray,
    arm: str,
    idle_eef7: Sequence[float],
    idle_joint7: Sequence[float],
    home_left7: Sequence[float],
    home_right7: Sequence[float],
    home_left_joint7: Sequence[float],
    home_right_joint7: Sequence[float],
    gripper_open: float,
    gripper_close: float,
    grasp_duration_s: float,
    close_duration_s: float,
    do_place: bool,
    place_x: float | None,
    place_y: float | None,
    place_lift_offset_m: float,
    place_lift_duration_s: float,
    place_transport_duration_s: float,
    place_lower_before_release_m: float,
    place_lower_duration_s: float,
    place_release_duration_s: float,
    place_retract_after_release_m: float,
    place_retract_duration_s: float,
    place_retract_home_fraction: float,
    home_after_place: bool,
    home_duration_s: float,
    check_ik: bool,
    arm_ik,
    place_ik_options,
) -> JointGraspPlacePlan:
    """Solve IK for the full sequence once; raise before any motion if any phase fails."""
    grasp_open7 = xyzrpy_to_eef7(xyzrpy, gripper=gripper_open)
    grasp_close7 = xyzrpy_to_eef7(xyzrpy, gripper=gripper_close)
    grasp_open_j = _solve_eef7_or_raise(grasp_open7, arm_ik=arm_ik, phase_name="grasp")
    grasp_close_j = _solve_eef7_or_raise(grasp_close7, arm_ik=arm_ik, phase_name="close")

    left_grasp_j, right_grasp_j = _assign_arm_joints(grasp_open_j, idle_joint7, arm)
    left_close_j, right_close_j = _assign_arm_joints(grasp_close_j, idle_joint7, arm)
    grasp_phases = [
        JointPhasePlan(
            name="grasp",
            left_j=left_grasp_j,
            right_j=right_grasp_j,
            duration_s=grasp_duration_s,
            eef_xyzrpy=np.asarray(xyzrpy, dtype=np.float64).reshape(6),
        ),
        JointPhasePlan(
            name="close",
            left_j=left_close_j,
            right_j=right_close_j,
            duration_s=close_duration_s,
            eef_xyzrpy=np.asarray(xyzrpy, dtype=np.float64).reshape(6),
        ),
    ]

    compromises: list[str] = []
    path_summary = [
        f"grasp @ ({xyzrpy[0]:.3f}, {xyzrpy[1]:.3f}, {xyzrpy[2]:.3f})",
        "close gripper",
    ]
    place_pose_plan: PlacePosePlan | None = None
    lift_transport: JointWaypointPlan | None = None
    place_phases: list[JointPhasePlan] = []
    retract_home: JointWaypointPlan | None = None

    if not do_place:
        return JointGraspPlacePlan(
            grasp_phases=grasp_phases,
            compromises=compromises,
            path_summary=path_summary,
            grasp_xyzrpy=xyzrpy,
        )

    assert place_x is not None and place_y is not None
    place_pose_plan = resolve_place_poses_with_ik(
        xyzrpy,
        place_x=float(place_x),
        place_y=float(place_y),
        lift_offset_m=place_lift_offset_m,
        lower_before_release_m=place_lower_before_release_m,
        gripper_close=gripper_close,
        carry_rpy_ref=idle_eef7[3:6],
        check_ik=check_ik,
        arm_ik=arm_ik,
        place_ik_options=place_ik_options,
    )
    compromises.extend(place_pose_plan.compromises)

    lifted_xyzrpy = place_pose_plan.lifted_xyzrpy
    transport_xyzrpy = place_pose_plan.transport_xyzrpy
    actual_lower_m = place_pose_plan.actual_lower_m

    if check_ik and place_pose_plan.joint_lift is not None:
        joint_lift = place_pose_plan.joint_lift
        joint_transport = place_pose_plan.joint_transport
        joint_lower = place_pose_plan.joint_lower
    else:
        lift7 = xyzrpy_to_eef7(lifted_xyzrpy, gripper=gripper_close)
        transport7 = xyzrpy_to_eef7(transport_xyzrpy, gripper=gripper_close)
        lower7 = xyzrpy_to_eef7(place_pose_plan.lower_xyzrpy, gripper=gripper_close)
        joint_lift = np.asarray(
            _solve_eef7_or_raise(lift7, arm_ik=arm_ik, phase_name="lift"),
            dtype=np.float64,
        )
        joint_transport = np.asarray(
            _solve_eef7_or_raise(transport7, arm_ik=arm_ik, phase_name="transport"),
            dtype=np.float64,
        )
        joint_lower = np.asarray(
            _solve_eef7_or_raise(lower7, arm_ik=arm_ik, phase_name="lower"),
            dtype=np.float64,
        )

    left_lt, right_lt = build_lift_transport_waypoints(
        grasp_close7,
        lifted_xyzrpy,
        transport_xyzrpy,
        arm=arm,
        idle7=idle_eef7,
        gripper_close=gripper_close,
    )
    lt_names = ["close", "lift", "transport"]
    lt_left: list[list[float]] = []
    lt_right: list[list[float]] = []
    lt_cached = [grasp_close_j, joint_lift.tolist(), joint_transport.tolist()]
    for i, (left_eef, right_eef) in enumerate(zip(left_lt, right_lt)):
        left_j, right_j = _assign_arm_joints(lt_cached[i], idle_joint7, arm)
        lt_left.append(left_j)
        lt_right.append(right_j)

    lift_transport = JointWaypointPlan(
        name="lift_transport",
        left_wps=lt_left,
        right_wps=lt_right,
        duration_s=float(place_lift_duration_s) + float(place_transport_duration_s),
        segment_weights=[0.4, 0.6],
        waypoint_names=lt_names,
    )
    path_summary.extend(
        [
            f"lift +{place_pose_plan.actual_lift_m:.3f}m -> z={float(lifted_xyzrpy[2]):.3f}",
            (
                f"transport -> ({float(transport_xyzrpy[0]):.3f}, "
                f"{float(transport_xyzrpy[1]):.3f}, z={float(transport_xyzrpy[2]):.3f})"
            ),
        ]
    )

    merge_retract_home = home_after_place and place_retract_after_release_m > 0.0
    do_retract = place_retract_after_release_m > 0.0
    place_phases_eef = build_place_phases(
        xyzrpy,
        place_x=place_x,
        place_y=place_y,
        arm=arm,
        idle7=idle_eef7,
        gripper_close=gripper_close,
        gripper_open=gripper_open,
        lift_offset_m=place_lift_offset_m,
        lift_duration_s=place_lift_duration_s,
        transport_duration_s=place_transport_duration_s,
        release_duration_s=place_release_duration_s,
        lower_before_release_m=actual_lower_m,
        lower_duration_s=place_lower_duration_s,
        retract_after_release_m=place_retract_after_release_m,
        retract_duration_s=place_retract_duration_s,
        append_retract=do_retract,
        lifted_xyzrpy=lifted_xyzrpy,
        transport_xyzrpy=transport_xyzrpy,
        skip_lift_transport=True,
    )

    place_names: list[str] = []
    if actual_lower_m > 0.0:
        place_names.append("lower")
    place_names.append("release")
    if do_retract:
        place_names.append("retract")
    if home_after_place and not merge_retract_home:
        place_names.append("home")

    release7 = xyzrpy_to_eef7(place_pose_plan.lower_xyzrpy, gripper=gripper_open)
    release_j = _solve_eef7_or_raise(release7, arm_ik=arm_ik, phase_name="release")

    retract_j: list[float] | None = None
    retract_xyzrpy: np.ndarray | None = None
    if do_retract:
        retract_xyzrpy = lift_xyzrpy_from_grasp(
            place_pose_plan.lower_xyzrpy,
            offset_m=place_retract_after_release_m,
        )
        retract7 = xyzrpy_to_eef7(retract_xyzrpy, gripper=gripper_open)
        retract_j = _solve_eef7_or_raise(retract7, arm_ik=arm_ik, phase_name="retract")

    for i, (left_eef, right_eef, duration_s) in enumerate(place_phases_eef):
        name = place_names[i] if i < len(place_names) else f"place_{i}"
        if name == "home":
            place_phases.append(
                _home_joint_phase(
                    home_left_joint7=home_left_joint7,
                    home_right_joint7=home_right_joint7,
                    duration_s=duration_s,
                )
            )
            path_summary.append("return home (joint)")
            continue

        cached = None
        eef_xyz = None
        if name == "lower":
            cached = joint_lower.tolist()
            eef_xyz = place_pose_plan.lower_xyzrpy
            path_summary.append(
                f"lower -{actual_lower_m:.3f}m -> z={float(place_pose_plan.lower_xyzrpy[2]):.3f}"
            )
        elif name == "release":
            cached = release_j
            eef_xyz = place_pose_plan.lower_xyzrpy
            path_summary.append("release gripper")
        elif name == "retract":
            cached = retract_j
            eef_xyz = retract_xyzrpy
            path_summary.append(
                f"retract +{place_retract_after_release_m:.3f}m "
                f"({place_retract_duration_s:.1f}s) -> z={float(retract_xyzrpy[2]):.3f}"
            )

        place_phases.append(
            _eef_phase_to_joint_phase(
                left_eef,
                right_eef,
                arm=arm,
                arm_ik=arm_ik,
                idle_joint7=idle_joint7,
                phase_name=name,
                duration_s=duration_s,
                cached_active_j7=cached,
                eef_xyzrpy=eef_xyz,
            )
        )

    if merge_retract_home:
        retract_home_eef = build_retract_home_waypoints(
            transport_xyzrpy,
            arm=arm,
            idle7=idle_eef7,
            home_left7=home_left7,
            home_right7=home_right7,
            gripper_open=gripper_open,
            lower_before_release_m=actual_lower_m,
            retract_after_release_m=place_retract_after_release_m,
            retract_fraction=place_retract_home_fraction,
            skip_release_waypoint=True,
        )
        left_wps, right_wps, weights = retract_home_eef
        rh_left: list[list[float]] = []
        rh_right: list[list[float]] = []
        rh_names: list[str] = []
        assert retract_j is not None
        for i, (left_eef, right_eef) in enumerate(zip(left_wps, right_wps)):
            if i == len(left_wps) - 1:
                left_j, right_j = list(home_left_joint7), list(home_right_joint7)
                rh_names.append("home")
            else:
                left_j, right_j = _assign_arm_joints(retract_j, idle_joint7, arm)
                rh_names.append("retract")
            rh_left.append(left_j)
            rh_right.append(right_j)

        retract_home = JointWaypointPlan(
            name="retract_home",
            left_wps=rh_left,
            right_wps=rh_right,
            duration_s=float(home_duration_s),
            segment_weights=list(weights),
            waypoint_names=rh_names,
        )
        path_summary.append(f"return home ({home_duration_s:.1f}s)")

    return JointGraspPlacePlan(
        grasp_phases=grasp_phases,
        lift_transport=lift_transport,
        place_phases=place_phases,
        retract_home=retract_home,
        compromises=compromises,
        path_summary=path_summary,
        grasp_xyzrpy=xyzrpy,
        place_pose_plan=place_pose_plan,
    )


def format_plan_report(
    plan: JointGraspPlacePlan,
    *,
    arm: str,
    place_x: float | None = None,
    place_y: float | None = None,
    grasp_width_m: float | None = None,
    quiet: bool = False,
) -> str:
    lines: list[str] = []
    if quiet:
        if plan.place_pose_plan is not None and place_x is not None and place_y is not None:
            tz = float(plan.place_pose_plan.transport_xyzrpy[2])
            lift_note = ""
            if abs(plan.place_pose_plan.actual_lift_m) > 1e-9:
                lift_note = f"  lift={plan.place_pose_plan.actual_lift_m:.2f}m"
            width_note = f"  width={grasp_width_m:.4f}m" if grasp_width_m is not None else ""
            lines.append(
                f"Execute ({arm}, joint): grasp -> place TCP ({place_x:.2f}, {place_y:.2f}, "
                f"transport_z={tz:.2f}){width_note}{lift_note}"
            )
        elif grasp_width_m is not None:
            lines.append(f"Execute ({arm}, joint): grasp  width={grasp_width_m:.4f}m")
        else:
            lines.append(f"Execute ({arm}, joint): reach")
        if plan.compromises:
            lines.append("  IK compromises: " + "; ".join(plan.compromises))
        return "\n".join(lines)

    lines.append("Joint trajectory plan (IK solved once, cached for execution):")
    lines.append("  Planned path:")
    for step in plan.path_summary:
        lines.append(f"    - {step}")
    if plan.compromises:
        lines.append("  IK compromises (requested vs actual):")
        for note in plan.compromises:
            lines.append(f"    * {note}")
    else:
        lines.append("  IK compromises: none (all poses at requested values)")
    return "\n".join(lines)
