#!/usr/bin/env bash
# Shared path bootstrap for PhysGnosis shell scripts.
# Usage from grasp-tools/: source "$(dirname "$0")/../configs/load_paths.sh"
# Usage from skills/*/scripts/: source via _env.sh

_physgnosis_resolve_root() {
  local start cand
  start="${_PHYSGNOSIS_CALLER:-${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}}"
  start="$(cd "$(dirname "$start")" && pwd)"
  cand="$start"
  while [[ "$cand" != "/" ]]; do
    if [[ -d "$cand/skills" && -d "$cand/grasp-tools" && -d "$cand/configs" ]]; then
      PHYSGNOSIS_ROOT="$cand"
      return 0
    fi
    cand="$(dirname "$cand")"
  done
  if [[ -d "$start/../skills" && -d "$start/../configs" ]]; then
    PHYSGNOSIS_ROOT="$(cd "$start/.." && pwd)"
    return 0
  fi
  echo "Warning: could not locate PhysGnosis root from $start" >&2
  PHYSGNOSIS_ROOT="$(cd "$start/.." && pwd)"
}

_physgnosis_load_paths() {
  local conf
  if [[ -z "${PHYSGNOSIS_ROOT:-}" || ! -d "${PHYSGNOSIS_ROOT}/grasp-tools" ]]; then
    _physgnosis_resolve_root
  fi
  export PHYSGNOSIS_ROOT

  conf="${PHYSGNOSIS_ROOT}/configs/paths.env"
  if [[ -f "$conf" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$conf"
    set +a
  fi

  # One project env for capture / motion / collect tooling.
  : "${CONDA_SH:=${HOME}/miniconda3/etc/profile.d/conda.sh}"
  : "${CONDA_ENV:=physgnosis}"

  # Official third-party envs (set to whatever names your AnyGrasp / SAM3 installs use).
  : "${CONDA_ENV_SAM3:=}"
  : "${CONDA_ENV_ANYGRASP:=}"

  : "${ROS_SETUP:=/opt/ros/noetic/setup.bash}"
  # Arm + camera ROS overlays / launches — fill for your robot (reference: Piper + RealSense).
  : "${ARM_ROS_SETUP:=${PIPER_ROS_SETUP:-}}"
  : "${CAMERA_ROS_SETUP:=${CAMERA_WS_SETUP:-}}"
  : "${CAMERA_LAUNCH:=${REALSENSE_LAUNCH:-}}"

  # IK / URDF for your arm (reference tree used Piper Pinocchio scripts).
  : "${ARM_ROS_SRC:=${PIPER_ROS_SRC:-}}"
  : "${ARM_IK_DIR:=${PIPER_PINOCCHIO_DIR:-}}"
  : "${ARM_URDF:=${PIPER_URDF:-}}"

  : "${ARM_LAUNCH:=}"
  : "${ARM_PRESTART_SH:=}"

  # ROS topics (override for your camera / arm drivers)
  : "${TOPIC_FRONT_COLOR:=/camera_f/color/image_raw}"
  : "${TOPIC_FRONT_DEPTH:=/camera_f/depth/image_raw}"
  : "${TOPIC_LEFT_COLOR:=/camera_l/color/image_raw}"
  : "${TOPIC_RIGHT_COLOR:=/camera_r/color/image_raw}"
  : "${TOPIC_JOINT_LEFT:=/puppet/joint_left}"
  : "${TOPIC_JOINT_RIGHT:=/puppet/joint_right}"

  # Front-camera intrinsics for grasp (pixels). Fill with your calibration.
  : "${CAM_FX:=}"
  : "${CAM_FY:=}"
  : "${CAM_CX:=}"
  : "${CAM_CY:=}"
  : "${DEPTH_SCALE:=1000.0}"

  # Optional path to JSON with cam2base extrinsics (see configs/camera_extrinsics.example.json)
  : "${CAMERA_EXTRINSICS_JSON:=}"

  : "${UNDERSTAND_SH:=${PHYSGNOSIS_ROOT}/skills/understand-three-view-images/scripts/auto_understand_images.sh}"
  : "${ANALYZE_YAML:=${PHYSGNOSIS_ROOT}/skills/self-learning/analyze_result.yaml}"
  : "${GRASP_TOOLS_DIR:=${PHYSGNOSIS_ROOT}/grasp-tools}"

  # Legacy aliases used by older scripts
  CONDA_ENV_CAPTURE="${CONDA_ENV}"
  CONDA_ENV_IK="${CONDA_ENV}"
  PIPER_ROS_SETUP="${ARM_ROS_SETUP}"
  CAMERA_WS_SETUP="${CAMERA_ROS_SETUP}"
  REALSENSE_LAUNCH="${CAMERA_LAUNCH}"
  PIPER_ROS_SRC="${ARM_ROS_SRC}"
  PIPER_PINOCCHIO_DIR="${ARM_IK_DIR}"
  PIPER_URDF="${ARM_URDF}"

  : "${ARM_LAUNCH:=}"
  : "${ARM_PRESTART_SH:=}"
  export ARM_LAUNCH ARM_PRESTART_SH
  export PIPER_ROS_SETUP CAMERA_WS_SETUP REALSENSE_LAUNCH PIPER_ROS_SRC PIPER_PINOCCHIO_DIR PIPER_URDF
  export ARM_LAUNCH ARM_PRESTART_SH
  export COLLECT_RECORDER_PY SAM3_CHECKPOINT
  export TOPIC_FRONT_COLOR TOPIC_FRONT_DEPTH TOPIC_LEFT_COLOR TOPIC_RIGHT_COLOR
  export TOPIC_JOINT_LEFT TOPIC_JOINT_RIGHT
  export CAM_FX CAM_FY CAM_CX CAM_CY DEPTH_SCALE CAMERA_EXTRINSICS_JSON
  export UNDERSTAND_SH ANALYZE_YAML GRASP_TOOLS_DIR
  export GSNET_DIR ANYGRASP_CHECKPOINT
}

_physgnosis_load_paths
