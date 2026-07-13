#!/usr/bin/env bash
# Reference helper to bring up cameras + arms.
# This is NOT Piper/RealSense-specific: set ARM_ROS_SETUP, CAMERA_ROS_SETUP,
# CAMERA_LAUNCH, and ARM_LAUNCH in configs/paths.env for your stack.
# The authors' lab used Agilex Piper + Intel RealSense as one reference.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PHYSGNOSIS_CALLER="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../../configs/load_paths.sh"

TMP="${TMPDIR:-/tmp}/physgnosis"
CONDA_CLIENT="${CONDA_ENV:-physgnosis}"

log_info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
log_ok()   { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
log_warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }

log_info "[1/4] Stopping old ROS/robot processes..."
bash "${KILL_ROS_SH:-$HOME/kill_ros.sh}" 2>/dev/null || true
pkill -f roslaunch 2>/dev/null || true
pkill -f rosmaster 2>/dev/null || true
sleep 3

log_info "[2/4] Optional site bring-up (CAN / drivers)..."
if [[ -n "${ARM_PRESTART_SH:-}" && -f "${ARM_PRESTART_SH}" ]]; then
    bash "${ARM_PRESTART_SH}"
else
    log_warn "ARM_PRESTART_SH unset; skip site-specific prestart."
fi

log_info "[3/4] Starting cameras..."
source "${ROS_SETUP}"
if [[ -n "${CAMERA_ROS_SETUP:-}" && -f "${CAMERA_ROS_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${CAMERA_ROS_SETUP}"
fi
if [[ -z "${CAMERA_LAUNCH:-}" ]]; then
    log_warn "CAMERA_LAUNCH unset; skip cameras."
else
    # shellcheck disable=SC2086
    nohup roslaunch ${CAMERA_LAUNCH} \
        < /dev/null > "${TMP}_ros-camera.log" 2>&1 &
    echo $! > "${TMP}_ros-camera.pid"
    sleep 8
fi

log_info "[4/4] Starting arms..."
source "${ROS_SETUP}"
if [[ -n "${ARM_ROS_SETUP:-}" && -f "${ARM_ROS_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${ARM_ROS_SETUP}"
fi
source "${CONDA_SH}"
conda activate "${CONDA_CLIENT}"
if [[ -z "${ARM_LAUNCH:-}" ]]; then
    log_warn "ARM_LAUNCH unset; skip arms."
else
    # shellcheck disable=SC2086
    nohup roslaunch ${ARM_LAUNCH} \
        < /dev/null > "${TMP}_ros-arm.log" 2>&1 &
    echo $! > "${TMP}_ros-arm.pid"
    sleep 12
fi

TOPICS=$(source "${ROS_SETUP}" && timeout 8 rostopic list 2>/dev/null || true)
echo "$TOPICS" | grep -qF "${TOPIC_FRONT_COLOR}" && log_ok "  front camera topic: OK" || log_warn "  front camera topic not ready: ${TOPIC_FRONT_COLOR}"
echo "$TOPICS" | grep -qF "${TOPIC_JOINT_LEFT}" && log_ok "  left joint topic: OK" || log_warn "  left joint topic not ready: ${TOPIC_JOINT_LEFT}"
echo "  Camera log:  tail -f ${TMP}_ros-camera.log"
echo "  Arm log:     tail -f ${TMP}_ros-arm.log"
log_ok "Robot infra start attempted (configure launches in paths.env)."
