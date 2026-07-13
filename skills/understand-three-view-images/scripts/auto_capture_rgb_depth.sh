#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

source "${ROS_SETUP}"
[[ -n "${ARM_ROS_SETUP:-}" && -f "${ARM_ROS_SETUP}" ]] && source "${ARM_ROS_SETUP}" || true

started_camera=0
tmp_session="auto_camera_capture_$$"

cleanup() {
    if [[ "$started_camera" -eq 1 ]]; then
        tmux kill-session -t "$tmp_session" >/dev/null 2>&1 || true
        echo "[cleanup] Stopped temporary camera session."
    fi
}
trap cleanup EXIT

if rosnode list 2>/dev/null | grep -qE "camera|realsense|${CAMERA_NODE_HINT:-camera}"; then
    echo "Camera node is already running; leaving as-is."
else
    echo "Camera node is not running; starting temporarily for this capture."
    tmux new-session -d -s "$tmp_session" \
        "source \"${ROS_SETUP}\"; source \"${CAMERA_ROS_SETUP}\"; source \"${CONDA_SH}\"; conda activate ${CONDA_ENV}; roslaunch ${CAMERA_LAUNCH} align_depth:=true; exec bash"
    started_camera=1

    ready=0
    for _ in {1..30}; do
        if rosnode list 2>/dev/null | grep -qE "camera|realsense|${CAMERA_NODE_HINT:-camera}"; then
            ready=1
            break
        fi
        sleep 1
    done

    if [[ "$ready" -ne 1 ]]; then
        echo "Error: camera node not ready within 30 seconds; exiting."
        exit 1
    fi

    echo "Waiting 3 seconds for auto white-balance and auto-exposure to stabilize..."
    sleep 3
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"
python3 capture_three_view_rgb_depth.py
echo "Image capture finished."
