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

camera_ready() {
    timeout 3 rostopic echo /camera_f/color/image_raw -n 1 >/dev/null 2>&1 && \
    timeout 3 rostopic echo /camera_l/color/image_raw -n 1 >/dev/null 2>&1 && \
    timeout 3 rostopic echo /camera_r/color/image_raw -n 1 >/dev/null 2>&1
}

cleanup_stale_camera_nodes() {
    local need_cleanup=0
    for node in $(rosnode list 2>/dev/null | grep -E '/camera_[flr]/(camera|realsense2_camera)$|realsense2_camera_manager'); do
        if ! timeout 2 rosnode ping "$node" >/dev/null 2>&1; then
            need_cleanup=1
            break
        fi
    done
    if [[ "$need_cleanup" -eq 1 ]]; then
        yes y | timeout 60 rosnode cleanup >/dev/null 2>&1 || true
        sleep 1
    fi
}

if camera_ready; then
    echo "Camera is already publishing images; leaving as-is."
else
    if rosnode list 2>/dev/null | grep -qE 'realsense2_camera|/camera_f/camera'; then
        echo "Camera node exists but is not publishing images; cleaning residual nodes..."
        cleanup_stale_camera_nodes
    else
        echo "Camera is not running; starting temporarily for this capture."
    fi

    tmux new-session -d -s "$tmp_session" \
        "source \"${ROS_SETUP}\"; source \"${CAMERA_ROS_SETUP}\"; roslaunch ${CAMERA_LAUNCH}; exec bash"
    started_camera=1

    ready=0
    for _ in {1..30}; do
        if camera_ready; then
            ready=1
            break
        fi
        sleep 1
    done

    if [[ "$ready" -ne 1 ]]; then
        echo "Error: camera did not start publishing images within 30 seconds; exiting."
        exit 1
    fi

    echo "Waiting 3 seconds for auto white-balance and auto-exposure to stabilize..."
    sleep 3
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"
python3 capture_three_view_images.py
echo "Image capture finished."
