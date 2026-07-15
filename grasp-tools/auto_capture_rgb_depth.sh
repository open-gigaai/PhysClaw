#!/bin/bash
# Capture front camera RGB + depth into example_data/color.png and depth.png.


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PHYSCLAW_0_CALLER="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../configs/load_paths.sh"

source "${ROS_SETUP}"
[[ -n "${ARM_ROS_SETUP:-}" && -f "${ARM_ROS_SETUP}" ]] && source "${ARM_ROS_SETUP}" || true

started_camera=0
tmp_session="grasp_front_capture_$$"

cleanup() {
    if [[ "$started_camera" -eq 1 ]]; then
        tmux kill-session -t "$tmp_session" >/dev/null 2>&1 || true
        echo "[cleanup] Stopped temporary camera session."
    fi
}

trap cleanup EXIT

camera_ready() {
    timeout 3 rostopic echo "${TOPIC_FRONT_COLOR}" -n 1 >/dev/null 2>&1
}

cleanup_stale_camera_nodes() {
    local need_cleanup=0
    for node in $(rosnode list 2>/dev/null | grep -E "${CAMERA_NODE_HINT:-camera}"); do
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
    true
else
    if rosnode list 2>/dev/null | grep -qE "${CAMERA_NODE_HINT:-camera}"; then
        echo "Camera node exists but is not publishing images; cleaning residual nodes..."
        cleanup_stale_camera_nodes
    else
        echo "Camera is not running; starting temporarily for this capture."
    fi

    if [[ -z "${CAMERA_LAUNCH:-}" ]]; then echo "Error: CAMERA_LAUNCH is empty; set it in configs/paths.env"; exit 1; fi
    tmux new-session -d -s "$tmp_session" "source \"${ROS_SETUP}\"; [[ -n \"${CAMERA_ROS_SETUP}\" ]] && source \"${CAMERA_ROS_SETUP}\"; roslaunch ${CAMERA_LAUNCH}; exec bash"
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
cd "$SCRIPT_DIR"
python3 capture_front_rgb_depth.py
echo "Image capture complete. Saved under: ${SCRIPT_DIR}/example_data/"
