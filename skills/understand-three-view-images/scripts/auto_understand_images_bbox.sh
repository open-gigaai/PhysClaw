#!/bin/bash
set -e

usage() {
    echo "Usage: $0 [--prompt \"text\"]"
}

PROMPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)
            shift
            PROMPT="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_env.sh"

source "${ROS_SETUP}"
[[ -n "${ARM_ROS_SETUP:-}" && -f "${ARM_ROS_SETUP}" ]] && source "${ARM_ROS_SETUP}" || true

started_camera=0
tmp_session="auto_camera_capture_$$"

if rosnode list 2>/dev/null | grep -qE "camera|realsense|${CAMERA_NODE_HINT:-camera}"; then
    echo "Camera node is already running; leaving as-is."
else
    echo "Camera node is not running; starting temporarily for this capture."
    tmux new-session -d -s "$tmp_session" \
        "source \"${ROS_SETUP}\"; source \"${CAMERA_ROS_SETUP}\"; source \"${CONDA_SH}\"; conda activate ${CONDA_ENV}; roslaunch ${CAMERA_LAUNCH}; exec bash"
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
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"

# Default bbox prompt when caller does not override
if [[ -z "$PROMPT" ]]; then
    PROMPT='Give bounding box coordinates for all objects in the image. Output Format: [object name: <bbox>...</bbox>][object name: <bbox>...</bbox>]'
fi
python3 capture_and_understand_one_view.py --prompt "$PROMPT"
