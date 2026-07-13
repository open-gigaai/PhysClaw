#!/bin/bash
# Stop the persistent SAM3 segmentation server started by start_seg_server.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PHYSGNOSIS_CALLER="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../configs/load_paths.sh"

SOCKET_PATH="${1:-${SCRIPT_DIR}/.sam3_seg.sock}"
PID_FILE="${SOCKET_PATH}.pid"

if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Stopped SAM3 server pid=${PID}"
    fi
    rm -f "$PID_FILE"
fi

if [[ -S "$SOCKET_PATH" ]]; then
    rm -f "$SOCKET_PATH"
fi
