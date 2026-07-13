#!/bin/bash
# Start a persistent SAM3 segmentation server (model loaded once).
#
# Usage:
#   bash start_seg_server.sh [socket_path]
#
# Then run segmentation via server:
#   python run_seg.py --text_prompt "orange" --data_dir ./example_data --via-server
#
# Or use run_full_pipeline.sh / run.sh (they auto-use the server when available).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PHYSGNOSIS_CALLER="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../configs/load_paths.sh"

SOCKET_PATH="${1:-${SCRIPT_DIR}/.sam3_seg.sock}"
PID_FILE="${SOCKET_PATH}.pid"
LOG_FILE="${SOCKET_PATH}.log"

is_server_alive() {
    local socket_path="$1"
    [[ -S "$socket_path" ]] || return 1
    [[ -f "${socket_path}.pid" ]] || return 1
    local pid
    pid="$(cat "${socket_path}.pid")"
    kill -0 "$pid" 2>/dev/null
}

if is_server_alive "$SOCKET_PATH"; then
    [[ "${QUIET:-0}" == "1" ]] || echo "SAM3 server already running on ${SOCKET_PATH}"
    exit 0
fi

if [[ -S "$SOCKET_PATH" || -f "$PID_FILE" ]]; then
    echo "Removing stale SAM3 server socket/pid (process was killed)..."
    rm -f "$SOCKET_PATH" "$PID_FILE"
fi

source "${CONDA_SH}"
if [[ -z "${CONDA_ENV_SAM3:-}" ]]; then echo "Error: set CONDA_ENV_SAM3 to your official SAM3 env"; exit 1; fi
conda activate "${CONDA_ENV_SAM3}"
cd "$SCRIPT_DIR"

nohup python run_seg.py --serve --socket "$SOCKET_PATH" >"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"

for _ in $(seq 1 60); do
    if [[ -S "$SOCKET_PATH" ]]; then
        echo "SAM3 server started (pid=$(cat "$PID_FILE"), socket=${SOCKET_PATH})"
        echo "Log: ${LOG_FILE}"
        exit 0
    fi
    sleep 0.5
done

echo "Error: SAM3 server did not become ready. See ${LOG_FILE}" >&2
exit 1
