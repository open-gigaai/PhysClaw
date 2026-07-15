#!/bin/bash
# Multi-object pipeline with optional HDF5 recording.
#
# Modes:
#   lazy      Home once -> capture once -> per task on the same frame:
#             SAM + AnyGrasp + grasp+place (no re-capture between tasks)
#   diligent  Home once -> per task: capture + SAM + AnyGrasp + grasp+place
#
# Prerequisites: bring up YOUR arm driver + cameras (reference lab used Piper + RealSense).
#
# Usage:
#   bash run_multi_pipeline_recorder.sh --mode lazy [options] \
#     --task "<text_prompt>" <arm> <place_x> <place_y> <approach_depth_offset> ...
#
# Examples:
#   bash run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down \
#     --task "banana" left 0.25 -0.24 0.02 \
#     --task "green pepper" right 0.25 0.24 0.03
#
#   bash run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down --collect \
#     --task "orange" right 0.3 0.27 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PHYSCLAW_0_CALLER="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../configs/load_paths.sh"


usage() {
    cat <<EOF
Usage: $0 --mode lazy|diligent [options] \\
  --task "<text_prompt>" <arm> <place_x> <place_y> <approach_depth_offset> \\
  [--task "<text_prompt>" <arm> <place_x> <place_y> <approach_depth_offset> ...]

approach_depth_offset: extra grasp depth (m) along the approach axis, per task
(|value| <= ${MAX_APPROACH_DEPTH_OFFSET_M:-0.05} m workspace guard).

Modes:
  lazy      Home once, capture once, then per task on the same RGB-D frame:
            SAM -> AnyGrasp -> grasp+place (no re-capture).
            Warning: task N grasps from the pre-task-1 photo; if an earlier task
            disturbed its object the pose is stale. Prefer diligent for close objects.
  diligent  Home once, then per task: capture -> SAM -> AnyGrasp -> grasp+place.

Options:
  --data-dir DIR            RGB-D output dir (default: ${SCRIPT_DIR}/example_data)
  --no-gui                  Disable AnyGrasp GUI
  --verbose                 Show full SAM3/AnyGrasp logs
  --debug                   Enable AnyGrasp debug visualizations and verbose logs
  --top-down                Pass --top_down_grasp to AnyGrasp

Data collection (one HDF5 episode per task, recorded ONLY during arm motion):
  --collect                 Record cameras + arm joints while each task executes.
                            Each recording gets its own timestamped session folder:
                            <dataset-dir>/grasp_<prompt>/<YYYY-MM-DD_HH-MM-SS>/
                              episode_0_part_M.hdf5, result.txt, collect.log,
                              verify.log, <task>_<timestamp>.mp4, stats.json
                            Capture / SAM / AnyGrasp inference is never recorded.
  --collect-dataset-dir D   Episode output root (default: ${SCRIPT_DIR}/collected_data)
  --collect-depth           Also record depth images
  --collect-chunk-size N    Frames buffered in RAM before flushing (default: 1000)
  --collect-verify          After each task, photograph the scene and ask the
                            vision model (understand-three-view-images skill)
                            whether the object was really placed; verdict is
                            appended to episode_N_result.txt as vision: success|failed|unknown
  --collect-verify-prompt P Override the auto-generated YES/NO question sent to
                            the vision model (applies to every task)
  --no-collect-stats        Disable the automatic per-episode statistics
                            (episode_stats.py: action stats, smoothness, image
                            quality, aesthetics -> stats/episode_N_stats.json +
                            <dataset-dir>/stats/report.html). On by default,
                            runs in the background after each episode.
  --collect-stats-stride N  Sample every Nth frame for image metrics
                            (default: 0 = auto, ~30 frames per episode)
  --no-collect-video        Disable the automatic per-episode video
                            (<task>_<timestamp>.mp4, on by default)
  -h, --help                Show this help
EOF
}

MODE=""
DATA_DIR=""
GUI_VIZ=1
TOP_DOWN=0
QUIET=1
DEBUG=0
COLLECT=0
COLLECT_DATASET_DIR=""
COLLECT_DEPTH=0
COLLECT_CHUNK_SIZE=1000
COLLECT_VERIFY=0
COLLECT_VERIFY_PROMPT=""
COLLECT_STATS=1
COLLECT_STATS_STRIDE=0
COLLECT_VIDEO=1

HOME_JOINT=(0 0 0 0 0 0 0)
HOME_DURATION_S=0.6
HOME_RATE_HZ=30
HOME_SETTLE_S=0.5
GRASP_DURATION_S=1
PRE_GRASP_JOINT=(0.0 1.1 -1.1 0 1.1 0.0 0.0)
PRE_GRASP_DURATION_S=0.6
PLACE_LIFT_OFFSET_M=0.20
PLACE_LIFT_DURATION_S=0.6
PLACE_TRANSPORT_DURATION_S=0.6
PLACE_LOWER_BEFORE_RELEASE_M=0.13
PLACE_LOWER_DURATION_S=0.6
PLACE_RETRACT_AFTER_RELEASE_M=0.08

# Workspace guards (meters) -- reject targets the arm should never be sent to.
# Adjust to your table/robot if a legitimate target falls outside these.
PLACE_X_MIN=0.05
PLACE_X_MAX=0.60
PLACE_Y_ABS_MAX=0.60
MAX_APPROACH_DEPTH_OFFSET_M=0.05

SAM3_SOCKET="${SCRIPT_DIR}/.sam3_seg.sock"
SAM_SERVER_STARTED=0

# Each task: "prompt|arm|place_x|place_y|approach_depth_offset"
TASKS=()

is_numeric() {
    [[ "$1" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]
}

add_task() {
    local prompt="$1"
    local arm="$2"
    local place_x="$3"
    local place_y="$4"
    local approach_depth_offset="$5"

    # Tasks are packed as "prompt|arm|x|y|offset"; a '|' (or newline) in the
    # prompt would shift every field after validation and corrupt the motion command.
    if [[ -z "$prompt" || "$prompt" == *"|"* || "$prompt" == *$'\n'* ]]; then
        echo "Error: task prompt must be non-empty and must not contain '|' or newlines, got: $prompt" >&2
        exit 1
    fi
    if [[ "$arm" != "left" && "$arm" != "right" ]]; then
        echo "Error: arm must be left or right, got: $arm" >&2
        exit 1
    fi
    if ! is_numeric "$place_x" || ! is_numeric "$place_y"; then
        echo "Error: place_x and place_y must be numeric, got: ($place_x, $place_y)" >&2
        exit 1
    fi
    if ! is_numeric "$approach_depth_offset"; then
        echo "Error: approach_depth_offset must be numeric, got: $approach_depth_offset" >&2
        exit 1
    fi
    if ! awk -v x="$place_x" -v y="$place_y" \
             -v xlo="$PLACE_X_MIN" -v xhi="$PLACE_X_MAX" -v yabs="$PLACE_Y_ABS_MAX" \
             'BEGIN{exit !(x>=xlo && x<=xhi && y>=-yabs && y<=yabs)}'; then
        echo "Error: place (${place_x}, ${place_y}) outside workspace guard" \
             "(x: ${PLACE_X_MIN}..${PLACE_X_MAX}, |y| <= ${PLACE_Y_ABS_MAX})." >&2
        echo "       If this target is intentional, adjust PLACE_X_* / PLACE_Y_ABS_MAX in this script." >&2
        exit 1
    fi
    if ! awk -v v="$approach_depth_offset" -v m="$MAX_APPROACH_DEPTH_OFFSET_M" \
             'BEGIN{exit !(v>=-m && v<=m)}'; then
        echo "Error: approach_depth_offset must satisfy |value| <= ${MAX_APPROACH_DEPTH_OFFSET_M} m, got: $approach_depth_offset" >&2
        echo "       Larger offsets drive the gripper past the grasp pose (table/self collision risk)." >&2
        exit 1
    fi
    TASKS+=("${prompt}|${arm}|${place_x}|${place_y}|${approach_depth_offset}")
}

source_ros() {
    set +u
    source "${ROS_SETUP}"
    if [[ -n "${ARM_ROS_SETUP:-}" && -f "${ARM_ROS_SETUP}" ]]; then
        source "${ARM_ROS_SETUP}"
    fi
    if [[ -n "${CAMERA_ROS_SETUP:-}" && -f "${CAMERA_ROS_SETUP}" ]]; then
        source "${CAMERA_ROS_SETUP}"
    fi
    set -u
}

require_conda_env() {
    local name="$1" what="$2"
    if [[ -z "$name" ]]; then
        echo "Error: $what conda env is not set. Configure CONDA_ENV / CONDA_ENV_SAM3 / CONDA_ENV_ANYGRASP in configs/paths.env" >&2
        exit 1
    fi
}

source_conda() {
    source "${CONDA_SH}"
}

# ---------------------------------------------------------------------------
# Background data collection: one HDF5 episode per task, recorded ONLY while
# do_execute runs (arm motion). Capture/SAM/AnyGrasp are never recorded.
# Ported from run_full_pipeline.sh.
# ---------------------------------------------------------------------------
COLLECT_PID=""
COLLECT_LOG=""
COLLECT_SESSION_DIR=""
COLLECT_SESSION_TS=""

# Recorder script location. Override with COLLECT_RECORDER_PY in configs/paths.env.
# Optional local vendored recorder next to this script takes precedence when present.
if [[ -f "${SCRIPT_DIR}/collect_dataliyishan.py" ]]; then
    COLLECT_RECORDER_PY="${SCRIPT_DIR}/collect_dataliyishan.py"
fi
: "${COLLECT_RECORDER_PY:=}"
if [[ -z "${COLLECT_RECORDER_PY}" ]]; then
    echo "Warning: COLLECT_RECORDER_PY is empty; --collect will fail until you set it in configs/paths.env" >&2
fi

# Per-episode statistics script (optional feature; missing script only warns).
COLLECT_STATS_PY="${COLLECT_STATS_PY:-${SCRIPT_DIR}/collect/episode_stats.py}"

# Per-episode video renderer (optional feature; missing script only warns).
COLLECT_VIDEO_PY="${COLLECT_VIDEO_PY:-${SCRIPT_DIR}/collect/render_episode_video.py}"

# Filesystem-safe per-task episode dir name from the text prompt.
collect_task_name() {
    local name
    name="$(echo "$1" | tr ' ' '_' | tr -cd '[:alnum:]_-')"
    echo "grasp_${name:-task}"
}

# Fail fast (with a clear message) if the topics the recorder needs are not
# publishing -- otherwise it silently records zero frames ("syn fail").
collect_preflight() {
    local topics
    topics="$( (source_ros; rostopic list 2>/dev/null) || true )"
    if [[ -z "$topics" ]]; then
        echo "Error: [collect] 'rostopic list' failed -- is roscore / your arm+camera drivers running?" >&2
        exit 1
    fi
    local required=(
        "${TOPIC_FRONT_COLOR}"
        "${TOPIC_LEFT_COLOR}"
        "${TOPIC_RIGHT_COLOR}"
        "${TOPIC_JOINT_LEFT}"
        "${TOPIC_JOINT_RIGHT}"
    )
    if [[ "$COLLECT_DEPTH" -eq 1 ]]; then
        required+=(
            "${TOPIC_FRONT_DEPTH}"
            "${TOPIC_LEFT_DEPTH:-/camera_l/depth/image_rect_raw}"
            "${TOPIC_RIGHT_DEPTH:-/camera_r/depth/image_rect_raw}"
        )
    fi
    local t missing=""
    for t in "${required[@]}"; do
        grep -qx "$t" <<< "$topics" || missing="${missing} ${t}"
    done
    if [[ -n "$missing" ]]; then
        echo "Error: [collect] required topics not registered:${missing}" >&2
        echo "       Start all 3 cameras + arms first (skill 0-init-robot), or run without --collect." >&2
        exit 1
    fi
    # Registration is not enough: a hung camera node keeps its topic listed
    # while publishing nothing, and the recorder then syncs zero frames
    # ("syn fail"). Demand one real message from every topic before the arm
    # is allowed to move.
    local silent=""
    silent="$( (source_ros
        for t in "${required[@]}"; do
            timeout 5 rostopic echo -n1 "$t" > /dev/null 2>&1 || printf ' %s' "$t"
        done) )"
    if [[ -n "$silent" ]]; then
        echo "Error: [collect] topics registered but NOT publishing (no message in 5 s):${silent}" >&2
        echo "       Usually a dropped/hung camera: restart cameras (skill 0-init-robot), verify with 'rostopic hz', then retry." >&2
        exit 1
    fi
}

# start_collector <task_name>: begin recording into a fresh timestamped session
# folder <dataset>/<task>/<YYYY-MM-DD_HH-MM-SS>/. No episode-index scanning:
# timestamped folders make overwriting impossible (zero-frame episodes used to
# cause index reuse that clobbered result/log files).
start_collector() {
    local task_name="$1"
    COLLECT_SESSION_TS="$(date +%Y-%m-%d_%H-%M-%S)"
    COLLECT_SESSION_DIR="${COLLECT_DATASET_DIR}/${task_name}/${COLLECT_SESSION_TS}"
    local n=2
    while [[ -e "$COLLECT_SESSION_DIR" ]]; do   # same-second re-run
        COLLECT_SESSION_DIR="${COLLECT_DATASET_DIR}/${task_name}/${COLLECT_SESSION_TS}-${n}"
        n=$((n + 1))
    done
    COLLECT_SESSION_TS="$(basename "$COLLECT_SESSION_DIR")"
    mkdir -p "$COLLECT_SESSION_DIR"

    COLLECT_LOG="${COLLECT_SESSION_DIR}/collect.log"
    echo "[collect] recording session ${task_name}/${COLLECT_SESSION_TS} (log: ${COLLECT_LOG})"
    (
        source_ros
        source_conda
        conda activate "${CONDA_ENV}"
        # The recorder (unmodified) writes to <--dataset_dir>/<--task_name>/
        # episode_<idx>_part_M.hdf5; passing the task dir + session timestamp
        # lands everything in the session folder with a constant episode_0_ prefix.
        COLLECT_ARGS=(
            --dataset_dir "${COLLECT_DATASET_DIR}/${task_name}"
            --task_name "$COLLECT_SESSION_TS"
            --episode_idx 0
            --max_timesteps -1
            --chunk_size "$COLLECT_CHUNK_SIZE"
        )
        [[ "$COLLECT_DEPTH" -eq 1 ]] && COLLECT_ARGS+=(--use_depth_image True)
        exec python -u "$COLLECT_RECORDER_PY" "${COLLECT_ARGS[@]}"
    ) > "$COLLECT_LOG" 2>&1 &
    COLLECT_PID=$!

    # Wait for the recorder to actually reach its record loop ("Start recording"
    # in the log) instead of a blind sleep: catches slow-start crashes.
    local waited=0 ready=0
    while true; do
        if ! kill -0 "$COLLECT_PID" 2>/dev/null; then
            wait "$COLLECT_PID" 2>/dev/null || true
            COLLECT_PID=""
            echo "Error: [collect] recorder exited during startup; see $COLLECT_LOG" >&2
            exit 1
        fi
        if grep -q "Start recording" "$COLLECT_LOG" 2>/dev/null; then
            ready=1
            break
        fi
        if [[ "$waited" -ge 30 ]]; then  # 30 * 0.5 s = 15 s
            break
        fi
        sleep 0.5
        waited=$((waited + 1))
    done
    if [[ "$ready" -eq 1 ]]; then
        echo "[collect] recorder ready (pid ${COLLECT_PID})"
    else
        echo "Warning: [collect] recorder not ready after 15 s; continuing anyway (see $COLLECT_LOG)" >&2
    fi
}

# Timestamped markers appended to the collect log for frame/step alignment.
collect_mark() {
    [[ -n "$COLLECT_PID" ]] && echo "[mark] $(date +%s.%N) $*" >> "$COLLECT_LOG" || true
}

# stop_collector [exit_code]: stop recorder (SIGINT -> clean save), wait for the
# flush, and write the session's result.txt marker (success/failed).
stop_collector() {
    local rc="${1:-0}"
    [[ -n "$COLLECT_PID" ]] || return 0
    if kill -0 "$COLLECT_PID" 2>/dev/null; then
        echo ""
        echo "[collect] stopping recorder (pid ${COLLECT_PID}), flushing episode to disk (do not Ctrl-C)..."
        kill -INT "$COLLECT_PID" 2>/dev/null || true
        # Escalate in case SIGINT is ignored; the recorder's SIGTERM handler is
        # registered explicitly and always works. The final save can take tens
        # of seconds, so only hard-kill after a generous timeout.
        local t=0
        while kill -0 "$COLLECT_PID" 2>/dev/null; do
            if [[ "$t" -eq 20 ]]; then      # 10 s: not exiting -> also send TERM
                kill -TERM "$COLLECT_PID" 2>/dev/null || true
            elif [[ "$t" -ge 240 ]]; then   # 120 s: give up, hard kill
                echo "[collect] Warning: recorder did not exit after 120 s; killing (episode may be incomplete)" >&2
                kill -KILL "$COLLECT_PID" 2>/dev/null || true
                break
            fi
            sleep 0.5
            t=$((t + 1))
        done
        wait "$COLLECT_PID" 2>/dev/null || true
    fi
    COLLECT_PID=""
    local result_file="${COLLECT_SESSION_DIR}/result.txt"
    if [[ "$rc" -eq 0 ]]; then
        echo "success" > "$result_file" || true
    else
        echo "failed rc=${rc}" > "$result_file" || true
    fi
    if compgen -G "${COLLECT_SESSION_DIR}/episode_0_part_*.hdf5" > /dev/null; then
        echo "[collect] saved (result: $(cat "$result_file" 2>/dev/null)):"
        ls -lh "${COLLECT_SESSION_DIR}/episode_0_part_"*.hdf5 || true
    else
        echo "[collect] Warning: no episode files were written (cameras not publishing?); see ${COLLECT_LOG}" >&2
    fi
}

# Vision-model verdict on physical task success. Photographs the scene via the
# understand-three-view-images skill, asks a strict YES/NO question, and appends
# the verdict to the episode's result file. Never fails the pipeline: a broken
# vision check yields "unknown", not an abort.
UNDERSTAND_SH="${UNDERSTAND_SH}"


collect_verify() {
    local prompt="$1" place_x="$2" place_y="$3"
    local result_file="${COLLECT_SESSION_DIR}/result.txt"
    local vlog="${COLLECT_SESSION_DIR}/verify.log"
    if [[ ! -f "$UNDERSTAND_SH" ]]; then
        echo "[collect] Warning: vision verify skipped, script not found: $UNDERSTAND_SH" >&2
        echo "vision: unknown (verify script missing)" >> "$result_file" || true
        return 0
    fi
    local question="$COLLECT_VERIFY_PROMPT"
    if [[ -z "$question" ]]; then
        question="The robot arm just tried to pick up the ${prompt} and place it at the white plate on the table. Look at the scene: is the ${prompt} now at the plate, and partially being on the plate still counts as a success? Start your reply with exactly one word: YES or NO. Then one short sentence of explanation."
    fi
    echo "[collect] vision verify: asking the model about \"${prompt}\"..."
    local answer verdict
    # Full judge output (stdout AND stderr) is kept in verify.log so every
    # "unknown" verdict is diagnosable; the whole call is time-boxed.
    timeout 120 bash "$UNDERSTAND_SH" --prompt "$question" > "$vlog" 2>&1 || true
    answer="$(tr '\n' ' ' < "$vlog" 2>/dev/null | tail -c 300)"
    # Infrastructure failures first: their messages may contain a stray
    # English "No" ("No route to host") that must not become a real verdict.
    if grep -qE 'Request failed|Unable to get|Image conversion failed|Image encoding failed|not ready|Traceback' "$vlog" 2>/dev/null; then
        verdict="unknown"
        echo "[collect] Warning: vision infrastructure failed; see $vlog" >&2
    elif grep -qiE '(^|[^A-Za-z])YES([^A-Za-z]|$)' "$vlog" 2>/dev/null; then
        verdict="success"
    elif grep -qiE '(^|[^A-Za-z])NO([^A-Za-z]|$)' "$vlog" 2>/dev/null; then
        verdict="failed"
    else
        verdict="unknown"
        echo "[collect] Warning: no YES/NO in vision answer; see $vlog" >&2
    fi
    {
        echo "vision: ${verdict}"
        echo "vision_answer: ${answer}"
    } >> "$result_file" || true
    echo "[collect] vision verdict: ${verdict}"
}

# Background per-episode statistics: episode_stats.py computes action stats,
# smoothness, image quality and aesthetics for the episode just saved, then
# refreshes <dataset-dir>/stats/report.html. Backgrounded + disowned so the
# next task's capture/inference is not delayed and the EXIT trap never waits
# on it; concurrent runs are safe (flock inside the Python). Never fails the
# pipeline. Must run AFTER collect_verify (it reads the vision: line).
collect_stats() {
    [[ "$COLLECT" -eq 1 && "$COLLECT_STATS" -eq 1 ]] || return 0
    [[ -n "$COLLECT_SESSION_DIR" ]] || return 0
    local stats_log="${COLLECT_SESSION_DIR}/stats.log"
    (
        source_conda
        conda activate "${CONDA_ENV}"
        exec python -u "$COLLECT_STATS_PY" \
            --session-dir "$COLLECT_SESSION_DIR" \
            --dataset-dir "$COLLECT_DATASET_DIR" \
            --image-stride "$COLLECT_STATS_STRIDE" \
            --quiet
    ) > "$stats_log" 2>&1 &
    disown $! 2>/dev/null || true
    echo "[collect] stats: computing session ${COLLECT_SESSION_TS} in background" \
         "(report: ${COLLECT_DATASET_DIR}/stats/report.html)"
}

# Background per-episode video: the 3 cameras side by side at the true 30 fps,
# saved as <task>_<session>.mp4 inside the session folder. Same warn-only,
# backgrounded pattern as collect_stats.
collect_video() {
    [[ "$COLLECT" -eq 1 && "$COLLECT_VIDEO" -eq 1 ]] || return 0
    [[ -n "$COLLECT_SESSION_DIR" ]] || return 0
    (
        source_conda
        conda activate "${CONDA_ENV}"
        exec python -u "$COLLECT_VIDEO_PY" --session-dir "$COLLECT_SESSION_DIR"
    ) > "${COLLECT_SESSION_DIR}/video.log" 2>&1 &
    disown $! 2>/dev/null || true
    echo "[collect] video: rendering session ${COLLECT_SESSION_TS} in background"
}

# Ensure a live recorder is stopped and its episode saved even if a task fails
# or the script is signalled. INT/TERM/HUP -> exit -> EXIT trap does cleanup.
trap 'rc=$?; stop_collector "$rc"' EXIT
trap 'echo "Interrupted (SIGINT)." >&2; exit 130' INT
trap 'echo "Terminated (SIGTERM)." >&2; exit 143' TERM
trap 'echo "Hangup (SIGHUP)." >&2; exit 129' HUP

do_home() {
    echo ""
    echo "=== Arm homing (joint) ==="
    source_ros
    cd "$SCRIPT_DIR"
    python3 joint_publisher.py \
        --left "${HOME_JOINT[@]}" \
        --right "${HOME_JOINT[@]}" \
        --duration-s "$HOME_DURATION_S" \
        --rate-hz "$HOME_RATE_HZ"
    sleep "$HOME_SETTLE_S"
}

do_capture() {
    echo ""
    echo "=== RGB-D capture ==="
    bash "$SCRIPT_DIR/auto_capture_rgb_depth.sh"
}

ensure_sam_server() {
    if [[ "$SAM_SERVER_STARTED" -eq 1 ]]; then
        return 0
    fi
    QUIET="$QUIET" bash "$SCRIPT_DIR/start_seg_server.sh" "$SAM3_SOCKET"
    SAM_SERVER_STARTED=1
}

do_sam() {
    local text_prompt="$1"
    echo ""
    echo "=== SAM3 segmentation: \"${text_prompt}\" ==="
    ensure_sam_server
    source_conda
    require_conda_env "${CONDA_ENV_SAM3}" "SAM3"; conda activate "${CONDA_ENV_SAM3}"
    cd "$SCRIPT_DIR"
    local seg_args=(
        --text_prompt "$text_prompt"
        --data_dir "$DATA_DIR"
        --via-server
        --socket "$SAM3_SOCKET"
    )
    [[ "$QUIET" -eq 1 ]] && seg_args+=(--quiet)
    python run_seg.py "${seg_args[@]}"
}

do_anygrasp() {
    local arm="$1"
    local default_npy="${SCRIPT_DIR}/eef_pose_xyzrpy.npy"
    echo ""
    echo "=== AnyGrasp (${arm}) -> ${default_npy} ==="
    rm -f "$default_npy"
    source_conda
    export PATH="/usr/sbin:/sbin:/usr/bin:/bin:$PATH"
    require_conda_env "${CONDA_ENV_ANYGRASP}" "AnyGrasp"; conda activate "${CONDA_ENV_ANYGRASP}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
    cd "$SCRIPT_DIR"
    local grasp_args=(--arm "$arm" --data_dir "$DATA_DIR")
    [[ "$TOP_DOWN" -eq 1 ]] && grasp_args+=(--top_down_grasp)
    [[ "$GUI_VIZ" -eq 1 ]] && grasp_args+=(--gui_viz)
    [[ "$QUIET" -eq 1 ]] && grasp_args+=(--quiet)
    [[ "$DEBUG" -eq 1 ]] && grasp_args+=(--debug)
    PYTHONNOUSERSITE=1 python run_grasp_mask.py "${grasp_args[@]}"
    [[ -f "$default_npy" ]] || {
        echo "Error: AnyGrasp did not produce grasp pose" >&2
        exit 1
    }
}

do_sam_and_anygrasp() {
    local text_prompt="$1"
    local arm="$2"
    do_sam "$text_prompt"
    do_anygrasp "$arm"
}

do_execute() {
    local arm="$1"
    local place_x="$2"
    local place_y="$3"
    local npy_path="$4"
    local approach_depth_offset="$5"
    echo ""
    echo "=== Execute grasp + place (${arm}) -> (${place_x}, ${place_y}) [approach offset=${approach_depth_offset} m] ==="
    source_ros
    source_conda
    conda activate "${CONDA_ENV}"
    cd "$SCRIPT_DIR"
    local exec_args=(
        --arm "$arm"
        --npy "$npy_path"
        --idle-joint "${HOME_JOINT[@]}"
        --home-left-joint "${HOME_JOINT[@]}"
        --home-right-joint "${HOME_JOINT[@]}"
        --approach-depth-offset-m "$approach_depth_offset"
        # --pre-grasp-joint "${PRE_GRASP_JOINT[@]}"
        --pre-grasp-duration-s "$PRE_GRASP_DURATION_S"
        --grasp-duration-s "$GRASP_DURATION_S"
        --place-lift-offset-m "$PLACE_LIFT_OFFSET_M"
        --place-lift-duration-s "$PLACE_LIFT_DURATION_S"
        --place-transport-duration-s "$PLACE_TRANSPORT_DURATION_S"
        --place-lower-before-release-m "$PLACE_LOWER_BEFORE_RELEASE_M"
        --place-lower-duration-s "$PLACE_LOWER_DURATION_S"
        --place-retract-after-release-m "$PLACE_RETRACT_AFTER_RELEASE_M"
        --place-x "$place_x"
        --place-y "$place_y"
        --home-duration-s "$HOME_DURATION_S"
    )
    [[ "$QUIET" -eq 1 ]] && exec_args+=(--quiet)
    python joint_grasp_publisher_npy.py "${exec_args[@]}"
}

run_lazy_mode() {
    local total=${#TASKS[@]}

    do_home
    do_capture

    for i in "${!TASKS[@]}"; do
        IFS='|' read -r prompt arm place_x place_y approach_depth_offset <<< "${TASKS[$i]}"
        local npy_path="${SCRIPT_DIR}/eef_pose_xyzrpy.npy"
        echo ""
        echo "########################################"
        echo "# Task $((i + 1))/${total} (lazy): \"${prompt}\" -> (${place_x}, ${place_y}) [${arm}] offset=${approach_depth_offset} m"
        echo "# Using the same RGB-D frame (no re-capture)"
        echo "########################################"
        do_sam_and_anygrasp "$prompt" "$arm"
        if [[ "$COLLECT" -eq 1 ]]; then
            start_collector "$(collect_task_name "$prompt")"
        fi
        collect_mark "task_$((i + 1))_execute_start prompt=${prompt} arm=${arm}"
        do_execute "$arm" "$place_x" "$place_y" "$npy_path" "$approach_depth_offset"
        collect_mark "task_$((i + 1))_execute_end"
        if [[ "$COLLECT" -eq 1 ]]; then
            stop_collector 0
            if [[ "$COLLECT_VERIFY" -eq 1 ]]; then
                collect_verify "$prompt" "$place_x" "$place_y"
            fi
            collect_stats
            collect_video
        fi
    done
}

run_diligent_mode() {
    local total=${#TASKS[@]}

    do_home

    for i in "${!TASKS[@]}"; do
        IFS='|' read -r prompt arm place_x place_y approach_depth_offset <<< "${TASKS[$i]}"
        local npy_path="${SCRIPT_DIR}/eef_pose_xyzrpy.npy"
        echo ""
        echo "########################################"
        echo "# Task $((i + 1))/${total}: \"${prompt}\" -> (${place_x}, ${place_y}) [${arm}] offset=${approach_depth_offset} m"
        echo "########################################"
        do_capture
        do_sam_and_anygrasp "$prompt" "$arm"
        if [[ "$COLLECT" -eq 1 ]]; then
            start_collector "$(collect_task_name "$prompt")"
        fi
        collect_mark "task_$((i + 1))_execute_start prompt=${prompt} arm=${arm}"
        do_execute "$arm" "$place_x" "$place_y" "$npy_path" "$approach_depth_offset"
        collect_mark "task_$((i + 1))_execute_end"
        if [[ "$COLLECT" -eq 1 ]]; then
            stop_collector 0
            if [[ "$COLLECT_VERIFY" -eq 1 ]]; then
                collect_verify "$prompt" "$place_x" "$place_y"
            fi
            collect_stats
            collect_video
        fi
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --mode)
            MODE="${2:?--mode requires lazy or diligent}"
            shift 2
            ;;
        --no-gui) GUI_VIZ=0; shift ;;
        --verbose) QUIET=0; shift ;;
        --debug) QUIET=0; DEBUG=1; shift ;;
        --top-down) TOP_DOWN=1; shift ;;
        --collect) COLLECT=1; shift ;;
        --collect-dataset-dir) COLLECT_DATASET_DIR="${2:?--collect-dataset-dir requires a value}"; shift 2 ;;
        --collect-depth) COLLECT_DEPTH=1; shift ;;
        --collect-chunk-size) COLLECT_CHUNK_SIZE="${2:?--collect-chunk-size requires a value}"; shift 2 ;;
        --collect-verify) COLLECT_VERIFY=1; shift ;;
        --collect-verify-prompt) COLLECT_VERIFY_PROMPT="${2:?--collect-verify-prompt requires a value}"; shift 2 ;;
        --no-collect-stats) COLLECT_STATS=0; shift ;;
        --collect-stats-stride) COLLECT_STATS_STRIDE="${2:?--collect-stats-stride requires a value}"; shift 2 ;;
        --no-collect-video) COLLECT_VIDEO=0; shift ;;
        --data-dir)
            DATA_DIR="${2:?--data-dir requires a value}"
            shift 2
            ;;
        --task)
            [[ $# -ge 6 ]] || {
                echo "Error: --task requires: \"<text_prompt>\" <arm> <place_x> <place_y> <approach_depth_offset>" >&2
                exit 1
            }
            add_task "$2" "$3" "$4" "$5" "$6"
            shift 6
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

[[ -n "$MODE" ]] || { echo "Error: --mode lazy|diligent is required" >&2; usage; exit 1; }
[[ "$MODE" == "lazy" || "$MODE" == "diligent" ]] || {
    echo "Error: --mode must be lazy or diligent, got: $MODE" >&2
    exit 1
}
[[ ${#TASKS[@]} -gt 0 ]] || { usage; exit 1; }

# Single-instance guard: two pipelines streaming joint commands to the same
# arms is dangerous. flock auto-releases when this process exits or dies.
LOCK_FILE="${SCRIPT_DIR}/.pipeline.lock"
if command -v flock >/dev/null 2>&1; then
    exec 200>"$LOCK_FILE"
    if ! flock -n 200; then
        echo "Error: another pipeline run is already active (lock: $LOCK_FILE)." >&2
        echo "       Wait for it to finish before starting a new run." >&2
        exit 1
    fi
fi

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/example_data}"
mkdir -p "$DATA_DIR"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

# Collection: resolve config and fail fast BEFORE the arms ever move.
if [[ "$COLLECT" -eq 1 ]]; then
    COLLECT_DATASET_DIR="${COLLECT_DATASET_DIR:-${SCRIPT_DIR}/collected_data}"
    if [[ ! -f "$COLLECT_RECORDER_PY" ]]; then
        echo "Error: [collect] recorder script not found: $COLLECT_RECORDER_PY" >&2
        echo "       Set COLLECT_RECORDER_PY or place the recorder at that path." >&2
        exit 1
    fi
    RECORDER_BASE="$(basename "$COLLECT_RECORDER_PY")"
    if pgrep -f "python.*${RECORDER_BASE}" > /dev/null 2>&1; then
        echo "Error: [collect] a recorder process (${RECORDER_BASE}) is already running." >&2
        echo "       Stop it first: pkill -INT -f ${RECORDER_BASE}" >&2
        exit 1
    fi
    collect_preflight
    # Stats is an optional feature: a missing script degrades with a warning
    # (contrast: a missing recorder is fatal above -- no data would be saved).
    if [[ "$COLLECT_STATS" -eq 1 && ! -f "$COLLECT_STATS_PY" ]]; then
        echo "Warning: [collect] stats script not found: $COLLECT_STATS_PY -- disabling per-episode stats." >&2
        echo "         Set COLLECT_STATS_PY or place episode_stats.py under collect/." >&2
        COLLECT_STATS=0
    fi
    if [[ "$COLLECT_VIDEO" -eq 1 && ! -f "$COLLECT_VIDEO_PY" ]]; then
        echo "Warning: [collect] video script not found: $COLLECT_VIDEO_PY -- disabling per-episode video." >&2
        echo "         Set COLLECT_VIDEO_PY or place render_episode_video.py under collect/." >&2
        COLLECT_VIDEO=0
    fi
fi

echo "========================================"
echo " Multi-object grasp pipeline"
echo " mode: ${MODE}  tasks: ${#TASKS[@]}  data: ${DATA_DIR}"
[[ "$COLLECT" -eq 1 ]] && echo " collect: ${COLLECT_DATASET_DIR} (per task, execute-only)"
echo "========================================"
for i in "${!TASKS[@]}"; do
    IFS='|' read -r prompt arm place_x place_y approach_depth_offset <<< "${TASKS[$i]}"
    echo "  $((i + 1)). \"${prompt}\"  arm=${arm}  place=(${place_x}, ${place_y})  offset=${approach_depth_offset} m"
done
echo "========================================"

if [[ "$MODE" == "lazy" ]]; then
    run_lazy_mode
else
    run_diligent_mode
fi

echo ""
echo "All ${#TASKS[@]} task(s) completed (mode=${MODE})."
