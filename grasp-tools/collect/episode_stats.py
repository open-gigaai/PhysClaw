#!/usr/bin/env python3
# -- coding: UTF-8
"""Post-collection episode statistics for ALOHA-style HDF5 episodes.

Each recording lives in its own session folder <dataset>/<task>/<timestamp>/
containing episode_0_part_M.hdf5, result.txt and collect.log. This computes:
  - action stats: per-dim mean/std/var/min/max/range, joint path length, gripper events
  - smoothness: RMS jerk, max joint jump, discontinuity count, 0-100 score
  - image quality per camera: sharpness / brightness / contrast / exposure / blur
  - aesthetics: heuristic 0-100 composite (sharpness, colorfulness, exposure);
    schema reserves model_score/model_name for a future server-side deep model
and writes:
  <session_dir>/stats.json                   per-episode stats
  <dataset_dir>/stats/dataset_stats.json     aggregate over all tasks
  <dataset_dir>/stats/report.html            self-contained visual report

Dependencies: numpy, h5py, cv2 (conda env `aloha`). No torch, no matplotlib.

Usage:
  # single session (called by the pipeline right after an episode is saved)
  python episode_stats.py --session-dir <dataset>/<task>/<timestamp> --dataset-dir <dir>

  # incremental batch over an existing dataset (skips up-to-date sessions)
  python episode_stats.py --dataset-dir <dir> --all [--force]

  # only re-aggregate + regenerate the HTML report
  python episode_stats.py --dataset-dir <dir> --report-only
"""
import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import h5py
import cv2

SCHEMA_VERSION = 3
FPS = 30
LEFT_ARM = list(range(0, 6))
RIGHT_ARM = list(range(7, 13))
GRIP_L, GRIP_R = 6, 13

# A frame-to-frame joint jump above this (rad per frame at 30 Hz ~ 3 rad/s) is a
# discontinuity: dropped frames, controller glitch, or collision recoil.
DISCONTINUITY_RAD = 0.10

# Smoothness score reference jerk (rad/s^3): score = 100*exp(-rms_jerk/J_REF).
# Calibration knob -- set J_REF ~= 10x the rms_jerk of a known-good episode so
# good runs score ~90. Calibrated on the first real Piper teleop batch: the
# smoothest episodes have rms_jerk ~120 (30 Hz triple finite-difference is noisy
# and inflates jerk), so J_REF=1200 puts normal motion in the ~85-90 range on the
# jerk term. The default 50 was far too strict -- only a motionless arm scored 100.
J_REF = 1200.0

# Gripper with total travel below this (rad) is considered never actuated.
GRIPPER_MIN_RANGE = 0.05
# Hysteresis thresholds on the 0-1 normalized gripper for open/close counting.
GRIP_CLOSE_T, GRIP_OPEN_T = 0.25, 0.75

SERIES_MAX_POINTS = 300  # chart series embedded in JSON are decimated to this


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _round(x, nd=4):
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return round(x, nd)


def jsonable(obj, nd=4):
    """Recursively convert numpy types to plain JSON types; NaN/inf -> null."""
    if isinstance(obj, dict):
        return {k: jsonable(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, nd) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist(), nd)
    if isinstance(obj, (np.floating, float)):
        return _round(obj, nd)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(obj), f, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def write_text_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def downsample(arr, max_points=SERIES_MAX_POINTS):
    """Uniform decimation for chart series. Returns (stride, decimated ndarray)."""
    n = len(arr)
    if n <= max_points:
        return 1, np.asarray(arr)
    stride = int(math.ceil(n / max_points))
    return stride, np.asarray(arr)[::stride]


# ---------------------------------------------------------------------------
# session discovery / loading
# ---------------------------------------------------------------------------

_PART_RE = re.compile(r"episode_(\d+)_part_(\d+)\.hdf5$")


def find_parts(session_dir):
    """Part files of one session's episode, sorted by numeric part index.
    The recorder always writes episode_0_* inside a session folder."""
    paths = glob.glob(os.path.join(session_dir, "episode_0_part_*.hdf5"))
    def part_idx(p):
        m = _PART_RE.search(p)
        return int(m.group(2)) if m else 0
    return sorted(paths, key=part_idx)


def fingerprint(parts):
    h = hashlib.sha1()
    for p in parts:
        st = os.stat(p)
        h.update(("%s:%d:%d;" % (os.path.basename(p), st.st_size, st.st_mtime_ns)).encode())
    return h.hexdigest()


def load_episode_arrays(parts, warnings):
    """Concatenate the small (non-image) arrays across parts.

    Returns dict with qpos/action (T,14), per-part frame counts aligned with the
    readable subset of `parts`, camera names, and depth availability. Images are
    never loaded here.
    """
    qpos_l, action_l, lengths, ok_parts = [], [], [], []
    cameras, has_depth = None, False
    for p in parts:
        try:
            with h5py.File(p, "r") as f:
                q = f["observations/qpos"][()]
                a = f["action"][()]
                if cameras is None:
                    cameras = sorted(f["observations/images"].keys())
                    has_depth = "images_depth" in f["observations"]
        except Exception as e:  # unreadable part: warn, keep going
            warnings.append("unreadable part %s: %s" % (os.path.basename(p), e))
            continue
        qpos_l.append(q)
        action_l.append(a)
        lengths.append(len(q))
        ok_parts.append(p)
    if not qpos_l:
        return {"qpos": None, "action": None, "part_lengths": [], "parts": [],
                "cameras": [], "has_depth": False}
    return {"qpos": np.concatenate(qpos_l), "action": np.concatenate(action_l),
            "part_lengths": lengths, "parts": ok_parts,
            "cameras": cameras or [], "has_depth": has_depth}


def iter_sampled_frames(parts, part_lengths, dataset_name, stride):
    """Yield (global_index, frame ndarray) reading every `stride`-th frame.

    Cheap because image datasets are chunked (1,H,W,...): one chunk per frame.
    Opens each part once, in order.
    """
    total = sum(part_lengths)
    wanted = list(range(0, total, max(stride, 1)))
    w = 0
    offset = 0
    for p, plen in zip(parts, part_lengths):
        local = []
        while w < len(wanted) and wanted[w] < offset + plen:
            local.append(wanted[w] - offset)
            w += 1
        if local:
            with h5py.File(p, "r") as f:
                dset = f[dataset_name]
                for li in local:
                    yield offset + li, dset[li]
        offset += plen
        if w >= len(wanted):
            break


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def count_gripper_events(g):
    """Open/close transition counts on one gripper trajectory via hysteresis."""
    gmin, gmax = float(np.min(g)), float(np.max(g))
    out = {"open_count": 0, "close_count": 0, "min": gmin, "max": gmax,
           "actuated": bool(gmax - gmin >= GRIPPER_MIN_RANGE)}
    if not out["actuated"]:
        return out
    gn = (g - gmin) / (gmax - gmin)
    state = "open" if gn[0] > 0.5 else "closed"
    for v in gn[1:]:
        if state == "open" and v < GRIP_CLOSE_T:
            state = "closed"
            out["close_count"] += 1
        elif state == "closed" and v > GRIP_OPEN_T:
            state = "open"
            out["open_count"] += 1
    return out


def compute_action_stats(action):
    """Per-dim stats + travel metrics + gripper events. action == qpos here."""
    arm_dims = LEFT_ARM + RIGHT_ARM
    d = np.diff(action, axis=0) if len(action) > 1 else np.zeros((0, action.shape[1]))
    return {
        "per_dim": {
            "mean": action.mean(axis=0),
            "std": action.std(axis=0),
            "var": action.var(axis=0),
            "min": action.min(axis=0),
            "max": action.max(axis=0),
            "range": action.max(axis=0) - action.min(axis=0),
        },
        "path_length_rad": float(np.abs(d[:, arm_dims]).sum()) if len(d) else 0.0,
        "mean_step_norm": float(np.linalg.norm(d[:, arm_dims], axis=1).mean()) if len(d) else 0.0,
        "gripper": {
            "left": count_gripper_events(action[:, GRIP_L]),
            "right": count_gripper_events(action[:, GRIP_R]),
        },
    }


def compute_smoothness(qpos, fps=FPS):
    """Jerk-based smoothness on arm joints (grippers excluded: bang-bang commands
    would dominate). Needs T >= 4 for jerk; otherwise returns nulls."""
    T = len(qpos)
    dq = np.abs(np.diff(qpos[:, LEFT_ARM + RIGHT_ARM], axis=0)) if T > 1 else None
    out = {
        "rms_jerk": {"left": None, "right": None, "overall": None},
        "max_joint_jump_rad": float(dq.max()) if dq is not None and len(dq) else None,
        "discontinuity_count": int((dq.max(axis=1) > DISCONTINUITY_RAD).sum())
                               if dq is not None and len(dq) else None,
        "score": None,
        "series": None,
    }
    if T < 4:
        return out
    dt = 1.0 / fps
    vel = {"left": np.diff(qpos[:, LEFT_ARM], axis=0) / dt,
           "right": np.diff(qpos[:, RIGHT_ARM], axis=0) / dt}
    jerk = {}
    for side, v in vel.items():
        acc = np.diff(v, axis=0) / dt
        j = np.diff(acc, axis=0) / dt
        jerk[side] = j
        out["rms_jerk"][side] = float(np.sqrt(np.mean(j ** 2)))
    out["rms_jerk"]["overall"] = float(np.sqrt(np.mean(
        np.concatenate([jerk["left"], jerk["right"]], axis=1) ** 2)))
    score = 100.0 * math.exp(-out["rms_jerk"]["overall"] / J_REF) \
        - 5.0 * out["discontinuity_count"]
    out["score"] = float(np.clip(score, 0.0, 100.0))

    stride, _ = downsample(np.zeros(T - 3))
    out["series"] = {
        "stride": stride,
        "vel_norm_left": np.linalg.norm(vel["left"], axis=1)[::stride],
        "vel_norm_right": np.linalg.norm(vel["right"], axis=1)[::stride],
        "jerk_norm_left": np.linalg.norm(jerk["left"], axis=1)[::stride],
        "jerk_norm_right": np.linalg.norm(jerk["right"], axis=1)[::stride],
        "discontinuity_frames": np.nonzero(dq.max(axis=1) > DISCONTINUITY_RAD)[0][:100],
    }
    return out


def frame_metrics(img):
    """Quality metrics of one RGB/BGR uint8 frame (channel order irrelevant here)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    over = float((gray >= 250).mean())
    under = float((gray <= 5).mean())
    # Hasler-Suesstrunk colorfulness
    b = img.astype(np.float32)
    rg = b[:, :, 2] - b[:, :, 1]
    yb = 0.5 * (b[:, :, 2] + b[:, :, 1]) - b[:, :, 0]
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return {"sharpness": lap_var, "brightness": brightness, "contrast": contrast,
            "over": over, "under": under, "colorfulness": colorfulness}


def _aesthetic_components(m):
    s = float(np.clip(math.log10(1.0 + m["sharpness"]) / 3.0, 0.0, 1.0))
    c = float(np.clip(m["colorfulness"] / 80.0, 0.0, 1.0))
    e = float(math.exp(-((m["brightness"] - 128.0) / 64.0) ** 2)
              * np.clip(1.0 - 4.0 * (m["over"] + m["under"]), 0.0, 1.0))
    return s, c, e


def compute_image_quality(parts, part_lengths, cameras, has_depth, stride):
    """Per-camera stats over sampled frames + heuristic aesthetics."""
    total = sum(part_lengths)
    if stride <= 0:
        stride = max(1, total // 30)  # ~30 samples per episode regardless of length
    cams_out = {}
    aest_cam = {}
    comp_sums = np.zeros(3)
    comp_n = 0
    for cam in cameras:
        per = []
        for _, frame in iter_sampled_frames(parts, part_lengths,
                                            "observations/images/" + cam, stride):
            per.append(frame_metrics(frame))
        if not per:
            continue
        sharp = np.array([m["sharpness"] for m in per])
        bright = np.array([m["brightness"] for m in per])
        blur_frac = float((sharp < 0.5 * np.median(sharp)).mean())
        depth_fill = None
        if has_depth:
            fills = [float((d > 0).mean()) for _, d in iter_sampled_frames(
                parts, part_lengths, "observations/images_depth/" + cam, stride * 3)]
            depth_fill = float(np.mean(fills)) if fills else None
        cams_out[cam] = {
            "frames_sampled": len(per),
            "sharpness": {"mean": sharp.mean(), "min": sharp.min(),
                          "max": sharp.max(), "series": sharp},
            "brightness": {"mean": bright.mean(), "series": bright},
            "contrast": {"mean": float(np.mean([m["contrast"] for m in per]))},
            "colorfulness": {"mean": float(np.mean([m["colorfulness"] for m in per]))},
            "overexposed_frac": float(np.mean([m["over"] for m in per])),
            "underexposed_frac": float(np.mean([m["under"] for m in per])),
            "blur_frac": blur_frac,
            "depth_fill_rate": depth_fill,
        }
        scores = []
        for m in per:
            s, c, e = _aesthetic_components(m)
            scores.append(100.0 * (0.4 * s + 0.3 * c + 0.3 * e))
            comp_sums += (s, c, e)
            comp_n += 1
        aest_cam[cam] = float(np.mean(scores))

    aesthetics = {
        "score": float(np.mean(list(aest_cam.values()))) if aest_cam else None,
        "method": "heuristic_v1",
        "components": ({"sharpness": comp_sums[0] / comp_n,
                        "colorfulness": comp_sums[1] / comp_n,
                        "exposure": comp_sums[2] / comp_n} if comp_n else None),
        "per_camera": aest_cam or None,
        # reserved for a server-side deep model (e.g. NIMA) to fill in later
        "model_score": None,
        "model_name": None,
    }
    return {"stride": stride, "has_depth": has_depth, "cameras": cams_out}, aesthetics


# ---------------------------------------------------------------------------
# sidecar files
# ---------------------------------------------------------------------------

def read_result(session_dir, warnings):
    out = {"script": "unknown", "script_rc": None, "vision": None, "vision_answer": None}
    path = os.path.join(session_dir, "result.txt")
    if not os.path.isfile(path):
        warnings.append("missing result.txt")
        return out
    try:
        # errors="replace": a vision_answer once contained a torn multi-byte
        # UTF-8 char (byte-level tail -c truncation); one bad byte must not
        # kill stats for the whole session
        with open(path, errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        warnings.append("unreadable result.txt: %s" % e)
        return out
    if lines:
        first = lines[0]
        if first == "success":
            out["script"], out["script_rc"] = "success", 0
        elif first.startswith("failed"):
            out["script"] = "failed"
            m = re.search(r"rc=(\d+)", first)
            out["script_rc"] = int(m.group(1)) if m else None
    for ln in lines[1:]:
        if ln.startswith("vision:"):
            out["vision"] = ln.split(":", 1)[1].split()[0].strip()
        elif ln.startswith("vision_answer:"):
            out["vision_answer"] = ln.split(":", 1)[1].strip()
    return out


def read_marks(session_dir):
    """Wall-clock execute window from the [mark] lines in the collect log."""
    path = os.path.join(session_dir, "collect.log")
    out = {"execute_start": None, "execute_end": None, "task_index": None}
    try:
        with open(path, errors="replace") as f:
            for ln in f:
                m = re.match(r"\[mark\] ([\d.]+) task_(\d+)_execute_(start|end)", ln)
                if m:
                    ts, idx, kind = float(m.group(1)), int(m.group(2)), m.group(3)
                    out["task_index"] = idx
                    out["execute_" + kind] = ts
    except OSError:
        return None
    if out["execute_start"] is None and out["execute_end"] is None:
        return None
    return out


# ---------------------------------------------------------------------------
# per-session driver
# ---------------------------------------------------------------------------

def build_episode_stats(session_dir, image_stride):
    warnings = []
    session_dir = os.path.normpath(session_dir)
    session = os.path.basename(session_dir)
    task = os.path.basename(os.path.dirname(session_dir))
    parts = find_parts(session_dir)
    result = read_result(session_dir, warnings)
    marks = read_marks(session_dir)
    stats = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "session": session,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"parts": [os.path.basename(p) for p in parts],
                   "total_bytes": sum(os.path.getsize(p) for p in parts),
                   "fingerprint": fingerprint(parts)},
        "frames": 0, "fps": FPS, "duration_s": 0.0, "action_is_qpos": True,
        "result": result, "marks": marks,
        "action": None, "smoothness": None, "trajectories": None,
        "image_quality": None, "aesthetics": None,
        "warnings": warnings,
    }
    if not parts:
        warnings.append("no hdf5 parts (zero-frame episode)")
        return stats

    arrs = load_episode_arrays(parts, warnings)
    if arrs["qpos"] is None:
        warnings.append("no readable parts")
        return stats
    qpos = arrs["qpos"]
    stats["frames"] = int(len(qpos))
    stats["duration_s"] = len(qpos) / float(FPS)

    stats["action"] = compute_action_stats(arrs["action"])
    stats["smoothness"] = compute_smoothness(qpos)
    stride, q_ds = downsample(qpos)
    stats["trajectories"] = {"stride": stride, "qpos": q_ds}
    stats["image_quality"], stats["aesthetics"] = compute_image_quality(
        arrs["parts"], arrs["part_lengths"], arrs["cameras"], arrs["has_depth"],
        image_stride)
    return stats


def stats_json_path(session_dir):
    return os.path.join(session_dir, "stats.json")


def process_session(session_dir, image_stride, force, quiet):
    """Compute + write one session's stats JSON. Returns True if (re)computed."""
    session_dir = os.path.normpath(session_dir)
    label = "%s/%s" % (os.path.basename(os.path.dirname(session_dir)),
                       os.path.basename(session_dir))
    jpath = stats_json_path(session_dir)
    parts = find_parts(session_dir)
    if not force and os.path.isfile(jpath):
        try:
            with open(jpath) as f:
                prev = json.load(f)
            # The fingerprint covers only the HDF5 parts (and must stay that
            # way -- the training manifest freezes on it), so also compare the
            # result.txt content: offline re-judging updates the vision verdict
            # without touching the parts, and the report must pick that up.
            if (prev.get("schema_version") == SCHEMA_VERSION
                    and prev.get("source", {}).get("fingerprint") == fingerprint(parts)
                    and prev.get("result") == jsonable(read_result(session_dir, []))):
                if not quiet:
                    print("skip (up to date): %s" % label)
                return False
        except (OSError, ValueError):
            pass  # unreadable previous JSON -> recompute
    t0 = time.time()
    stats = build_episode_stats(session_dir, image_stride)
    write_json_atomic(jpath, stats)
    if not quiet:
        print("stats: %s -> %s (%.1fs, %d frames, warnings: %d)"
              % (label, jpath, time.time() - t0,
                 stats["frames"], len(stats["warnings"])))
    return True


# ---------------------------------------------------------------------------
# dataset aggregation + report
# ---------------------------------------------------------------------------

def _rate(succ, fail):
    n = succ + fail
    return (succ / n) if n else None


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _std(vals):
    """Sample std (ddof=1); None if <2 finite values, 0.0 if exactly one."""
    vals = [v for v in vals if v is not None]
    if len(vals) >= 2:
        return float(np.std(vals, ddof=1))
    if len(vals) == 1:
        return 0.0
    return None


def _median(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def summarize_rows(rows):
    """Dataset / per-task overview aggregates for the report header.

    Script success and script↔vision agreement are intentionally omitted:
    pipeline script exit is nearly always success and is not a useful overview
    signal; vision success is the task-outcome metric that matters here.
    """
    v_succ = sum(1 for r in rows if r["vision"] == "success")
    v_fail = sum(1 for r in rows if r["vision"] == "failed")
    smooth = [r["smoothness_score"] for r in rows]
    aesth = [r["aesthetics"] for r in rows]
    sharp = [r["sharpness_mean"] for r in rows]
    bright = [r["brightness_mean"] for r in rows]
    jerk = [r["rms_jerk"] for r in rows]
    disc = [r["discontinuities"] for r in rows]
    dur = [r["duration_s"] for r in rows]
    return {
        "episodes": len(rows),
        "success_rate_vision": _rate(v_succ, v_fail),
        "unknown_vision": sum(1 for r in rows if r["vision"] not in ("success", "failed")),
        "mean_smoothness": _mean(smooth),
        "std_smoothness": _std(smooth),
        "median_smoothness": _median(smooth),
        "mean_aesthetics": _mean(aesth),
        "std_aesthetics": _std(aesth),
        "mean_sharpness": _mean(sharp),
        "std_sharpness": _std(sharp),
        "mean_brightness": _mean(bright),
        "std_brightness": _std(bright),
        "mean_rms_jerk": _mean(jerk),
        "std_rms_jerk": _std(jerk),
        "mean_discontinuities": _mean(disc),
        "mean_duration_s": _mean(dur),
        "std_duration_s": _std(dur),
        "median_duration_s": _median(dur),
        "total_duration_s": sum(r["duration_s"] or 0 for r in rows),
        "total_bytes": sum(r["bytes"] or 0 for r in rows),
    }


def load_training_summary(dataset_dir):
    """Compact summary of training runs + policy evals for the report's scaling
    curve. Reads <dataset>/training/{training_runs.json,eval_results.json}
    (written by transfer_and_train.sh / eval_policy.sh). Returns None when no
    training data exists so pre-training datasets render unchanged; corrupt
    files are warn-only."""
    tdir = os.path.join(dataset_dir, "training")

    def _load(name, key):
        path = os.path.join(tdir, name)
        if not os.path.isfile(path):
            return []
        try:
            with open(path) as f:
                return json.load(f).get(key) or []
        except (OSError, ValueError) as e:
            print("Warning: unreadable %s: %s" % (path, e), file=sys.stderr)
            return []

    runs = _load("training_runs.json", "runs")
    evals = _load("eval_results.json", "evals")
    if not runs and not evals:
        return None

    run_rows, best_loss = [], {}
    for r in runs:
        res = r.get("result") or {}
        run_rows.append({
            "run_id": r.get("run_id"), "milestone": r.get("milestone"),
            "status": r.get("status"), "num_episodes": r.get("num_episodes"),
            "best_val_loss": res.get("best_val_loss"),
            "started_at": r.get("started_at"), "finished_at": r.get("finished_at"),
        })
        if r.get("status") == "done" and r.get("milestone") is not None:
            best_loss[r["milestone"]] = res.get("best_val_loss")

    curve = {}
    for e in evals:  # later entries overwrite: newest finalized eval wins
        m = e.get("milestone")
        if m is None or not e.get("finalized"):
            continue
        curve[m] = {
            "milestone": m, "success_rate": e.get("success_rate"),
            "successes": e.get("successes"), "trials": e.get("num_trials"),
            "unknowns": e.get("unknowns"), "eval_id": e.get("eval_id"),
            "run_id": e.get("run_id"), "evaluated_at": e.get("generated_at"),
            "best_val_loss": best_loss.get(m),
        }
    return {"runs": run_rows, "curve": [curve[m] for m in sorted(curve)]}


def aggregate_dataset(dataset_dir, embed_series_max):
    """Build dataset_stats.json content from all per-session JSONs."""
    ep_jsons = sorted(glob.glob(os.path.join(dataset_dir, "*", "*", "stats.json")))
    loaded = []
    for p in ep_jsons:
        # task dir is two levels up from the stats.json; skip non-task trees
        task_entry = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if task_entry in ("stats", "training") or task_entry.startswith("."):
            continue
        try:
            with open(p) as f:
                loaded.append((os.path.getmtime(p), json.load(f)))
        except (OSError, ValueError):
            continue
    # newest N episodes keep their full chart series embedded in the report
    embed = {id(s) for _, s in
             sorted(loaded, key=lambda x: x[0], reverse=True)[:embed_series_max]}

    tasks = {}
    for _, s in loaded:
        iq = s.get("image_quality") or {}
        cams = iq.get("cameras") or {}
        sm = s.get("smoothness") or {}
        rj = sm.get("rms_jerk") or {}
        row = {
            "session": s.get("session"),
            "frames": s.get("frames"),
            "duration_s": s.get("duration_s"),
            "result": (s.get("result") or {}).get("script") or "unknown",
            "vision": (s.get("result") or {}).get("vision"),
            "smoothness_score": sm.get("score"),
            "rms_jerk": rj.get("overall"),
            "discontinuities": sm.get("discontinuity_count"),
            "aesthetics": (s.get("aesthetics") or {}).get("model_score")
                          or (s.get("aesthetics") or {}).get("score"),
            "sharpness_mean": _mean([(c.get("sharpness") or {}).get("mean")
                                     for c in cams.values()]),
            "brightness_mean": _mean([(c.get("brightness") or {}).get("mean")
                                      for c in cams.values()]),
            "bytes": (s.get("source") or {}).get("total_bytes"),
            "warnings": len(s.get("warnings") or []),
            "generated_at": s.get("generated_at"),
            "detail": s if id(s) in embed else None,
        }
        tasks.setdefault(s.get("task") or "unknown", []).append(row)

    all_rows = []
    tasks_out = {}
    for name, rows in sorted(tasks.items()):
        # session names are timestamps, so string sort == chronological order
        rows.sort(key=lambda r: (r["session"] or ""))
        tasks_out[name] = {"aggregate": summarize_rows(rows), "episodes": rows}
        all_rows.extend(rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_dir": os.path.abspath(dataset_dir),
        "overall": summarize_rows(all_rows),
        "tasks": tasks_out,
        "training": load_training_summary(dataset_dir),
    }


def regenerate_report(dataset_dir, embed_series_max, quiet):
    """Aggregate + render report.html under an exclusive lock (safe when several
    backgrounded per-episode runs finish at once)."""
    stats_dir = os.path.join(dataset_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    with open(os.path.join(stats_dir, ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = aggregate_dataset(dataset_dir, embed_series_max)
        write_json_atomic(os.path.join(stats_dir, "dataset_stats.json"), data)
        try:
            import stats_report
            html = stats_report.render(jsonable(data))
            write_text_atomic(os.path.join(stats_dir, "report.html"), html)
            if not quiet:
                print("report: %s" % os.path.join(stats_dir, "report.html"))
        except Exception as e:
            # JSON outputs above are already written; a report failure is not fatal
            print("Warning: report generation failed: %s" % e, file=sys.stderr)


# ---------------------------------------------------------------------------
# batch discovery
# ---------------------------------------------------------------------------

def discover_sessions(task_dir):
    """Session dirs (one per recording) inside a task dir: any subdir holding
    episode part files or a result.txt (so zero-frame sessions are included).
    Old flat-layout episode files directly in the task dir are ignored with a
    warning -- the session layout is a clean break."""
    flat = (glob.glob(os.path.join(task_dir, "episode_*_part_*.hdf5"))
            + glob.glob(os.path.join(task_dir, "episode_*_result.txt")))
    if flat:
        print("Warning: %d old flat-layout episode file(s) in %s ignored "
              "(new layout: <task>/<timestamp>/...)"
              % (len(flat), task_dir), file=sys.stderr)
    sessions = []
    for entry in sorted(os.listdir(task_dir)):
        sdir = os.path.join(task_dir, entry)
        if not os.path.isdir(sdir) or entry == "stats" or entry.startswith("."):
            continue
        if find_parts(sdir) or os.path.isfile(os.path.join(sdir, "result.txt")):
            sessions.append(sdir)
    return sessions


def run_batch(dataset_dir, image_stride, force, quiet):
    changed = 0
    for entry in sorted(os.listdir(dataset_dir)):
        task_dir = os.path.join(dataset_dir, entry)
        if (not os.path.isdir(task_dir) or entry in ("stats", "training")
                or entry.startswith(".")):
            continue
        for sdir in discover_sessions(task_dir):
            try:
                if process_session(sdir, image_stride, force, quiet):
                    changed += 1
            except Exception as e:
                print("Warning: session %s/%s failed: %s"
                      % (entry, os.path.basename(sdir), e), file=sys.stderr)
    if not quiet:
        print("batch: %d session(s) (re)computed" % changed)
    return changed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-session statistics + HTML report for collected episodes.")
    ap.add_argument("--session-dir",
                    help="one recording's session folder (<dataset>/<task>/<timestamp>)")
    ap.add_argument("--dataset-dir",
                    help="dataset root (default: two levels above --session-dir)")
    ap.add_argument("--all", action="store_true", help="batch over the whole dataset")
    ap.add_argument("--report-only", action="store_true",
                    help="only re-aggregate + regenerate the HTML report")
    ap.add_argument("--force", action="store_true", help="recompute even if up to date")
    ap.add_argument("--image-stride", type=int, default=0,
                    help="sample every Nth frame for image metrics (0 = auto ~30 samples)")
    ap.add_argument("--embed-series-max", type=int, default=50,
                    help="newest N episodes keep full chart series in the report")
    ap.add_argument("--no-report", action="store_true",
                    help="skip aggregation + report regeneration")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    single = args.session_dir is not None
    if not (single or args.all or args.report_only):
        ap.error("need --session-dir, or --all, or --report-only")
    dataset_dir = args.dataset_dir or (
        os.path.dirname(os.path.dirname(os.path.abspath(args.session_dir)))
        if args.session_dir else None)
    if dataset_dir is None:
        ap.error("--dataset-dir is required with --all/--report-only")

    try:
        if single:
            process_session(args.session_dir, args.image_stride,
                            force=True, quiet=args.quiet)
        elif args.all:
            run_batch(dataset_dir, args.image_stride, args.force, args.quiet)
        if not args.no_report:
            regenerate_report(dataset_dir, args.embed_series_max, args.quiet)
    except Exception as e:
        if args.quiet:
            print("Error: %s" % e, file=sys.stderr)
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
