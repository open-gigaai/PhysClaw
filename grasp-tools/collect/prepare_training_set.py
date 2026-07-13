#!/usr/bin/env python3
# -- coding: UTF-8
"""Build the ACT training set from collected episodes.

Each recording is a session folder <dataset>/<task>/<timestamp>/ (see the
collect pipelines). This selects "qualified" sessions (script success + vision
rule + optional smoothness filter), merges their chunked part files into
single ACT-style episode files with contiguous indices, and maintains a
manifest with a FROZEN ordering so every training milestone ("first N
episodes") is reproducible.

  <dataset_dir>/training/merged/episode_<i>.hdf5   merged, renumbered episodes
  <dataset_dir>/training/manifest.json             frozen ordering + provenance

Qualification of one session:
  - result.txt first line == "success"
  - vision rule: strict  -> "vision: success" required
                 lenient -> missing/unknown vision also passes (never "failed")
  - at least one readable HDF5 part with >= 1 frame
  - optional: smoothness.discontinuity_count <= --max-discontinuities
    (from the session's stats.json; missing stats -> included with a warning:
    result.txt is authoritative, stats are advisory)

Manifest entries are append-only. A fingerprint change or missing parts for an
already-manifested session is a hard error (exit 3) -- delete
<dataset_dir>/training/ to rebuild from scratch, accepting renumbering.

Usage:
  python prepare_training_set.py --dataset-dir D --count-only [--vision ...]
  python prepare_training_set.py --dataset-dir D [--limit N] [--force] ...

Dependencies: numpy, h5py (conda env `aloha`); imports episode_stats (sibling).
"""
import argparse
import fcntl
import json
import os
import sys
import time

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_stats  # find_parts / fingerprint / write_json_atomic / read_result / discover_sessions

MANIFEST_SCHEMA = 1
IMAGE_COPY_BATCH = 32  # frames per read/write batch when copying image datasets

SKIP_DIRS = {"stats", "training"}


def log(msg, quiet=False):
    if not quiet:
        print(msg)


def warn(msg):
    print("Warning: %s" % msg, file=sys.stderr)


def episode_frames(session_dir, parts):
    """Total frames of a session; prefers the (fingerprint-matching) stats JSON
    to avoid opening HDF5. Returns (frames, stats_json_or_None)."""
    jp = episode_stats.stats_json_path(session_dir)
    if os.path.isfile(jp):
        try:
            with open(jp) as f:
                prev = json.load(f)
            if (prev.get("source", {}) or {}).get("fingerprint") == \
                    episode_stats.fingerprint(parts):
                return int(prev.get("frames") or 0), prev
        except (OSError, ValueError):
            pass
    total = 0
    for p in parts:
        try:
            with h5py.File(p, "r") as f:
                total += len(f["observations/qpos"])
        except Exception as e:
            warn("unreadable part %s: %s" % (os.path.basename(p), e))
    return total, None


def qualify(session_dir, vision_mode, max_disc):
    """Returns candidate dict if the session qualifies, else None."""
    session_dir = os.path.normpath(session_dir)
    result = episode_stats.read_result(session_dir, [])
    if result["script"] != "success":
        return None
    vision = result.get("vision")
    if vision_mode == "strict":
        if vision != "success":
            return None
    else:  # lenient: a positive or absent verdict passes, an explicit fail doesn't
        if vision == "failed":
            return None
    parts = episode_stats.find_parts(session_dir)
    if not parts:
        return None
    frames, stats_json = episode_frames(session_dir, parts)
    if frames <= 0:
        return None
    sm = (stats_json or {}).get("smoothness") or {}
    disc = sm.get("discontinuity_count")
    task = os.path.basename(os.path.dirname(session_dir))
    session = os.path.basename(session_dir)
    if max_disc >= 0:
        if stats_json is None:
            warn("%s/%s: no stats JSON, discontinuity filter skipped"
                 % (task, session))
        elif disc is not None and disc > max_disc:
            return None
    result_path = os.path.join(session_dir, "result.txt")
    return {
        "task": task,
        "session": session,
        "parts": [os.path.basename(p) for p in parts],
        "fingerprint": episode_stats.fingerprint(parts),
        "frames": frames,
        "result": result["script"],
        "vision": vision,
        "stats": {
            "smoothness_score": sm.get("score"),
            "discontinuity_count": disc,
            "aesthetics": ((stats_json or {}).get("aesthetics") or {}).get("score"),
        },
        "_mtime": os.path.getmtime(result_path),
    }


def discover_candidates(dataset_dir, vision_mode, max_disc):
    out = []
    for entry in sorted(os.listdir(dataset_dir)):
        task_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(task_dir) or entry in SKIP_DIRS or entry.startswith("."):
            continue
        for sdir in episode_stats.discover_sessions(task_dir):
            c = qualify(sdir, vision_mode, max_disc)
            if c:
                out.append(c)
    # deterministic near-chronological order for NEW entries only; manifested
    # entries keep their frozen position forever
    out.sort(key=lambda c: (c["_mtime"], c["task"], c["session"]))
    return out


def load_manifest(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit("Error: corrupt manifest %s: %s (delete the training/ "
                         "dir to rebuild)" % (path, e))


def verify_frozen(dataset_dir, manifest):
    """Manifested sessions must still exist byte-identically (exit 3 if not)."""
    for e in manifest.get("episodes", []):
        session_dir = os.path.join(dataset_dir, e["task"], e["session"])
        parts = episode_stats.find_parts(session_dir)
        if not parts:
            print("Error: manifested session %s/%s has no part files anymore; "
                  "the frozen ordering is broken. Delete %s to rebuild."
                  % (e["task"], e["session"],
                     os.path.join(dataset_dir, "training")), file=sys.stderr)
            sys.exit(3)
        if episode_stats.fingerprint(parts) != e["fingerprint"]:
            print("Error: source parts changed for manifested session %s/%s "
                  "(fingerprint mismatch). Never silently renumbering. Delete "
                  "%s to rebuild." % (e["task"], e["session"],
                                      os.path.join(dataset_dir, "training")),
                  file=sys.stderr)
            sys.exit(3)


def merge_episode(dataset_dir, entry, action_shift, quiet):
    """Concatenate part files into training/merged/episode_<index>.hdf5 with the
    exact layout of collect_dataliyishan.save_data."""
    session_dir = os.path.join(dataset_dir, entry["task"], entry["session"])
    parts = episode_stats.find_parts(session_dir)
    out_path = os.path.join(dataset_dir, "training", "merged", entry["merged_file"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    small = {"observations/qpos": [], "observations/qvel": [],
             "observations/effort": [], "action": [], "base_action": []}
    cameras, has_depth, img_shape, depth_shape = None, False, None, None
    lengths = []
    for p in parts:
        with h5py.File(p, "r") as f:
            for k in small:
                small[k].append(f[k][()])
            lengths.append(len(small["observations/qpos"][-1]))
            if cameras is None:
                cameras = sorted(f["observations/images"].keys())
                img_shape = f["observations/images"][cameras[0]].shape[1:]
                has_depth = "images_depth" in f["observations"]
                if has_depth:
                    depth_shape = f["observations/images_depth"][cameras[0]].shape[1:]
    cat = {k: np.concatenate(v) for k, v in small.items()}
    T = len(cat["observations/qpos"])
    if action_shift > 0:
        # one-step-ahead achieved position as the action (mitigates action==qpos)
        idx = np.minimum(np.arange(T) + action_shift, T - 1)
        cat["action"] = cat["observations/qpos"][idx]

    tmp = out_path + ".tmp"
    t0 = time.time()
    with h5py.File(tmp, "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        obs = root.create_group("observations")
        image = obs.create_group("images")
        dsets = {}
        for cam in cameras:
            dsets[cam] = image.create_dataset(
                cam, (T,) + img_shape, dtype="uint8",
                chunks=(1,) + img_shape, compression="gzip",
                compression_opts=4, shuffle=True)
        depth_grp, depth_dsets = None, {}
        if has_depth:
            depth_grp = obs.create_group("images_depth")
            for cam in cameras:
                depth_dsets[cam] = depth_grp.create_dataset(
                    cam, (T,) + depth_shape, dtype="uint16",
                    chunks=(1,) + depth_shape, compression="gzip",
                    compression_opts=4, shuffle=True)
        obs.create_dataset("qpos", data=cat["observations/qpos"])
        obs.create_dataset("qvel", data=cat["observations/qvel"])
        obs.create_dataset("effort", data=cat["observations/effort"])
        root.create_dataset("action", data=cat["action"])
        root.create_dataset("base_action", data=cat["base_action"])

        off = 0
        for p, plen in zip(parts, lengths):
            with h5py.File(p, "r") as f:
                for cam in cameras:
                    src = f["observations/images"][cam]
                    for i in range(0, plen, IMAGE_COPY_BATCH):
                        n = min(IMAGE_COPY_BATCH, plen - i)
                        dsets[cam][off + i:off + i + n] = src[i:i + n]
                    if has_depth:
                        dsrc = f["observations/images_depth"][cam]
                        for i in range(0, plen, IMAGE_COPY_BATCH):
                            n = min(IMAGE_COPY_BATCH, plen - i)
                            depth_dsets[cam][off + i:off + i + n] = dsrc[i:i + n]
            off += plen
    os.replace(tmp, out_path)
    entry["frames"] = T
    entry["merged_bytes"] = os.path.getsize(out_path)
    entry["merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    log("merged: %s/%s -> %s (%d frames, %.1fs)"
        % (entry["task"], entry["session"], entry["merged_file"], T,
           time.time() - t0), quiet)
    return cameras


def merged_up_to_date(dataset_dir, entry):
    out_path = os.path.join(dataset_dir, "training", "merged", entry["merged_file"])
    return (os.path.isfile(out_path)
            and entry.get("merged_bytes") == os.path.getsize(out_path))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Qualify + merge episodes into the "
                                             "frozen ACT training set.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="merge only the first N manifest episodes (0 = all)")
    ap.add_argument("--count-only", action="store_true",
                    help="print qualified_count and change nothing")
    ap.add_argument("--vision", choices=["strict", "lenient"], default="strict")
    ap.add_argument("--max-discontinuities", type=int, default=-1)
    ap.add_argument("--action-shift", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-merge everything")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    dataset_dir = os.path.abspath(args.dataset_dir)
    training_dir = os.path.join(dataset_dir, "training")
    manifest_path = os.path.join(training_dir, "manifest.json")

    manifest = load_manifest(manifest_path)
    if manifest is not None:
        verify_frozen(dataset_dir, manifest)
        prev_q = manifest.get("qualify") or {}
        cur_q = {"vision": args.vision,
                 "max_discontinuities": args.max_discontinuities,
                 "action_shift": args.action_shift}
        if {k: prev_q.get(k) for k in cur_q} != cur_q:
            warn("qualification rules changed (%s -> %s); frozen entries are "
                 "kept, new rules apply to new episodes only" % (prev_q, cur_q))

    known = {(e["task"], e["session"])
             for e in (manifest or {}).get("episodes", [])}
    candidates = discover_candidates(dataset_dir, args.vision,
                                     args.max_discontinuities)
    fresh = [c for c in candidates if (c["task"], c["session"]) not in known]
    count = len(known) + len(fresh)

    if args.count_only:
        print("qualified_count: %d" % count)
        return 0

    os.makedirs(training_dir, exist_ok=True)
    with open(manifest_path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # re-load under the lock in case a concurrent prepare appended
        manifest = load_manifest(manifest_path) or {
            "schema_version": MANIFEST_SCHEMA,
            "dataset_dir": dataset_dir,
            "qualify": {"vision": args.vision,
                        "max_discontinuities": args.max_discontinuities,
                        "action_shift": args.action_shift},
            "camera_names": None, "max_frames": None, "episodes": [],
        }
        known = {(e["task"], e["session"]) for e in manifest["episodes"]}
        for c in candidates:
            if (c["task"], c["session"]) in known:
                continue
            c.pop("_mtime", None)
            c["index"] = len(manifest["episodes"])
            c["merged_file"] = "episode_%d.hdf5" % c["index"]
            c["merged_bytes"] = None
            c["merged_at"] = None
            manifest["episodes"].append(c)

        todo = manifest["episodes"]
        if args.limit > 0:
            todo = todo[:args.limit]
        cameras = manifest.get("camera_names")
        merged_n = 0
        for e in todo:
            if not args.force and merged_up_to_date(dataset_dir, e):
                continue
            cams = merge_episode(dataset_dir, e, args.action_shift, args.quiet)
            cameras = cameras or cams
            merged_n += 1
        manifest["camera_names"] = cameras
        frames = [e["frames"] for e in manifest["episodes"] if e.get("frames")]
        manifest["max_frames"] = max(frames) if frames else None
        manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        episode_stats.write_json_atomic(manifest_path, manifest)

    log("manifest: %d episode(s), %d merged this run -> %s"
        % (len(manifest["episodes"]), merged_n, manifest_path), args.quiet)
    print("qualified_count: %d" % len(manifest["episodes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
