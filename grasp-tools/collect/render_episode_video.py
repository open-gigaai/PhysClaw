#!/usr/bin/env python3
# -- coding: UTF-8
"""Render one recorded session into a review video.

Reads every episode_0_part_M.hdf5 in a session folder
(<dataset>/<task>/<timestamp>/) and writes <task>_<timestamp>.mp4 into the
same folder: the 3 cameras side by side, at the true recording rate (30 fps
-- unlike the old visualize tool whose DT=0.02 played everything 1.67x fast).

Called automatically by the pipelines' collect_video hook (warn-only,
backgrounded); also usable by hand:

  python render_episode_video.py --session-dir <dataset>/<task>/<timestamp>
  python render_episode_video.py --session-dir ... --force   # re-render

Idempotent: skips when the mp4 is newer than every part file.
Dependencies: numpy, h5py, cv2 (conda env `aloha`).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import h5py
import cv2

FPS = 30
BATCH = 32  # frames per read batch (~30 MB for 3x 480x640x3)

_PART_RE = re.compile(r"episode_0_part_(\d+)\.hdf5$")


def find_parts(session_dir):
    paths = glob.glob(os.path.join(session_dir, "episode_0_part_*.hdf5"))
    def idx(p):
        m = _PART_RE.search(p)
        return int(m.group(1)) if m else 0
    return sorted(paths, key=idx)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Session episode -> side-by-side mp4")
    ap.add_argument("--session-dir", required=True,
                    help="<dataset>/<task>/<timestamp> folder of one recording")
    ap.add_argument("--out", help="output path (default: <session>/<task>_<timestamp>.mp4)")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--force", action="store_true", help="re-render even if up to date")
    args = ap.parse_args(argv)

    session_dir = os.path.abspath(args.session_dir)
    session = os.path.basename(os.path.normpath(session_dir))
    task = os.path.basename(os.path.dirname(os.path.normpath(session_dir)))
    out = args.out or os.path.join(session_dir, "%s_%s.mp4" % (task, session))

    parts = find_parts(session_dir)
    if not parts:
        print("Error: no episode_0_part_*.hdf5 in %s" % session_dir, file=sys.stderr)
        return 1
    newest_part = max(os.path.getmtime(p) for p in parts)
    if not args.force and os.path.isfile(out) and os.path.getmtime(out) > newest_part:
        print("skip (up to date): %s" % out)
        return 0

    with h5py.File(parts[0], "r") as f:
        cameras = sorted(f["observations/images"].keys())
        _, H, W, _ = f["observations/images"][cameras[0]].shape

    tmp = out + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W * len(cameras), H))
    if not writer.isOpened():
        print("Error: cannot open video writer for %s" % tmp, file=sys.stderr)
        return 1

    total = 0
    for p in parts:
        with h5py.File(p, "r") as f:
            dsets = [f["observations/images"][cam] for cam in cameras]
            T = len(dsets[0])
            for i in range(0, T, BATCH):
                n = min(BATCH, T - i)
                batch = np.concatenate([d[i:i + n] for d in dsets], axis=2)
                for t in range(n):
                    # stored channel order matches the existing visualize tool:
                    # swap to BGR for VideoWriter
                    writer.write(batch[t][:, :, [2, 1, 0]])
                total += n
    writer.release()
    os.replace(tmp, out)
    print("video: %d frames @ %d fps -> %s" % (total, args.fps, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
