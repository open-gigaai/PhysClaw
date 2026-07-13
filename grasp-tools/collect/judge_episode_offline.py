#!/usr/bin/env python3
# -- coding: UTF-8
"""Judge task success of an ALREADY-RECORDED episode from its own frames.

The pipeline's script-level "success" only means the motions completed --
a grasp that missed the object still exits 0. The live VLM verify photographs
the scene after the task; this tool does the same judgment OFFLINE using the
last recorded frame(s) of the episode HDF5, so you can (re-)judge episodes
without the robot, its cameras, or the arm.

  python judge_episode_offline.py --session-dir <dataset>/<task>/<ts> \
      [--prompt "Is the grape on the plate? ... YES or NO."] [--write] [--no-report]

- Default prompt is built from the task folder name (grasp_<object>).
- Sends the LAST frame of each camera (scene after the task) to the same
  Ark/Doubao endpoint as the live judge (reads key/model out of
  capture_and_understand_one_view.py so there is one source of truth).
- --write updates the "vision:"/"vision_answer:" lines in the session's
  result.txt, then refreshes the session's stats.json and the dataset report
  (<dataset>/stats/report.html + dataset_stats.json) via episode_stats, so
  stats and training-set qualification pick the new verdict up immediately.
  --no-report skips that refresh (for batch loops: re-judge every session
  with --no-report, then run episode_stats.py --dataset-dir D --all once).

Needs network access to ark.cn-beijing.volces.com (works on the robot; most
non-CN networks cannot reach it). No ROS needed.
"""
import argparse
import base64
import glob
import os
import re
import sys

import h5py
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
_ENV_UNDERSTAND = os.environ.get("UNDERSTAND_SH", "").strip()
_ENV_JUDGE = ""
if _ENV_UNDERSTAND:
    _ENV_JUDGE = os.path.join(
        os.path.dirname(_ENV_UNDERSTAND), "capture_and_understand_one_view.py"
    )
JUDGE_CANDIDATES = [
    _ENV_JUDGE,
    os.path.join(
        _REPO_ROOT,
        "skills",
        "understand-three-view-images",
        "scripts",
        "capture_and_understand_one_view.py",
    ),
    os.path.join(HERE, "skills", "understand-three-view-images", "scripts",
                 "capture_and_understand_one_view.py"),
]
JUDGE_CANDIDATES = [p for p in JUDGE_CANDIDATES if p]

SYSTEM_PROMPT = ("Images show the post-task scene captured by cameras on the robot. "
                 "Start the answer with a single English word YES or NO (uppercase, alone), "
                 "then briefly explain in one sentence."
                 "Your reply MUST start with exactly one English word: YES or NO. ")

_PART_RE = re.compile(r"episode_0_part_(\d+)\.hdf5$")


def load_judge_config():
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENGINE_API_KEY") or ""
    base_url = os.environ.get(
        "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
    )
    model = os.environ.get("ARK_MODEL", "doubao-seed-2-0-mini-260428")
    if api_key:
        return api_key, base_url, model
    for path in JUDGE_CANDIDATES:
        if not os.path.isfile(path):
            continue
        src = open(path).read()
        key_m = re.search(
            r'API_KEY\s*=\s*(?:os\.environ\.get\([^)]+\)\s*or\s*)*[\'"]([^\'"]+)[\'"]',
            src,
        )
        # Prefer env-style scripts: fall through if key empty placeholder
        url_m = re.search(r'ARK_BASE_URL["\']?\s*,\s*[\'"]([^\'"]+)[\'"]', src) or re.search(
            r'BASE_URL\s*=\s*os\.environ\.get\([^,]+,\s*[\'"]([^\'"]+)[\'"]', src
        )
        model_m = re.search(
            r'ARK_MODEL["\']?\s*,\s*[\'"]([^\'"]+)[\'"]', src
        ) or re.search(r'model\s*=\s*os\.environ\.get\([^,]+,\s*[\'"]([^\'"]+)[\'"]', src)
        if url_m:
            base_url = url_m.group(1)
        if model_m:
            model = model_m.group(1)
        if key_m and key_m.group(1) and "YOUR_API" not in key_m.group(1):
            return key_m.group(1), base_url, model
        # Script found; env still required
        break
    raise SystemExit(
        "Error: set ARK_API_KEY (or VOLCENGINE_API_KEY); "
        "judge script candidates: %s" % ", ".join(JUDGE_CANDIDATES)
    )


def last_frames(session_dir):
    """{camera: last frame} from the final part file."""
    parts = sorted(glob.glob(os.path.join(session_dir, "episode_0_part_*.hdf5")),
                   key=lambda p: int(_PART_RE.search(p).group(1)))
    if not parts:
        raise SystemExit("Error: no episode_0_part_*.hdf5 in %s" % session_dir)
    out = {}
    with h5py.File(parts[-1], "r") as f:
        for cam in sorted(f["observations/images"].keys()):
            out[cam] = f["observations/images"][cam][-1]
    return out


def refresh_stats(session_dir, dataset_dir):
    """Refresh the session's stats.json + the dataset report after a verdict
    change. Warn-only (same contract as collect_stats in the pipelines): the
    result.txt update must stand even if stats or the report fail."""
    try:
        sys.path.insert(0, HERE)
        import episode_stats
        # up-to-date check compares the stored result against result.txt, so
        # an unchanged verdict is a cheap skip, a changed one recomputes
        episode_stats.process_session(session_dir, image_stride=0,
                                      force=False, quiet=True)
        episode_stats.regenerate_report(dataset_dir, embed_series_max=50,
                                        quiet=False)
    except Exception as e:
        print("Warning: stats/report refresh failed: %s -- run "
              "episode_stats.py --dataset-dir %s --all to refresh manually"
              % (e, dataset_dir), file=sys.stderr)


def verdict_of(text):
    """Same rules as collect_verify / eval_policy."""
    if re.search(r"Request failed|Unable to get|Image conversion failed|Image encoding failed|not ready|Traceback", text):
        return "unknown"
    if re.search(r"(^|[^A-Za-z])YES([^A-Za-z]|$)", text, re.I):
        return "success"
    if re.search(r"(^|[^A-Za-z])NO([^A-Za-z]|$)", text, re.I):
        return "failed"
    return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline VLM judge on recorded frames")
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--prompt", default="",
                    help="YES/NO question (default: built from task folder name)")
    ap.add_argument("--write", action="store_true",
                    help="update vision:/vision_answer: lines in result.txt "
                         "and refresh stats.json + the dataset report")
    ap.add_argument("--dataset-dir", default="",
                    help="dataset root for the report refresh "
                         "(default: two levels above --session-dir)")
    ap.add_argument("--no-report", action="store_true",
                    help="with --write: skip the stats.json/report.html refresh")
    args = ap.parse_args(argv)

    session_dir = os.path.normpath(os.path.abspath(args.session_dir))
    task = os.path.basename(os.path.dirname(session_dir))
    obj = task.replace("grasp_", "").replace("_", " ") or "object"
    question = args.prompt.strip() or (
        "The robot arm just tried to pick up the %s and place it at the target "
        "location (a basket or marked spot). This image shows the scene AFTER "
        "the attempt. Is the %s now at the target location and no longer at its "
        "original position? Start your reply with exactly one word: YES or NO. "
        "Then one short sentence of explanation." % (obj, obj))

    api_key, base_url, model = load_judge_config()
    from openai import OpenAI  # deferred: frames load even without the SDK
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=1)

    frames = last_frames(session_dir)
    content = [{"type": "text", "text": SYSTEM_PROMPT + question}]
    for cam, img in frames.items():
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1])  # stored RGB -> BGR for JPEG
        if not ok:
            print("Warning: encode failed for %s" % cam, file=sys.stderr)
            continue
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64,"
                   + base64.b64encode(buf.tobytes()).decode()}})
    print("judging %s/%s with %d camera frame(s)..."
          % (task, os.path.basename(session_dir), len(content) - 1))

    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": content}],
            max_tokens=1024)
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        answer = "Request failed, error: %s" % e
    verdict = verdict_of(answer)
    print("answer: %s" % answer[:300])
    print("vision_verdict: %s" % verdict)

    if args.write:
        result_path = os.path.join(session_dir, "result.txt")
        lines = []
        if os.path.isfile(result_path):
            # errors="replace": re-judging must work on a result.txt poisoned
            # by the old byte-level truncation (torn UTF-8 char)
            with open(result_path, errors="replace") as f:
                lines = [ln.rstrip("\n") for ln in f
                         if not ln.startswith(("vision:", "vision_answer:"))]
        lines += ["vision: %s" % verdict,
                  "vision_answer: %s (offline re-judge)" % answer[:300].replace("\n", " ")]
        tmp = result_path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, result_path)
        print("updated: %s" % result_path)
        if args.no_report:
            print("stats/report refresh skipped (--no-report); run "
                  "episode_stats.py --dataset-dir <D> --all when done")
        else:
            dataset_dir = (os.path.abspath(args.dataset_dir) if args.dataset_dir
                           else os.path.dirname(os.path.dirname(session_dir)))
            refresh_stats(session_dir, dataset_dir)
    return 0 if verdict != "unknown" else 1


if __name__ == "__main__":
    sys.exit(main())
