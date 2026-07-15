#!/usr/bin/env python3
# -- coding: UTF-8
"""Serve the dataset stats report with live episode deletion.

The report (stats/report.html) is a static, self-contained page and cannot
touch the filesystem on its own. This tiny stdlib HTTP server hosts it and adds:
  POST /api/delete                     remove episode folders + regenerate report
  GET  /api/video?task=&session=       stream the session's review video, with
                                       HTTP Range support. Renders the mp4 on
                                       demand (render_episode_video.py) if missing
                                       and transcodes it to browser-playable H.264
                                       (cached as <task>_<session>.web.mp4) when
                                       ffmpeg is available.

Run it from the conda env that has episode_stats.py's deps (numpy/h5py/cv2),
because deletion regenerates the report via `episode_stats.py --report-only`:

  conda activate physclaw-0
  python serve_report.py                       # serves ../collected_data
  python serve_report.py --dataset-dir <dir> --port 8000

Then open the printed URL (http://127.0.0.1:8000/) in a browser. Opening the
file directly with file:// still works but disables the delete buttons.

Deletion is irreversible: each request runs `rm -rf` on
<dataset_dir>/<task>/<session>. Requests are validated so <task>/<session> must
be a real episode folder directly under the dataset dir (no path traversal, and
the reserved "stats"/"training" dirs are refused).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EPISODE_STATS = os.path.join(SCRIPT_DIR, "episode_stats.py")
RENDER_VIDEO = os.path.join(SCRIPT_DIR, "render_episode_video.py")

# Serialize deletions so two concurrent requests can't race the report rebuild.
_DELETE_LOCK = threading.Lock()
# Serialize on-demand video renders (one heavy ffmpeg-less cv2 encode at a time).
_RENDER_LOCK = threading.Lock()


FFMPEG = shutil.which("ffmpeg")


def session_video_path(session_dir, task, session):
    """Path of the side-by-side review mp4 render_episode_video.py writes for a
    session (matches its default: <session>/<task>_<session>.mp4). It is encoded
    with the mp4v (MPEG-4 Part 2) codec, which most browsers cannot decode."""
    return os.path.join(session_dir, "%s_%s.mp4" % (task, session))


def web_video_path(session_dir, task, session):
    """Browser-playable H.264 copy served to <video>, cached next to the source."""
    return os.path.join(session_dir, "%s_%s.web.mp4" % (task, session))


def safe_session_dir(dataset_dir, task, session):
    """Resolve <dataset_dir>/<task>/<session>, or None if it is not a valid,
    existing episode folder sitting directly under the dataset dir."""
    if not isinstance(task, str) or not isinstance(session, str):
        return None
    for name in (task, session):
        if not name or name in (".", "..") or "/" in name or "\\" in name \
                or "\x00" in name:
            return None
    if task in ("stats", "training") or task.startswith("."):
        return None
    base = os.path.realpath(dataset_dir)
    sdir = os.path.realpath(os.path.join(base, task, session))
    # must be exactly two levels below the dataset root: <base>/<task>/<session>
    if os.path.dirname(os.path.dirname(sdir)) != base:
        return None
    if not os.path.isdir(sdir):
        return None
    return sdir


def regenerate_report(dataset_dir):
    """Re-aggregate per-session stats and re-render report.html (fast: no
    per-episode recompute). Raises CalledProcessError on failure."""
    subprocess.run(
        [sys.executable, EPISODE_STATS, "--dataset-dir", dataset_dir,
         "--report-only", "--quiet"],
        cwd=SCRIPT_DIR, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def make_handler(dataset_dir, stats_dir):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=stats_dir, **kw)

        def log_message(self, fmt, *args):  # concise console logging
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def end_headers(self):
            # never cache the report: it is rewritten after every deletion
            if self.path in ("/", "/report.html"):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _video_target(self):
            """(session_dir, task, session) for /api/video?task=&session=, else None."""
            parsed = urlparse(self.path)
            if parsed.path != "/api/video":
                return None
            qs = parse_qs(parsed.query)
            task = (qs.get("task") or [None])[0]
            session = (qs.get("session") or [None])[0]
            sdir = safe_session_dir(dataset_dir, task, session)
            return (sdir, task, session) if sdir else (None, task, session)

        def do_GET(self):
            target = self._video_target()
            if target is not None:
                return self._serve_video(*target)
            if self.path == "/":
                self.path = "/report.html"
            return super().do_GET()

        def do_HEAD(self):
            target = self._video_target()
            if target is not None:
                return self._serve_video(*target)
            return super().do_HEAD()

        def _run(self, cmd, err):
            """Run a subprocess; on failure send a 500 JSON error and return False."""
            try:
                subprocess.run(cmd, cwd=SCRIPT_DIR, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                return True
            except subprocess.CalledProcessError as e:
                out = (e.stdout or b"").decode("utf-8", "replace")
                self._json(500, {"ok": False, "error": err,
                                 "detail": out[-800:] or str(e)})
                return False
            except OSError as e:
                self._json(500, {"ok": False, "error": str(e)})
                return False

        def _ensure_playable(self, session_dir, task, session):
            """Return a browser-playable mp4 path for the session, rendering the
            source and transcoding to H.264 as needed. On error sends the HTTP
            response itself and returns None."""
            src = session_video_path(session_dir, task, session)
            web = web_video_path(session_dir, task, session)

            def web_fresh():
                return os.path.isfile(web) and (
                    not os.path.isfile(src)
                    or os.path.getmtime(web) >= os.path.getmtime(src))

            if web_fresh():
                return web
            with _RENDER_LOCK:
                if web_fresh():
                    return web
                if not os.path.isfile(src):
                    if not self._run([sys.executable, RENDER_VIDEO,
                                      "--session-dir", session_dir],
                                     "video render failed"):
                        return None
                if not os.path.isfile(src):
                    self._json(404, {"ok": False, "error": "video not available"})
                    return None
                if not FFMPEG:
                    # No transcoder: serve the raw mp4v render (may not play in
                    # some browsers, but better than nothing).
                    return src
                tmp = web + ".tmp.mp4"
                if not self._run([FFMPEG, "-y", "-i", src,
                                  "-c:v", "libx264", "-preset", "veryfast",
                                  "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                  "-an", tmp],
                                 "video transcode failed"):
                    return None
                os.replace(tmp, web)
                return web

        def _serve_video(self, session_dir, task, session):
            if session_dir is None:
                self._json(404, {"ok": False, "error": "not a valid episode folder"})
                return
            path = self._ensure_playable(session_dir, task, session)
            if path is None:
                return  # error response already sent
            self._send_file_range(path, "video/mp4")

        def _send_file_range(self, path, ctype):
            """Serve a file honoring a single HTTP Range header so browser <video>
            can stream and seek."""
            try:
                fsize = os.path.getsize(path)
                f = open(path, "rb")
            except OSError:
                self._json(404, {"ok": False, "error": "cannot open file"})
                return
            try:
                start, end, status = 0, fsize - 1, 200
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    try:
                        s, e = rng[len("bytes="):].split("-", 1)
                        if s.strip():
                            start = int(s)
                        if e.strip():
                            end = int(e)
                        if start < 0 or start > end or start >= fsize:
                            raise ValueError
                        end = min(end, fsize - 1)
                        status = 206
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", "bytes */%d" % fsize)
                        self.end_headers()
                        return
                length = end - start + 1
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                if status == 206:
                    self.send_header("Content-Range",
                                     "bytes %d-%d/%d" % (start, end, fsize))
                self.end_headers()
                if self.command == "HEAD":
                    return
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)
            finally:
                f.close()

        def _json(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/delete":
                self._json(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._json(400, {"ok": False, "error": "invalid JSON body"})
                return

            items = payload.get("items")
            if items is None and payload.get("task") and payload.get("session"):
                items = [{"task": payload["task"], "session": payload["session"]}]
            if not isinstance(items, list) or not items:
                self._json(400, {"ok": False, "error": "no items to delete"})
                return

            with _DELETE_LOCK:
                deleted, errors = [], []
                for it in items:
                    task = (it or {}).get("task")
                    session = (it or {}).get("session")
                    sdir = safe_session_dir(dataset_dir, task, session)
                    label = "%s/%s" % (task, session)
                    if sdir is None:
                        errors.append({"item": label,
                                       "error": "not a valid episode folder"})
                        continue
                    try:
                        shutil.rmtree(sdir)
                        deleted.append(label)
                        sys.stderr.write("deleted: %s\n" % sdir)
                    except OSError as e:
                        errors.append({"item": label, "error": str(e)})

                regen_ok, regen_err = True, None
                if deleted:
                    try:
                        regenerate_report(dataset_dir)
                    except subprocess.CalledProcessError as e:
                        regen_ok = False
                        out = (e.stdout or b"").decode("utf-8", "replace")
                        regen_err = out[-800:] or str(e)

            self._json(200, {"ok": not errors and regen_ok,
                             "deleted": deleted, "errors": errors,
                             "report_regenerated": regen_ok,
                             "regen_error": regen_err})

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir",
                    default=os.path.join(os.path.dirname(SCRIPT_DIR),
                                         "collected_data"),
                    help="dataset root containing stats/report.html "
                         "(default: %(default)s)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: %(default)s)")
    ap.add_argument("--port", type=int, default=8000,
                    help="port (default: %(default)s)")
    args = ap.parse_args(argv)

    dataset_dir = os.path.abspath(args.dataset_dir)
    stats_dir = os.path.join(dataset_dir, "stats")
    report = os.path.join(stats_dir, "report.html")
    if not os.path.isfile(report):
        ap.error("report not found: %s\n(run episode_stats.py first)" % report)
    if not os.path.isfile(EPISODE_STATS):
        ap.error("episode_stats.py not found next to serve_report.py")

    httpd = ThreadingHTTPServer((args.host, args.port),
                                make_handler(dataset_dir, stats_dir))
    url = "http://%s:%d/" % (args.host, args.port)
    print("serving %s" % report)
    print("dataset: %s" % dataset_dir)
    print("open:    %s" % url)
    print("(Ctrl-C to stop; deletions are irreversible)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
