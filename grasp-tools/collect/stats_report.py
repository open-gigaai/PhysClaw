#!/usr/bin/env python3
# -- coding: UTF-8
"""Render dataset_stats.json into the self-contained HTML report.

Used by episode_stats.py (import stats_report; stats_report.render(data)) and
standalone:  python stats_report.py --dataset-dir <dir>
"""
import argparse
import json
import os
import sys

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "report_template.html")
DATA_TOKEN = "/*__DATA__*/{}"


def render(data):
    """Return the report HTML with `data` (a JSON-safe dict) embedded."""
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    if DATA_TOKEN not in html:
        raise RuntimeError("data token missing from template: %s" % TEMPLATE)
    # \/ keeps any '</script>' inside JSON strings from terminating the script tag
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=False).replace("</", "<\\/")
    return html.replace(DATA_TOKEN, blob)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True,
                    help="dataset root containing stats/dataset_stats.json")
    args = ap.parse_args(argv)
    stats_dir = os.path.join(args.dataset_dir, "stats")
    with open(os.path.join(stats_dir, "dataset_stats.json")) as f:
        data = json.load(f)
    out = os.path.join(stats_dir, "report.html")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(render(data))
    os.replace(tmp, out)
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
