#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Parse <bbox>x1 y1 x2 y2</bbox> from VLM output text and visualize on the image.

Example input:
[green gourd: <bbox>112 452 193 616</bbox>][white plate: <bbox>270 456 568 850</bbox>]
"""

import argparse
import os
import re
import sys

import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
DEFAULT_IMAGE = os.environ.get(
    "PHYSGNOSIS_BBOX_IMAGE",
    os.path.join(_REPO_ROOT, "grasp-tools", "example_data", "color.png"),
)

# [label: <bbox>x1 y1 x2 y2</bbox>]
BBOX_PATTERN = re.compile(
    r"\[([^:\]]+):\s*<bbox>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</bbox>\]",
    re.IGNORECASE,
)

COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (128, 0, 255),
    (255, 0, 0),
]


def parse_bboxes(text):
    """Parse model output; return [(label, x1, y1, x2, y2), ...]."""
    results = []
    for match in BBOX_PATTERN.finditer(text):
        label = match.group(1).strip()
        x1, y1, x2, y2 = map(int, match.group(2, 3, 4, 5))
        results.append((label, x1, y1, x2, y2))
    return results


def scale_bboxes(bboxes, image_shape, coord_scale):
    """Scale model coordinates into image pixel space."""
    h, w = image_shape[:2]
    if coord_scale is None or coord_scale <= 0:
        max_coord = max(
            (max(x1, y1, x2, y2) for _, x1, y1, x2, y2 in bboxes),
            default=0,
        )
        if max_coord <= max(w, h):
            return bboxes
        coord_scale = 1000

    scaled = []
    for label, x1, y1, x2, y2 in bboxes:
        scaled.append((
            label,
            int(round(x1 * w / coord_scale)),
            int(round(y1 * h / coord_scale)),
            int(round(x2 * w / coord_scale)),
            int(round(y2 * h / coord_scale)),
        ))
    return scaled


def draw_bboxes(image, bboxes):
    """Draw bboxes and labels on the image."""
    vis = image.copy()
    h, w = vis.shape[:2]

    for i, (label, x1, y1, x2, y2) in enumerate(bboxes):
        color = COLORS[i % len(COLORS)]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        text = label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
        cv2.putText(vis, text, (x1 + 2, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return vis


def read_text(args):
    if args.text:
        return args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Parse bbox from LLM response and visualize on image"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help=f"Input image path (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="LLM response text containing bbox tags",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default="",
        help="Read LLM response from a text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output image path (default: <image_stem>_bbox_vis.jpg next to input)",
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1000,
        help="Model coord normalize scale, default 1000; set 0 to auto-detect from image size",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display result in a window",
    )
    args = parser.parse_args()

    text = read_text(args)
    if not text.strip():
        print("Error: provide model output text via --text, --text-file, or stdin.", file=sys.stderr)
        return 1

    bboxes = parse_bboxes(text)
    if not bboxes:
        print("Error: no bbox found in the text.", file=sys.stderr)
        print("Expected format: [object name: <bbox>x1 y1 x2 y2</bbox>]", file=sys.stderr)
        return 1

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image: {args.image}", file=sys.stderr)
        return 1

    coord_scale = None if args.coord_scale == 0 else args.coord_scale
    bboxes = scale_bboxes(bboxes, image.shape, coord_scale)

    print(f"Image: {args.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"Parsed {len(bboxes)} bbox(es):")
    for label, x1, y1, x2, y2 in bboxes:
        print(f"  - {label}: ({x1}, {y1}) -> ({x2}, {y2})")

    vis = draw_bboxes(image, bboxes)

    if args.output:
        out_path = args.output
    else:
        stem, _ = os.path.splitext(args.image)
        out_path = f"{stem}_bbox_vis.jpg"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"Visualization saved: {out_path}")

    if args.show:
        cv2.imshow("bbox visualization", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# python visualize_bbox.py --image $PHYSGNOSIS_ROOT/skills/understand-three-view-images/tmp/front_preview.jpg --text "[green fruit: <bbox>209 520 289 689</bbox>]"