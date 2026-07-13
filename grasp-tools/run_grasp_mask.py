"""AnyGrasp with precomputed object mask (BW image) and left/right arm selection.

Expects mask.png in data_dir (white=target object, black=background).
Run run_seg.py first to generate the mask from a text prompt.

This is the entry point used by run.sh. It delegates to run_grasp.run_grasp(),
which runs AnyGrasp on the full scene point cloud (collision detection includes
table/background), then keeps grasps projecting into the SAM3 mask. Saves
eef_pose_xyzrpy.npy for the best grasp after camera-up
canonicalization (180° about TCP approach when wrist camera/TCP+Z points down) and
oblique-approach filtering (ray from ~base xy at z=0.5m toward contact;
use --no_approach_filter for score-only selection).

Example:
  python run_grasp_mask.py \\
    --arm left \\
    --data_dir ./example_data \\
    --debug --top_down_grasp
"""

from __future__ import annotations

import os
import sys

if '--quiet' in sys.argv:
    os.environ.setdefault('OMP_NUM_THREADS', '16')
    import logging
    import warnings

    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.ERROR)

import argparse
import os
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from run_grasp import build_parser as build_base_parser, load_rgb_depth, parse_args as parse_base_args, run_grasp


DEFAULT_MASK_NAME = 'mask.png'


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = 'AnyGrasp with precomputed mask to select target object'
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return parse_base_args(argv)


def load_binary_mask(mask_path: str, *, quiet: bool = False) -> np.ndarray:
    """Load mask as (H, W) bool; True = target object (white)."""
    img = np.array(Image.open(mask_path))
    if img.ndim == 3:
        img = img[..., 0]
    foreground = img > 127
    h, w = foreground.shape
    n_fg = int(np.count_nonzero(foreground))
    if not quiet:
        print(f'Mask {mask_path}: {n_fg}/{h * w} foreground pixels ({100.0 * n_fg / (h * w):.1f}%)')
    if n_fg == 0:
        raise ValueError(f'Mask has no foreground pixels: {mask_path}')
    return foreground


def main() -> None:
    cfgs = parse_args()
    mask_path = os.path.join(cfgs.data_dir, DEFAULT_MASK_NAME)
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(
            f'Mask not found: {mask_path}. Run run_seg.py first to generate it.'
        )

    _, depths = load_rgb_depth(cfgs.data_dir)
    pixel_mask = load_binary_mask(mask_path, quiet=cfgs.quiet)
    if pixel_mask.shape != depths.shape[:2]:
        raise ValueError(
            f'Mask shape {pixel_mask.shape} != depth shape {depths.shape[:2]}'
        )

    # User mask already selects the object; skip network objectness mask.
    run_grasp(
        cfgs.data_dir,
        cfgs,
        pixel_mask=pixel_mask,
        apply_object_mask=False,
    )


if __name__ == '__main__':
    main()
