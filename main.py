"""main.py – CLI entry-point
================================
Run the **Computational-Zoomy × ULR** pipeline end-to-end from the shell.

Assumes `hole_filler_weighted.py` is in the same folder (importable).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fillhole import create_composited_zoom  # type: ignore

def main() -> None:
    p = argparse.ArgumentParser("CZ-ULR hole-filling pipeline")

    p.add_argument("--img1",      required=True, type=Path, help="foreground image #1 (PNG/JPG)")
    p.add_argument("--depth1",    required=True, type=Path, help="NumPy .npz depth of img1")
    p.add_argument("--img2",      required=True, type=Path, help="background image #2")
    p.add_argument("--depth2",    required=True, type=Path, help="NumPy .npz depth of img2")

    p.add_argument("--stack_dir", required=True, type=Path,
                   help="COLMAP folder containing cameras.txt / images.txt / images/…")

    p.add_argument("--out",       default="zoom.mp4", type=Path, help="output video / image filename")
    p.add_argument("--frames",    default=30, type=int,   help="number of zoom frames")
    p.add_argument("--dolly_z",   default=20.0, type=float, help="dolly-plane depth (same units as depth map)")

    args = p.parse_args()

    create_composited_zoom(
        img1_path=str(args.img1),
        depth1_path=str(args.depth1),
        img2_path=str(args.img2),
        depth2_path=str(args.depth2),
        stack_dir=Path(args.stack_dir),
        n_frames=args.frames,
        dolly_plane_depth=args.dolly_z,
        video_name=str(args.out),
    )


if __name__ == "__main__":
    main()