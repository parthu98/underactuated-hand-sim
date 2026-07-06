"""Stack 2+ videos side by side (horizontally) into one comparison video.

Scales every input to the same height (aspect ratio preserved) and
horizontally concatenates them via ffmpeg's `hstack` filter, in the order
given on the command line. Stops at the shortest input's length. Uses the
same standalone ffmpeg (imageio-ffmpeg, libx264/CRF) as the rest of this
pipeline -- no system ffmpeg install required.

Usage:
    python make_side_by_side.py --videos a.mp4 b.mp4 c.mp4 --out compare.mp4
    python make_side_by_side.py --videos a.mp4 b.mp4 c.mp4 --out compare.mp4 --height 480
"""

from __future__ import annotations

from _bootstrap import ensure_venv
ensure_venv()

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

CONFIG = {
    "height": 480,   # per-pane scaled height (px); width auto, aspect preserved
    "crf": 18,       # libx264 CRF -- lower = higher quality/bigger file
    "preset": "medium",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", nargs="+", required=True, metavar="VIDEO",
                    help="2 or more video paths to place side by side, left to right.")
    p.add_argument("--out", required=True, help="Output video path.")
    p.add_argument("--height", type=int, default=CONFIG["height"],
                    help=f"Per-pane scaled height in px (default: {CONFIG['height']}).")
    p.add_argument("--crf", type=int, default=CONFIG["crf"],
                    help=f"libx264 CRF (default: {CONFIG['crf']}).")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.videos) < 2:
        raise SystemExit("--videos needs at least 2 paths.")
    for v in args.videos:
        if not Path(v).is_file():
            raise SystemExit(f"No such video: {v}")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    n = len(args.videos)

    scale_chain = "".join(f"[{i}:v]scale=-2:{args.height}[v{i}];" for i in range(n))
    stack_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_complex = f"{scale_chain}{stack_inputs}hstack=inputs={n}:shortest=1[out]"

    cmd = [ffmpeg, "-y"]
    for v in args.videos:
        cmd += ["-i", str(v)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", str(args.crf), "-preset", CONFIG["preset"],
        "-pix_fmt", "yuv420p",
        str(args.out),
    ]

    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
