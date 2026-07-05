"""Step 2b: render the fingertip-trajectory video (a growing trail line) and
the end-of-motion workspace-area PNG, from a track_points.py cache.

The source video is only ever opened for READING -- the trail overlay is
written to a brand-new mp4 in its own folder, separate from the joint-overlay
video, and the raw file is never modified.

Usage:
    python render_tip_trajectory.py --video <path> --cache <path> \
        --out-video <path> --out-png <path> [--flexion-sign 1]
"""

from __future__ import annotations

from _bootstrap import ensure_venv
ensure_venv()

import argparse
from pathlib import Path

import cv2
import numpy as np

import geometry
from video_writer import FFmpegWriter

# ---------------------------------------------------------------------------
# CONFIG -- trail look, markers, and workspace-fill styling live here.
# ---------------------------------------------------------------------------
CONFIG = {
    "colormap": cv2.COLORMAP_VIRIDIS,   # trail color gradient across the sweep
    "trail_thickness": 3,
    "marker_radius": 7,
    "marker_color": (0, 255, 255),      # BGR -- current tip position each frame
    "start_color": (0, 200, 0),         # BGR green -- sweep start marker
    "end_color": (0, 0, 255),           # BGR red -- sweep end marker
    "workspace_fill_color": (200, 160, 40),  # BGR teal-ish
    "workspace_fill_alpha": 0.25,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.8,
    "font_thickness": 2,
    "text_outline_thickness": 4,
    "text_outline_color": (0, 0, 0),
    "text_color": (255, 255, 255),
    "codec": "libx264",
    "crf": 18,
    "preset": "medium",

    # motion-window detection (reused to pick the freeze frame / path extent)
    "motion_eps_deg": 0.15,
    "motion_min_run": 5,
}

_TIP_IDX = geometry.POINT_NAMES.index("tip")
_BASE_IDX = geometry.POINT_NAMES.index("base")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Path to the source mp4 (read-only).")
    p.add_argument("--cache", required=True, help="track_points.py .npz cache path.")
    p.add_argument("--out-video", required=True, help="Output tip-trajectory mp4 path.")
    p.add_argument("--out-png", required=True, help="Output workspace-area PNG path.")
    p.add_argument("--flexion-sign", type=int, default=1, choices=(1, -1))
    return p.parse_args()


def _build_colormap_lut(colormap):
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    return cv2.applyColorMap(ramp, colormap).reshape(256, 3)


def _put_text_outlined(img, text, org, config):
    cv2.putText(img, text, org, config["font"], config["font_scale"],
                config["text_outline_color"], config["text_outline_thickness"], cv2.LINE_AA)
    cv2.putText(img, text, org, config["font"], config["font_scale"],
                config["text_color"], config["font_thickness"], cv2.LINE_AA)


def render_trail_video(video_path, points, fps, out_video, freeze_idx, config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    if config["codec"] == "libx264":
        writer = FFmpegWriter(out_video, fps, (width, height), crf=config["crf"], preset=config["preset"])
    else:
        fourcc = cv2.VideoWriter_fourcc(*config["codec"])
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    lut = _build_colormap_lut(config["colormap"])
    tip = points[:, _TIP_IDX, :]
    n_frames = points.shape[0]

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    freeze_frame = None

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        pt_i = tuple(np.round(tip[i]).astype(int))
        if i > 0:
            color_idx = int(round(255 * i / max(n_frames - 1, 1)))
            color = tuple(int(c) for c in lut[color_idx])
            pt_prev = tuple(np.round(tip[i - 1]).astype(int))
            cv2.line(canvas, pt_prev, pt_i, color, config["trail_thickness"], cv2.LINE_AA)

        nonzero = np.any(canvas != 0, axis=2)
        frame[nonzero] = canvas[nonzero]
        cv2.circle(frame, pt_i, config["marker_radius"], config["marker_color"], -1, cv2.LINE_AA)
        writer.write(frame)

        if i == freeze_idx:
            freeze_frame = frame.copy()

    cap.release()
    writer.release()

    if freeze_frame is None:
        raise RuntimeError("Freeze frame index was never reached while rendering the trail video.")
    return freeze_frame, canvas


def build_workspace_png(freeze_frame, canvas, points, motion_start, motion_end, out_png, config):
    tip = points[:, _TIP_IDX, :]
    base_point = np.median(points[:, _BASE_IDX, :], axis=0)
    path = tip[motion_start:motion_end + 1]

    poly_for_fill = np.vstack([base_point[None, :], path, base_point[None, :]]).astype(np.int32)
    poly_for_area = np.vstack([base_point[None, :], path])

    overlay = freeze_frame.copy()
    cv2.fillPoly(overlay, [poly_for_fill], config["workspace_fill_color"])
    blended = cv2.addWeighted(overlay, config["workspace_fill_alpha"], freeze_frame,
                               1 - config["workspace_fill_alpha"], 0)

    nonzero = np.any(canvas != 0, axis=2)
    blended[nonzero] = canvas[nonzero]

    start_pt = tuple(np.round(tip[motion_start]).astype(int))
    end_pt = tuple(np.round(tip[motion_end]).astype(int))
    cv2.circle(blended, start_pt, config["marker_radius"] + 2, config["start_color"], -1, cv2.LINE_AA)
    cv2.circle(blended, end_pt, config["marker_radius"] + 2, config["end_color"], -1, cv2.LINE_AA)

    area_px2 = geometry.polygon_area(poly_for_area)
    _put_text_outlined(blended, f"Workspace area: {area_px2:,.0f} px^2", (10, 30), config)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, blended)
    return area_px2


def main():
    args = parse_args()
    cache = np.load(args.cache)
    if int(cache.get("schema_version", 1)) < 7:
        raise SystemExit(
            f"'{args.cache}' was written by an old, incompatible tracking scheme "
            f"(schema_version={int(cache.get('schema_version', 1))}, need >= 7). "
            "Re-run track_points.py on this video to regenerate the cache -- v1 "
            "(direct tip tracking), v2 (distal_anchor), v3 (color+PCA), v4/v5 "
            "(rotation-search template matching, which drifted at the ends of the "
            "sweep), and v6 (farthest-band geometric tip, which sat a few px "
            "inboard of the true corner) are all obsolete and cannot be "
            "reinterpreted safely. Tip: `track_points.py --from-cache <old.npz>` "
            "re-derives just the tip column from an existing cache's tracks "
            "without re-clicking."
        )
    points = cache["points"]
    fps = float(cache["fps"])

    angles = geometry.compute_joint_angles(points, flexion_sign=args.flexion_sign)
    motion_start, motion_end = geometry.detect_motion_window(
        angles, eps_deg=CONFIG["motion_eps_deg"], min_run=CONFIG["motion_min_run"]
    )

    freeze_frame, canvas = render_trail_video(args.video, points, fps, args.out_video, motion_end, CONFIG)
    area_px2 = build_workspace_png(freeze_frame, canvas, points, motion_start, motion_end, args.out_png, CONFIG)

    print(f"Tip-trajectory video saved: {args.out_video}")
    print(f"Workspace PNG saved: {args.out_png} (area = {area_px2:,.0f} px^2)")


if __name__ == "__main__":
    main()
