"""Step 2a: render the link-skeleton + joint-angle overlay video, and write
the trimmed/deduped joint-angle CSV, from a track_points.py cache.

The source video is only ever opened for READING -- the overlay is written to
a brand-new mp4, the raw file is never modified.

Usage:
    python render_joint_overlay.py --video <path> --cache <path> \
        --out-video <path> --out-csv <path> [--flexion-sign 1]
"""

from __future__ import annotations

from _bootstrap import ensure_venv
ensure_venv()

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import geometry
from video_writer import FFmpegWriter

# ---------------------------------------------------------------------------
# CONFIG -- colors, styling, and CSV-trimming thresholds live here.
# ---------------------------------------------------------------------------
CONFIG = {
    # BGR, one per kinematic point drawn, in order: base, mcp, pip, dip, tip
    "point_colors": [
        (255, 255, 255),
        (255, 120, 0),
        (0, 200, 0),
        (0, 0, 255),
        (0, 255, 255),
    ],
    "point_radius": 6,
    "link_color": (60, 60, 60),
    "link_thickness": 2,
    "interpolated_link_color": (0, 140, 255),
    "interpolated_dash_len": 9,
    "interpolated_gap_len": 6,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.65,
    "font_thickness": 2,
    "text_outline_thickness": 4,
    "text_outline_color": (0, 0, 0),
    # per-joint label color, keyed by geometry.JOINT_NAMES
    "joint_text_colors": {"mcp": (255, 120, 0), "pip": (0, 200, 0), "dip": (0, 0, 255)},
    "codec": "libx264",
    "crf": 18,
    "preset": "medium",

    # CSV trimming (see geometry.detect_motion_window / dedup_rows)
    "motion_eps_deg": 0.15,
    "motion_min_run": 5,
    "dedup_eps_deg": 0.15,
}

# index of each joint's own marker point, for label placement
_JOINT_POINT_INDEX = {"mcp": 1, "pip": 2, "dip": 3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Path to the source mp4 (read-only).")
    p.add_argument("--cache", required=True, help="track_points.py .npz cache path.")
    p.add_argument("--out-video", required=True, help="Output joint-overlay mp4 path.")
    p.add_argument("--out-csv", required=True, help="Output trimmed/deduped angle CSV path.")
    p.add_argument("--flexion-sign", type=int, default=1, choices=(1, -1))
    return p.parse_args()


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len, gap_len):
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)
    dist = float(np.linalg.norm(pt2 - pt1))
    if dist < 1e-6:
        return
    direction = (pt2 - pt1) / dist
    step = dash_len + gap_len
    n_dashes = int(dist // step) + 1
    for k in range(n_dashes):
        start = pt1 + direction * (k * step)
        end = pt1 + direction * min(k * step + dash_len, dist)
        cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness, cv2.LINE_AA)


def _put_text_outlined(frame, text, org, config):
    cv2.putText(frame, text, org, config["font"], config["font_scale"],
                config["text_outline_color"], config["text_outline_thickness"], cv2.LINE_AA)


def draw_overlay(frame, pts, angles_i, is_interpolated, config):
    pts_i = [tuple(np.round(p).astype(int)) for p in pts]

    link_color = config["interpolated_link_color"] if is_interpolated else config["link_color"]
    for a, b in zip(pts_i[:-1], pts_i[1:]):
        if is_interpolated:
            _draw_dashed_line(frame, a, b, link_color, config["link_thickness"],
                               config["interpolated_dash_len"], config["interpolated_gap_len"])
        else:
            cv2.line(frame, a, b, link_color, config["link_thickness"], cv2.LINE_AA)

    for idx, p in enumerate(pts_i):
        cv2.circle(frame, p, config["point_radius"], config["point_colors"][idx], -1, cv2.LINE_AA)

    for joint in geometry.JOINT_NAMES:
        idx = _JOINT_POINT_INDEX[joint]
        x, y = pts_i[idx]
        color = config["joint_text_colors"][joint]
        text = f"{joint.upper()} {angles_i[joint]:+.1f}"
        org = (x + 10, y - 10)
        _put_text_outlined(frame, text, org, config)
        cv2.putText(frame, text, org, config["font"], config["font_scale"], color,
                    config["font_thickness"], cv2.LINE_AA)

    if is_interpolated:
        badge = "INTERPOLATED"
        org = (10, 26)
        _put_text_outlined(frame, badge, org, config)
        cv2.putText(frame, badge, org, config["font"], config["font_scale"],
                    config["interpolated_link_color"], config["font_thickness"], cv2.LINE_AA)


def render_video(video_path, points, interpolated, fps, out_video, flexion_sign, config):
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

    # Select the 5 kinematic points by name (base, mcp, pip, dip, tip) so
    # this stays correct regardless of the underlying POINT_NAMES order.
    kinematic_idx = [geometry.POINT_NAMES.index(n) for n in ("base", "mcp", "pip", "dip", "tip")]

    angles = geometry.compute_joint_angles(points, flexion_sign=flexion_sign)
    n_frames = points.shape[0]

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        angles_i = {k: v[i] for k, v in angles.items()}
        draw_overlay(frame, points[i][kinematic_idx], angles_i, bool(interpolated[i]), config)
        writer.write(frame)

    cap.release()
    writer.release()


def write_angle_csv(frame_idx, time_s, angles, out_csv, config):
    start, end = geometry.detect_motion_window(
        angles, eps_deg=config["motion_eps_deg"], min_run=config["motion_min_run"]
    )
    window = slice(start, end + 1)
    windowed_angles = {k: v[window] for k, v in angles.items()}
    rows = geometry.dedup_rows(
        frame_idx[window], time_s[window], windowed_angles, eps_deg=config["dedup_eps_deg"]
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "time_s", "mcp_deg", "pip_deg", "dip_deg"])
        for i in range(len(rows["frame_idx"])):
            w.writerow([
                int(rows["frame_idx"][i]),
                f"{rows['time_s'][i]:.4f}",
                f"{rows['mcp_deg'][i]:.3f}",
                f"{rows['pip_deg'][i]:.3f}",
                f"{rows['dip_deg'][i]:.3f}",
            ])
    print(f"Kept {len(rows['frame_idx'])} / {end - start + 1} rows in the motion window "
          f"[{start}, {end}] (full video had {len(frame_idx)} frames).")


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
    interpolated = cache["interpolated"]
    fps = float(cache["fps"])
    n_frames = int(cache["n_frames"])

    render_video(args.video, points, interpolated, fps, args.out_video, args.flexion_sign, CONFIG)

    angles = geometry.compute_joint_angles(points, flexion_sign=args.flexion_sign)
    frame_idx = np.arange(n_frames)
    time_s = frame_idx / fps
    write_angle_csv(frame_idx, time_s, angles, args.out_csv, CONFIG)

    print(f"Joint-overlay video saved: {args.out_video}")
    print(f"Angle CSV saved: {args.out_csv}")


if __name__ == "__main__":
    main()
