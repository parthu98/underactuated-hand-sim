"""Step 2c: render a synced angle-vs-time chart video (same fps/frame count as
the source, so it can be stacked/composited with the joint-overlay and
tip-trajectory videos), plus a publication-quality static PNG/PDF of the
final, fully-drawn chart -- from a track_points.py cache.

Usage:
    python render_angle_chart.py --cache <path> --out-video <path> \
        --out-png <path> --out-pdf <path> [--flexion-sign 1]
"""

from __future__ import annotations

from _bootstrap import ensure_venv
ensure_venv()

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
import numpy as np  # noqa: E402

import geometry  # noqa: E402
from video_writer import FFmpegWriter  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG -- colors, linestyles, labels, figure sizes/dpi all live here.
# ---------------------------------------------------------------------------
CONFIG = {
    "joint_style": {
        "mcp": {"color": "#1f77b4", "linestyle": "-", "label": "MCP"},
        "pip": {"color": "#2ca02c", "linestyle": "--", "label": "PIP"},
        "dip": {"color": "#d62728", "linestyle": "-.", "label": "DIP"},
    },
    "linewidth": 1.3,
    "marker_color": "#888888",
    "marker_linestyle": ":",
    "marker_linewidth": 1.0,
    "xlabel": "Time (s)",
    "ylabel": "Joint angle (deg)",
    "legend_loc": "best",
    "legend_fontsize": 8,
    "use_scienceplots": True,

    # synced video export -- moderate resolution/dpi, speed matters here
    "video_size_in": (6.0, 3.4),
    "video_dpi": 100,
    "codec": "libx264",
    "crf": 18,
    "preset": "medium",

    # final static paper figure -- quality matters here
    "pub_size_in": (7.0, 4.0),
    "pub_dpi": 300,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="track_points.py .npz cache path.")
    p.add_argument("--out-video", required=True, help="Output synced chart mp4 path.")
    p.add_argument("--out-png", required=True, help="Output final publication PNG path.")
    p.add_argument("--out-pdf", required=True, help="Output final publication PDF path.")
    p.add_argument("--flexion-sign", type=int, default=1, choices=(1, -1))
    return p.parse_args()


def _apply_style(config):
    if config["use_scienceplots"]:
        try:
            import scienceplots  # noqa: F401

            plt.style.use(["science", "no-latex"])
            return
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
    })


def render_chart_video(time, angles, fps, out_video, config):
    _apply_style(config)
    fig, ax = plt.subplots(figsize=config["video_size_in"], dpi=config["video_dpi"])
    canvas = FigureCanvasAgg(fig)

    lines = {}
    for joint in geometry.JOINT_NAMES:
        style = config["joint_style"][joint]
        (line,) = ax.plot([], [], color=style["color"], linestyle=style["linestyle"],
                           linewidth=config["linewidth"], label=style["label"])
        lines[joint] = line

    vline = ax.axvline(time[0] if len(time) else 0.0, color=config["marker_color"],
                        linestyle=config["marker_linestyle"], linewidth=config["marker_linewidth"])

    ax.set_xlim(time[0], time[-1] if len(time) > 1 else 1.0)
    y_all = np.concatenate([angles[j] for j in geometry.JOINT_NAMES])
    pad = 0.08 * (float(y_all.max()) - float(y_all.min()) + 1e-6)
    ax.set_ylim(float(y_all.min()) - pad, float(y_all.max()) + pad)
    ax.set_xlabel(config["xlabel"])
    ax.set_ylabel(config["ylabel"])
    ax.legend(loc=config["legend_loc"], fontsize=config["legend_fontsize"])
    ax.grid(True)
    fig.tight_layout(pad=0.3)

    canvas.draw()
    w_px, h_px = canvas.get_width_height()

    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
    if config["codec"] == "libx264":
        writer = FFmpegWriter(out_video, fps, (w_px, h_px), crf=config["crf"], preset=config["preset"])
    else:
        fourcc = cv2.VideoWriter_fourcc(*config["codec"])
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w_px, h_px))

    n = len(time)
    for i in range(n):
        for joint in geometry.JOINT_NAMES:
            lines[joint].set_data(time[: i + 1], angles[joint][: i + 1])
        vline.set_xdata([time[i], time[i]])

        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).reshape(h_px, w_px, 4)
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

    writer.release()
    plt.close(fig)


def render_final_figure(time, angles, out_png, out_pdf, config):
    _apply_style(config)
    fig, ax = plt.subplots(figsize=config["pub_size_in"])
    for joint in geometry.JOINT_NAMES:
        style = config["joint_style"][joint]
        ax.plot(time, angles[joint], color=style["color"], linestyle=style["linestyle"],
                linewidth=config["linewidth"], label=style["label"])
    ax.set_xlabel(config["xlabel"])
    ax.set_ylabel(config["ylabel"])
    ax.legend(loc=config["legend_loc"], fontsize=config["legend_fontsize"])
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout(pad=0.3)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=config["pub_dpi"])
    fig.savefig(out_pdf)
    plt.close(fig)


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
    n_frames = int(cache["n_frames"])

    angles = geometry.compute_joint_angles(points, flexion_sign=args.flexion_sign)
    time = np.arange(n_frames) / fps

    render_chart_video(time, angles, fps, args.out_video, CONFIG)
    render_final_figure(time, angles, args.out_png, args.out_pdf, CONFIG)

    print(f"Synced angle-chart video saved: {args.out_video}")
    print(f"Final figure saved: {args.out_png} and {args.out_pdf}")


if __name__ == "__main__":
    main()
