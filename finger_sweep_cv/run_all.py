"""Driver for the finger-sweep video analysis pipeline.

Runs track_points.py (interactive -- you'll click 5 points per video) and
then the three renderers (joint overlay, tip trajectory, angle chart) for
every raw video found in the local data root's "raw\\" folder. CASES is only
an optional label lookup for progress printouts -- any .mp4 dropped into
"raw\\" is picked up automatically, labeled or not.

The local data root (raw videos in, everything else generated) is created on
demand the first time this module is imported -- see `ensure_data_dirs()` --
so a fresh clone of this repo works immediately: drop video(s) into
`<data root>\\raw\\` and run `run_all.py` or `dashboard.py`. It defaults to
`Finger Sweep Videos\\` next to this file (gitignored: videos/caches don't
belong in git), overridable via the `FINGER_SWEEP_DATA_ROOT` env var if you'd
rather point it somewhere else (e.g. a synced Drive folder).

`build_paths`, `run_stage`, and `discover_cases` are also imported directly by
`dashboard.py`, which lets a user pick exactly which (case, stage)
combinations to (re-)run instead of always processing the whole pipeline.

Usage:
    python run_all.py                          # process every case in raw\
    python run_all.py --only uniform_medium__20260702_182515
    python run_all.py --skip-tracking          # reuse an existing cache per case
    python run_all.py --video "C:\path\to\some_video.mp4" --label "My new case"
                                                # ingest a video from anywhere on
                                                # disk (copied into raw\ first)
"""

from __future__ import annotations

from _bootstrap import ensure_venv
ensure_venv()

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = Path(os.environ.get("FINGER_SWEEP_DATA_ROOT", SCRIPTS_DIR / "Finger Sweep Videos")).resolve()

# stem (raw filename without .mp4) -> human label, for progress printouts only.
# Not required: any raw video without an entry here still gets processed,
# just labeled by its own filename stem (see discover_cases()).
CASES = {
    "uniform_medium__20260702_182515": "Uniform medium",
    "hard_medium_distal_dominant__20260702_185711": "Hard-medium (distal-dominant)",
    "medium_hard_proximal_dominant__20260702_190423": "Medium-hard (proximal-dominant)",
}

# Every generated subfolder under VIDEOS_DIR -- created on demand so a fresh
# clone works without the user having to mkdir anything by hand.
_DATA_SUBFOLDERS = ("raw", "data", "joint_overlay", "tip_trajectory", "workspace", "angle_chart")


def ensure_data_dirs(videos_dir: Path = VIDEOS_DIR) -> None:
    for name in _DATA_SUBFOLDERS:
        (videos_dir / name).mkdir(parents=True, exist_ok=True)


ensure_data_dirs()


def discover_cases() -> tuple[list[str], dict[str, str], list[str]]:
    """Scan VIDEOS_DIR/raw for *.mp4 and return (ordered stems, label-by-stem,
    warnings). Labeled CASES entries come first (in CASES order), followed by
    any unlabeled raw videos (alphabetical), so a fresh video just works
    without editing this file."""
    labels = dict(CASES)
    warnings = []

    raw_stems = {p.stem for p in (VIDEOS_DIR / "raw").glob("*.mp4")}

    for stem in CASES:
        if stem not in raw_stems:
            warnings.append(f"case '{stem}' has a label but no raw video in {VIDEOS_DIR / 'raw'}")

    unlabeled = sorted(raw_stems - set(CASES))
    ordered = [s for s in CASES if s in raw_stems] + unlabeled
    return ordered, labels, warnings


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--only", default=None, help="Process only this case stem (default: all).")
    group.add_argument("--video", type=Path, default=None,
                        help="Path to a video file anywhere on disk to ingest as a new "
                             "case (copied into raw\\ under its own filename stem, then "
                             "processed like any other case).")
    p.add_argument("--label", default=None,
                    help="Human-readable label for --video's progress-print header "
                         "(optional; defaults to the filename stem).")
    p.add_argument("--skip-tracking", action="store_true",
                    help="Reuse an existing tracking cache instead of re-running the click UI.")
    return p.parse_args()


def ingest_video(video_path: Path) -> str:
    """Copy an arbitrary video into VIDEOS_DIR/raw (if not already there) and
    return its stem, so it can be processed like any other case."""
    if not video_path.is_file():
        sys.exit(f"--video: no such file: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        sys.exit(f"--video: expected a .mp4 file, got: {video_path}")

    stem = video_path.stem
    dest = VIDEOS_DIR / "raw" / f"{stem}.mp4"
    if dest.exists() and dest.resolve() != video_path.resolve():
        sys.exit(f"--video: a raw video already exists at {dest} -- rename your "
                  f"input or remove the existing one first.")
    if not dest.exists():
        shutil.copy2(video_path, dest)
        print(f"[info] copied {video_path} -> {dest}")
    return stem


def build_paths(stem: str) -> dict:
    """Pure path assembly for one case -- no side effects, no existence checks."""
    return {
        "raw_video": VIDEOS_DIR / "raw" / f"{stem}.mp4",
        "cache": VIDEOS_DIR / "data" / f"{stem}_points.npz",
        "debug_csv": VIDEOS_DIR / "data" / f"{stem}_tracking_debug.csv",
        "joint_video": VIDEOS_DIR / "joint_overlay" / f"{stem}.mp4",
        "angle_csv": VIDEOS_DIR / "data" / f"{stem}_angles.csv",
        "tip_video": VIDEOS_DIR / "tip_trajectory" / f"{stem}.mp4",
        "workspace_png": VIDEOS_DIR / "workspace" / f"{stem}.png",
        "chart_video": VIDEOS_DIR / "angle_chart" / f"{stem}.mp4",
        "chart_png": VIDEOS_DIR / "angle_chart" / f"{stem}.png",
        "chart_pdf": VIDEOS_DIR / "angle_chart" / f"{stem}.pdf",
    }


def run_stage(stage: str, stem: str, log=print) -> int:
    """Run exactly one pipeline stage for one case.

    stage : "track" | "overlay" | "trajectory" | "chart"
    log : callable(str) -- receives every subprocess output line, plus a
        couple of progress lines of our own. Defaults to print(), but the
        dashboard passes something that routes lines into its log panel.

    Returns the subprocess's exit code (0 = success). Does NOT raise on
    failure -- that's the caller's decision (the CLI driver below raises to
    fail fast; the dashboard catches and continues).
    """
    paths = build_paths(stem)
    if stage == "track":
        cmd = [sys.executable, str(SCRIPTS_DIR / "track_points.py"),
               "--video", str(paths["raw_video"]),
               "--out-cache", str(paths["cache"]),
               "--out-debug-csv", str(paths["debug_csv"])]
    elif stage == "overlay":
        cmd = [sys.executable, str(SCRIPTS_DIR / "render_joint_overlay.py"),
               "--video", str(paths["raw_video"]),
               "--cache", str(paths["cache"]),
               "--out-video", str(paths["joint_video"]),
               "--out-csv", str(paths["angle_csv"])]
    elif stage == "trajectory":
        cmd = [sys.executable, str(SCRIPTS_DIR / "render_tip_trajectory.py"),
               "--video", str(paths["raw_video"]),
               "--cache", str(paths["cache"]),
               "--out-video", str(paths["tip_video"]),
               "--out-png", str(paths["workspace_png"])]
    elif stage == "chart":
        cmd = [sys.executable, str(SCRIPTS_DIR / "render_angle_chart.py"),
               "--cache", str(paths["cache"]),
               "--out-video", str(paths["chart_video"]),
               "--out-png", str(paths["chart_png"]),
               "--out-pdf", str(paths["chart_pdf"])]
    else:
        raise ValueError(f"Unknown stage: {stage!r}")

    log(f"\n$ {' '.join(cmd)}")
    if stage == "track":
        log("Waiting for the interactive point-selection window "
            "(click on the video frame that pops up)...")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log(line.rstrip("\n"))
    proc.wait()
    return proc.returncode


def process_case(stem, args, label=None):
    paths = build_paths(stem)
    if not paths["raw_video"].exists():
        print(f"[skip] {stem}: raw video not found at {paths['raw_video']}")
        return

    print(f"\n=== {stem} ({label or CASES.get(stem, stem)}) ===")

    for stage in ("track", "overlay", "trajectory", "chart"):
        if stage == "track" and args.skip_tracking and paths["cache"].exists():
            print(f"[skip] tracking cache already exists: {paths['cache']}")
            continue
        rc = run_stage(stage, stem, log=print)
        if rc != 0:
            raise RuntimeError(f"{stage} stage failed for {stem} (exit {rc}).")


def main():
    args = parse_args()
    if args.video:
        stem = ingest_video(args.video)
        stems, labels = [stem], {stem: args.label} if args.label else CASES
    elif args.only:
        stems, labels = [args.only], CASES
    else:
        stems, labels, warnings = discover_cases()
        for w in warnings:
            print(f"[warn] {w}")
        if not stems:
            print(f"No raw videos found in {VIDEOS_DIR / 'raw'} -- drop a .mp4 there and re-run.")
    for stem in stems:
        process_case(stem, args, label=labels.get(stem))


if __name__ == "__main__":
    main()
