# Finger Sweep Video Analysis — Common Scripts

Markerless computer-vision pipeline that extracts 2D joint angles and
fingertip-trajectory/workspace data from video of a mechanical finger sweep,
and renders the results as annotated videos + a publication-ready chart.
Built for 3 recordings for the RAL 2026 paper, but written to be reusable for
future recordings without editing any code.

**Code lives here (git); data does not.** Raw videos and every generated
output (multi-hundred-MB mp4s, tracking caches) live in a local
`Finger Sweep Videos\` folder next to this README — created automatically the
first time any script here runs, and gitignored so it never gets committed.
The finished dataset for the 3 recordings above (everything that folder held
when this pipeline was last run) has been copied out to
`Google Drive\Underactuated arm\` (`tip trajectory\`, `joint overlay\`,
`workspace\`, `angle chart\`, `data\` — raw videos are already there too,
under `Single finger sweep\`, just not renamed to match the case stems; see
the README on the Drive for the name mapping). To reproduce or extend that
dataset, drop a raw `.mp4` into `Finger Sweep Videos\raw\` here and run
`dashboard.py` or `run_all.py` — see "Running it" below.

## Data flow

```
raw video
   |
   v
track_points.py   <-- the ONE interactive step (click 5 points on frame 1,
   |                   then optical-flow tracks 4 of them across every frame)
   v
data\<stem>_points.npz   (cached tracked points; everything below reads this,
   |                      not the raw video's pixels, so nothing re-tracks
   |                      or re-prompts for clicks)
   |
   +--> render_joint_overlay.py  --> joint_overlay\<stem>.mp4
   |                               --> data\<stem>_angles.csv
   |
   +--> render_tip_trajectory.py --> tip_trajectory\<stem>.mp4
   |                               --> workspace\<stem>.png
   |
   +--> render_angle_chart.py    --> angle_chart\<stem>.mp4  (synced w/ source)
                                   --> angle_chart\<stem>.png / .pdf (paper figure)
```

The 5 clicked points, in fixed order, are: `base` (fixed anchor on the mount),
`mcp`, `pip`, `dip`, `tip`. Only the first 4 are tracked frame-to-frame via
optical flow — `tip` sits on the smooth, textureless rounded end of the
finger, which causes classic aperture-problem drift (it silently slides
along the curve as the segment rotates). **Four** approaches to deriving the
tip were tried and abandoned before the current one (see `track_points.py`'s
module docstring for the full diagnosis of each): (1) tracking a second point
partway along the distal segment (no second corner feature good enough for
optical flow on this hardware); (2) color-segmentation + PCA on the segment's
silhouette (degenerates into noisy, unphysical swings when the silhouette
foreshortens toward a round blob); (3) rotation-search template matching —
measuring the segment's rotation since a reference frame and rotating the
frame-0 (tip − dip) offset by it — which **drifted at the ends of the sweep**
because it is an integrator (residual per-frame error accumulates and is worst
at the largest offsets from the reference; this rig sweeps ~150°); (4) a
checkpoint-chain drift-correction bolted onto (3), which only moved *where* it
broke; and (5) an earlier version of the current geometric approach that took
the direction to the farthest *interior-pixel* band and pinned to the clicked
length — drift-free and correct, but it sat ~15-20px **inboard** of the true tip
corner (a visible offset). A CoTracker (Meta, model-based) run was also
evaluated as a cross-check (see `..\Model Tracking\`); it confirmed the ballpark
but tracked one clicked pixel that stayed inboard and slowly drifted over the
long sweep, so it was not adopted.

The approach used now: an **absolute geometric extremum via a tip-cap circle
fit**, with nothing integrated so nothing can accumulate. The distal link is a
bright blue rigid bar on a dark background and the `dip` is a reliably-tracked
real screw feature, so every frame *independently*: threshold the blue (HSV
range auto-sampled from the distal link in that video's frame 0), keep only the
finger's own connected blue (drops disconnected clutter — the control board in
frame is itself blue), take that region's contour and keep only the distal tip
cap edge (points at ≈ the rigid link length from `dip`, farther from `pip` than
`dip` is, within a continuity cone of the previous direction, then the farthest
slice of those), **fit a circle to that cap edge, and take its apex** (the point
on the circle farthest from `dip`) as the tip. The apex is the *true extreme
corner* — a multi-point circle fit is noise-robust, and it lands on the physical
corner regardless of exactly where the frame-0 click fell (an earlier "farthest
interior-pixel band" version sat ~15-20px inboard of the real corner). Over the
whole video the radius is pinned to the **median measured corner distance**:
`tip[t] = dip[t] + L_corner · dir[t]`, which makes the dip→tip length **constant
by construction** (it's a rigid link), click-independent, and 0% length-CV, so
only the *direction* is measured per frame. Nothing is tracked across frames —
the cap arc is re-detected from scratch each frame, so it can't drift. A physical
rate limit rejects non-physical single-frame direction swings (segmentation
outliers, e.g. the tip briefly crossing the cluttered electronics board) and
holds the last good direction. `track_points.py --from-cache <old.npz>`
re-derives just the tip column from an existing cache's tracks, so tuning the
tip parameters never requires re-clicking a video. All angle/geometry math lives
in `geometry.py` and is shared by
every renderer, so the joint-overlay video, the CSV, the tip trajectory, and
the angle chart can never disagree with each other — they're all views of
the same cached measurement.

## Folder map (`Finger Sweep Videos\`, auto-created next to this file)

| Folder | Contents |
|---|---|
| `raw\` | Source mp4s (read-only from here on out; never opened for writing) — drop videos in here |
| `joint_overlay\` | Skeleton + live joint-angle overlay video per case |
| `tip_trajectory\` | Fingertip trail video per case |
| `workspace\` | Freeze-frame + filled workspace-area PNG per case |
| `angle_chart\` | Synced angle-vs-time chart mp4 + final publication PNG/PDF per case |
| `data\` | `<stem>_points.npz` (tracking cache), `<stem>_tracking_debug.csv` (FB-error diagnostics), `<stem>_angles.csv` (trimmed/deduped angle dataset) |

All 6 are created on import by `run_all.py` (`ensure_data_dirs()`) if missing,
so a fresh clone just needs raw video(s) dropped into `raw\` before running
anything. Point this whole tree somewhere else (e.g. a synced Drive folder)
by setting the `FINGER_SWEEP_DATA_ROOT` environment variable to an absolute
path before running — every script picks it up via `run_all.VIDEOS_DIR`.

## Setup

```powershell
cd "C:\Users\namit\Documents\IITgn\underactuated finger\Common Scripts"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Video is always read via `cv2.VideoCapture` (the raw file is only ever opened
for reading), and the chart video's frames come from rendering matplotlib to
an in-memory array. Writing the output videos, though, does shell out to a
standalone `ffmpeg` binary: `requirements.txt` pulls in `imageio-ffmpeg`
(a self-contained ffmpeg build with `libx264`), and `video_writer.py` pipes
raw BGR frames to it over stdin with a fixed CRF (visually-lossless-quality
target, size adapts to content) instead of a fixed bitrate. This isn't
optional polish — OpenCV's own bundled FFmpeg plugin can only reach H.264 via
an external Cisco OpenH264 DLL, and that path has no CRF/bitrate control at
all (confirmed: `cv2.VIDEOWRITER_PROP_QUALITY` is silently ignored), so it
defaults to a fixed per-pixel bitrate that came out to ~62 Mbps regardless of
content — nearly double the raw source's own bitrate and ~3x this pipeline's
original renders for no visual benefit. See `video_writer.py`'s module
docstring for the full diagnosis.

**You do not need to activate the venv or pick the right interpreter** —
every script starts with a bootstrap check (`_bootstrap.py`) that transparently
relaunches itself under `Common Scripts\.venv\Scripts\python.exe` if it isn't
already running there. This means double-clicking a script, running it from
an IDE's "Run" button (whatever interpreter the IDE defaults to), or calling
`python foo.py` from a plain unactivated terminal all just work. A
`.vscode\settings.json` is also included so VS Code's own UI shows the right
interpreter, though that's a convenience, not what actually makes execution
correct — the bootstrap is.

## Running it

**Dashboard (recommended once you've run the full pipeline at least once):**
```powershell
python dashboard.py
```
Opens a grid — rows are video cases, columns are the 4 pipeline stages
(Track Points / Joint Overlay / Tip Trajectory / Angle Chart). Checkboxes
default to "checked" only for outputs that don't exist yet, so a first run
does everything, but afterwards you can check just the one stage/case you
need to regenerate (e.g. after tweaking a renderer's `CONFIG`) instead of
re-running the whole thing. A log panel streams each subprocess's output;
one failing case is reported and skipped rather than stopping the batch.

**All cases at once, from the CLI:**
```powershell
python run_all.py
```
This is interactive — a click window pops up for each raw video found in
`Finger Sweep Videos\raw\` in turn (see `track_points.py` below); any `.mp4`
dropped there gets picked up automatically, no code edit needed. Add
`--only <stem>` to process a single case, or `--skip-tracking` to reuse an
existing `data\<stem>_points.npz` cache instead of re-clicking.

**Per-script, one case at a time** (useful while iterating on a single video):

```powershell
$stem = "uniform_medium__20260702_182515"
$vid = "Finger Sweep Videos\raw\$stem.mp4"

# 1. Click the 5 points, track 4 of them, derive the tip, cache the result.
python track_points.py --video $vid `
    --out-cache "Finger Sweep Videos\data\${stem}_points.npz" `
    --out-debug-csv "Finger Sweep Videos\data\${stem}_tracking_debug.csv"

# 2. Skeleton overlay video + trimmed angle CSV.
python render_joint_overlay.py --video $vid `
    --cache "Finger Sweep Videos\data\${stem}_points.npz" `
    --out-video "Finger Sweep Videos\joint_overlay\$stem.mp4" `
    --out-csv "Finger Sweep Videos\data\${stem}_angles.csv"

# 3. Tip-trajectory video + workspace-area PNG.
python render_tip_trajectory.py --video $vid `
    --cache "Finger Sweep Videos\data\${stem}_points.npz" `
    --out-video "Finger Sweep Videos\tip_trajectory\$stem.mp4" `
    --out-png "Finger Sweep Videos\workspace\$stem.png"

# 4. Synced angle-chart video + final publication PNG/PDF.
python render_angle_chart.py `
    --cache "Finger Sweep Videos\data\${stem}_points.npz" `
    --out-video "Finger Sweep Videos\angle_chart\$stem.mp4" `
    --out-png "Finger Sweep Videos\angle_chart\$stem.png" `
    --out-pdf "Finger Sweep Videos\angle_chart\$stem.pdf"
```
(Plain `python`, not `.\.venv\Scripts\python.exe` — the bootstrap handles it
either way, so use whichever is convenient.)

If flexion reads negative on a video you can visually confirm is flexing, add
`--flexion-sign -1` to the render scripts (whichever camera side the rig was
filmed from flips the sign).

**Cache compatibility note:** the tip-derivation scheme has changed several
times since this pipeline was first built (`schema_version` is now 7).
Every renderer checks this field and refuses to load an older cache with a
clear error rather than silently misinterpreting it — v1 (tip tracked
directly), v2 (`distal_anchor` point), v3 (color+PCA segment orientation),
v4/v5 (rotation-search template matching, which drifted at the ends of the
sweep), and v6 (farthest interior-pixel band, which sat a few px inboard of
the true corner — see the module docstring in `track_points.py`) are all
obsolete.
Unlike the earlier scheme changes, upgrading an old cache to the current
geometric scheme does **not** require re-clicking: the base/mcp/pip/dip LK
tracks and the frame-0 tip click are all still valid, so
`python track_points.py --video <mp4> --out-cache <cache.npz> --from-cache <old.npz>`
re-derives just the tip column in place (a few seconds per video, no UI).

## Script-by-script notes

- **`_bootstrap.py`** — no CLI, stdlib-only. `ensure_venv()` re-launches the
  calling script under `Common Scripts\.venv\Scripts\python.exe` if it isn't
  already running there. Imported as the first real statement (after the
  module docstring and the mandatory `from __future__ import annotations`)
  in every other script in this folder. Uses `subprocess.run` + `sys.exit`
  rather than `os.execv` — the latter was found to mishandle this venv's path
  (which contains spaces) in this environment.
- **`video_writer.py`** — no CLI. `FFmpegWriter` is a drop-in replacement for
  `cv2.VideoWriter` (`write()`/`isOpened()`/`release()`) used by all 3
  renderers whenever `CONFIG["codec"] == "libx264"` (the default); pipes raw
  BGR frames to the standalone ffmpeg from `imageio-ffmpeg` with `crf`/`preset`
  read from each renderer's own `CONFIG`. Lower `crf` = higher quality/bigger
  file (18 is the standard "visually lossless" default). Any other
  `CONFIG["codec"]` string falls back to plain `cv2.VideoWriter` with that
  fourcc.
- **`geometry.py`** — no CLI, just shared math: `wrap_deg`, `signed_angle`,
  `compute_joint_angles`, `detect_motion_window`, `dedup_rows`,
  `polygon_area`. Edit here if the angle convention or motion/dedup
  thresholds need to change for *every* script at once.
- **`track_points.py`** — the only interactive step. A window opens on frame
  1; click BASE → MCP → PIP → DIP → TIP in order (a magnified inset near the
  cursor helps with precise placement). `u`/backspace undoes the last click,
  `r` resets, `ENTER` confirms once all 5 are placed, `ESC` cancels. Only the
  first 4 are tracked via optical flow; `tip` is derived as an absolute
  geometric extremum — a circle fit to the blue tip-cap edge each frame, apex
  taken as the tip (see the data-flow section above and the module docstring
  for the full design/diagnosis). Tuning knobs are in the `CONFIG` dict at the
  top of the file: LK window size/FB-threshold/hold-frames for the 4 tracked
  points (also have CLI overrides), plus the tip-derivation knobs (blue-sample/
  HSV-tolerance settings, the `cap_*` cap-edge gates, the circle-fit radius
  bounds, and the direction rate limit). `--from-cache <old.npz>` re-derives
  just the tip column from an existing cache without re-clicking. Watch the
  console for `[warn] frame ...` messages — either LK drift or a long
  tip-apex hold — during a long sweep.
- **`render_joint_overlay.py`** — per-joint colors, line/marker sizes, font,
  interpolated-frame dashed styling, and the CSV motion-window/dedup epsilon
  are all in its `CONFIG` block. Draws all 5 kinematic points (base, mcp,
  pip, dip, tip).
- **`render_tip_trajectory.py`** — trail colormap, marker styles, and
  workspace-fill color/alpha are in its `CONFIG` block. The workspace area is
  the polygon formed by the tip path (over the detected motion window only)
  closed back to the fixed base point, filled and measured via the shoelace
  formula (reported in px², annotated directly on the PNG).
- **`render_angle_chart.py`** — joint colors/linestyles/labels, the video's
  resolution/dpi (kept modest for render speed since it's a syncing artifact,
  not the paper figure) vs. the final figure's resolution/dpi (7.0×4.0in @
  300dpi), and the SciencePlots on/off toggle are all in its `CONFIG` block.
  The video's last frame and the final PNG/PDF show the same completed curve
  by construction — the PNG/PDF is just a second, higher-quality render pass
  of the same data.
- **`run_all.py`** — `ensure_data_dirs()` (called on import) creates
  `Finger Sweep Videos\` and its 6 subfolders if missing. `discover_cases()`
  scans `raw\*.mp4` and returns every case found, labeled via the optional
  `CASES` dict (add an entry there for a friendlier progress-print label; not
  required for a video to be processed). Also exposes `build_paths(stem)` and
  `run_stage(stage, stem, log=print)`, which `dashboard.py` imports and reuses
  directly rather than duplicating any path-building, discovery, or subprocess
  logic.
- **`dashboard.py`** — stdlib-only (`tkinter`); see "Running it" above. The
  `STAGE_OUTPUTS` dict controls which `build_paths()` keys count as "this
  stage's output" for the default-checkbox logic.

## Known limitations

- **Workspace area is in px², not physical units.** No pixel-to-mm
  calibration was set up (would need a known reference length visible in
  frame). Fine for relative comparison across the 3 stiffness cases; convert
  manually if an absolute area is needed for the paper.
- **Flexion sign / camera side** is a per-video assumption
  (`--flexion-sign`), not auto-detected — verify against a frame you know is
  flexing.
- **Classical optical flow (Lucas-Kanade) can drift** on very long or very
  low-texture stretches. `track_points.py` prints a console warning if
  forward-backward error creeps up, and `data\<stem>_tracking_debug.csv` has
  the full per-frame, per-point error trace if you need to audit a run. This
  is why `tip` is no longer tracked directly — it sat on a smooth,
  textureless curve and drifted along it (confirmed both visually and via
  20-28% frame-to-frame dip→tip length variation). Four point/shape/rotation
  derivations were tried and abandoned before the current absolute geometric
  extremum (see `track_points.py`'s module docstring for the full diagnosis of
  each). The current approach depends on the distal link being a distinctly
  colored rigid bar on a contrasting background and the `dip` being tracked
  reliably; if a future rig's distal link isn't a clean color blob against
  its background, the tip would need yet another approach (e.g. an added
  physical marker).
- Camera is assumed **fixed (tripod)** and **side-profile** to the flexion
  plane for every video — if a future recording violates either assumption,
  the pixel-space angles/areas will need a correction step this pipeline
  doesn't implement.
