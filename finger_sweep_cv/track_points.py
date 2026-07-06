"""Interactive step 1 of the pipeline: click 5 points on frame 1 of a
finger-sweep video -- base, mcp, pip, dip, tip, in that order -- then track
base/mcp/pip/dip across every remaining frame with pyramidal Lucas-Kanade
optical flow (forward-backward validated). ``tip`` is clicked once for
calibration only: it fixes the distal link's length and its initial pointing
direction. It is never fed into optical flow (it sits on a smooth, textureless
silhouette edge that causes tracking drift).

WHY THE TIP IS DERIVED, NOT TRACKED -- and the full history of approaches tried
and diagnosed broken on this hardware (each confirmed by replay against the
real ``uniform_medium`` sweep, not just in theory):

1. LK-tracking the tip directly: classic optical-flow aperture problem. The tip
   sits on a smooth curved edge, so LK can't constrain motion tangential to the
   curve and the point slides along the fingertip; forward-backward error stays
   tiny (<0.04px) even while it drifts, so it can't self-detect.
2. A second LK point (``distal_anchor``) partway up the distal segment, deriving
   the tip from how it rotated about the dip. Failed: the only real corner
   texture near the tip is the DIP joint itself; everywhere else is smooth blue
   plastic (Shi-Tomasi corner strength ~0.089 vs 0.2-0.32 for real joints), so
   that point drifts the same silent way.
3. Color segmentation + PCA on the distal silhouette (orientation, not a
   trackable point). Failed: the silhouette foreshortens toward a near-round
   blob at some rotation angles where the two PCA eigenvalues converge
   (ratio 1.2-1.8) and the principal axis becomes genuinely noisy -- 80+ degree
   swings frame-to-frame no plausibility patch fully removed.
4. Rotation-search template matching + checkpoint-chain drift correction: measure
   the distal segment's rotation since a reference frame via normalized-cross-
   correlation over candidate rotations, then rotate the frame-0 (tip-dip)
   offset by that angle. Failed at the ENDS of the sweep. It is fundamentally an
   *integrator*: the tip is reconstructed from a rotation accumulated relative to
   a reference, so any residual per-frame angular error compounds and is worst at
   the largest offsets from the reference (this rig sweeps ~150 degrees). Every
   patch (throttled template refresh, then a checkpoint chain that re-anchors to
   the most recent validated checkpoint instead of frame 0) only moved *where*
   it broke -- on a genuinely fresh re-click the tip still visibly floated off
   the finger for a large middle/late stretch of the sweep, while passing every
   jump/length numeric check because the drift was smooth.

THE APPROACH USED HERE -- absolute geometric extremum (no reference frame, no
rotation integration, so nothing accumulates and the end of a sweep is no harder
than the start):

The distal link is a bright blue rigid bar on a dark background, and the DIP
joint is one of the reliably LK-tracked real screw features. So every frame,
independently:

  - threshold the blue plastic (HSV range auto-sampled from the distal link in
    THIS video's frame 0, so it adapts to each recording's lighting);
  - keep only the finger's own connected blue (drops disconnected clutter -- the
    control board in frame is itself blue), then take that region's contour and
    keep only the distal tip cap edge: contour points at ~the rigid link length
    from the tracked DIP (an annulus, since the link can't stretch), farther from
    the PIP than the DIP is (the distal side, which disambiguates the near-
    colinear extended pose), within a cone of the previous frame's direction
    (continuity), and then the farthest slice of those (the rounded cap);
  - fit a circle to that cap edge and take its APEX (the point on the circle
    farthest from the DIP) as the tip -- the true extreme corner. A multi-point
    circle fit is noise-robust, and the apex is the physical corner regardless of
    exactly where the frame-0 click landed (an interior-pixel "farthest band"
    centroid, the previous method, sat ~15-20px inboard of the real corner --
    that was a visible inboard bias);
  - over the whole video, pin the radius to the MEDIAN measured apex distance:
    ``tip[t] = dip[t] + L_corner * dir[t]``.

Pinning the radius to the measured median makes the DIP->tip length constant by
construction (it is a rigid link) and click-independent (0% length CV), so only
the *direction* is measured per frame -- and the direction to a circle-fit apex
is far more stable than any trackable point on the smooth edge. Nothing is
tracked across frames: the cap arc is re-detected from scratch every frame, so
nothing can accumulate/drift. If a frame has too little of the cap visible to
trust (occlusion, blur) -- or the measured direction jumps non-physically far in
one frame (a segmentation outlier, e.g. the tip briefly crossing the cluttered
electronics board) -- the last confident direction is held, with a widening
tolerance so it can never lock up.

The result is cached to a .npz (schema_version=7) so the three render scripts
reuse the same tracked points without re-clicking or re-tracking. ``--from-cache``
re-derives just the tip column from an existing cache's already-good
base/mcp/pip/dip tracks + frame-0 tip click, so tuning the geometric parameters
never requires re-clicking a video.

Usage:
    python track_points.py --video <path> --out-cache <path> [options]
    python track_points.py --video <path> --out-cache <path> \
        --from-cache <existing.npz>       # re-derive tip only, no click/track
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

# ---------------------------------------------------------------------------
# CONFIG -- tune these to change tracking behavior / click-UI look & feel.
# CLI flags (see parse_args) override the *_px / frame-count defaults below.
# ---------------------------------------------------------------------------
CONFIG = {
    # Lucas-Kanade optical flow
    "lk_win_size": (31, 31),
    "lk_max_level": 3,
    "lk_criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    "blur_ksize": 3,  # odd int; 0/1 disables the pre-tracking Gaussian blur

    # Forward-backward validation / recovery
    "fb_thresh_px": 2.0,
    "hold_frames": 5,
    "reanchor_interval": 150,
    "drift_warn_px": 1.5,

    # Click UI
    "max_display_dim": 1400,
    "magnifier_crop_px": 61,
    "magnifier_zoom": 6,
    # BGR, one per POINT_NAMES order: base, mcp, pip, dip, tip
    "point_colors": [
        (255, 255, 255),
        (255, 120, 0),
        (0, 200, 0),
        (0, 0, 255),
        (0, 255, 255),
    ],
    "point_labels": [
        "BASE (fixed anchor)", "MCP", "PIP", "DIP", "TIP",
    ],

    # --- Distal-segment tip derivation: absolute geometric extremum ---
    # See the module docstring. The tip is NOT tracked and NOT integrated from a
    # measured rotation (both failed). Each frame it is placed at the frame-0
    # clicked link length off the reliably-tracked DIP, pointed along the distal
    # segment's direction measured absolutely from the blue mask THAT frame --
    # no reference frame, nothing accumulates, so there is no drift to correct.

    # Blue HSV range is auto-sampled per video from the distal link in frame 0
    # (robust to lighting differences between recordings). Samples are taken at
    # these fractions along the dip0->tip0 line (safely on the distal link).
    "blue_sample_fracs": (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
    "blue_sample_radius_px": 4,
    "blue_hue_tol": 18,     # +/- around sampled median hue
    "blue_sat_floor": 40,   # never let the low-saturation bound drop below this
    "blue_sat_tol": 90,     # sampled median sat minus this = low bound (clamped)
    "blue_val_floor": 40,
    "blue_val_tol": 90,
    "morph_kernel_px": 5,   # ellipse kernel for open+close mask cleanup

    # Tip-cap circle fit. Each frame, RE-DETECT the distal link's rounded tip
    # edge arc from the finger contour and fit a circle to it; the tip is that
    # circle's apex (its point farthest from the dip) -- the TRUE extreme
    # corner, not a few px inboard of it (an interior-pixel "farthest band"
    # centroid, the previous method, sat ~15-20px short of the real corner --
    # that was the visible inboard bias). Nothing is tracked frame-to-frame:
    # the arc is re-found from scratch every frame, so it cannot drift. Gates
    # (all using the reliably-tracked DIP/PIP) keep only contour points that
    # are physically the distal cap:
    "cap_lo_frac": 0.60,         # arc at 0.60..1.45 x link length from the DIP
    "cap_hi_frac": 1.45,
    "distal_pip_frac": 0.98,     # ... and farther from PIP than the DIP is
    "cap_cone_deg": 70.0,        # ... and within this cone of last frame's dir
    "cap_far_pct": 45.0,         # then fit the circle to the farthest 55% (the cap)
    "min_cap_points": 8,         # too few cap edge points -> hold
    "circle_min_radius_px": 3.0, # reject a degenerate circle fit (fall back to
    "circle_max_radius_frac": 2.5,  # farthest contour point if R outside these)
    "min_blue_pixels": 200,      # too few total blue px this frame -> hold
    # Keep only the finger's OWN blue: connected components of the mask that
    # reach within this fraction of the link length of the tracked dip. Both
    # finger links wrap around the dip screw so both are kept, but DISCONNECTED
    # blue clutter is dropped -- critically the blue Arduino board the tip
    # sweeps over mid-stroke, which is otherwise a false 'farthest' blue point.
    "cc_dip_radius_frac": 0.45,
    # Physical rate limit on the measured direction. Real motion here is well
    # under ~2 deg/frame (tip frame-to-frame p99 ~1.8 deg); a larger single-frame
    # swing is a segmentation outlier (e.g. the tip passing over the cluttered
    # electronics board briefly pulls a stray blue region into the gates), so the
    # last confident direction is held instead. The allowance WIDENS by
    # dir_widen_per_frame for each consecutive held frame (capped) so a genuinely
    # fast/large real move -- or a persistent, non-transient change -- is never
    # permanently rejected.
    "max_dir_step_deg": 8.0,
    "dir_widen_per_frame": 2.0,
    "dir_widen_cap_deg": 18.0,
    "tip_hold_warn_frames": 30,  # console warning threshold for consecutive holds
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Path to the source mp4.")
    p.add_argument("--out-cache", required=True, help="Output .npz cache path.")
    p.add_argument("--out-debug-csv", default=None,
                    help="Output tracking-debug CSV path (default: derived from --out-cache).")
    p.add_argument("--from-cache", default=None,
                    help="Re-derive ONLY the tip column from an existing cache's "
                         "base/mcp/pip/dip tracks + frame-0 tip click (no click/track UI). "
                         "Use after tuning the geometric tip parameters.")
    p.add_argument("--fb-thresh", type=float, default=CONFIG["fb_thresh_px"],
                    help="Forward-backward pixel-error threshold to flag a point untracked.")
    p.add_argument("--hold-frames", type=int, default=CONFIG["hold_frames"],
                    help="Frames to hold a point's last-good position before switching to gap interpolation.")
    p.add_argument("--reanchor-interval", type=int, default=CONFIG["reanchor_interval"],
                    help="Frame interval between drift checks (0 disables).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Click UI
# ---------------------------------------------------------------------------

def select_points(frame, config):
    """Interactive 5-point click UI. Returns ndarray (5, 2) float32, full-res coords."""
    h, w = frame.shape[:2]
    max_disp = config["max_display_dim"]
    scale = min(1.0, max_disp / float(max(h, w)))
    disp_w, disp_h = int(round(w * scale)), int(round(h * scale))
    base_display = (
        cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        if scale < 1.0 else frame.copy()
    )

    labels = config["point_labels"]
    colors = config["point_colors"]
    inset_size = config["magnifier_crop_px"]
    inset_zoom = config["magnifier_zoom"]

    state = {"points": [], "mouse_disp": (disp_w // 2, disp_h // 2), "quit": False}

    def on_mouse(event, x, y, flags, userdata):  # noqa: ARG001
        state["mouse_disp"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(state["points"]) < len(labels):
            state["points"].append((x / scale, y / scale))

    win = "select points"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        canvas = base_display.copy()
        pts = state["points"]
        n = len(pts)

        pts_disp = [(int(round(fx * scale)), int(round(fy * scale))) for fx, fy in pts]
        for i in range(1, n):
            cv2.line(canvas, pts_disp[i - 1], pts_disp[i], (180, 180, 180), 1, cv2.LINE_AA)
        for i, (dx, dy) in enumerate(pts_disp):
            color = colors[i]
            cv2.circle(canvas, (dx, dy), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, labels[i].split()[0], (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if n < len(labels):
            prompt = f"Click {n + 1}/{len(labels)}: {labels[n]}   [u=undo  r=reset  ESC=quit]"
        else:
            prompt = "ENTER=confirm   u=undo   r=reset   ESC=quit"
        cv2.putText(canvas, prompt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, prompt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        if n < len(labels):
            mx, my = state["mouse_disp"]
            full_x, full_y = mx / scale, my / scale
            fx0 = int(np.clip(full_x - inset_size // 2, 0, max(w - inset_size, 0)))
            fy0 = int(np.clip(full_y - inset_size // 2, 0, max(h - inset_size, 0)))
            crop = frame[fy0:fy0 + inset_size, fx0:fx0 + inset_size]
            if crop.shape[0] == inset_size and crop.shape[1] == inset_size:
                mag = cv2.resize(crop, (inset_size * inset_zoom, inset_size * inset_zoom),
                                  interpolation=cv2.INTER_NEAREST)
                ih, iw = mag.shape[:2]
                cv2.line(mag, (iw // 2, 0), (iw // 2, ih), (0, 255, 255), 1)
                cv2.line(mag, (0, ih // 2), (iw, ih // 2), (0, 255, 255), 1)
                canvas[0:ih, disp_w - iw:disp_w] = mag
                cv2.rectangle(canvas, (disp_w - iw, 0), (disp_w - 1, ih - 1), (0, 255, 255), 2)

        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            state["quit"] = True
            break
        elif key in (8, ord("u")):
            if state["points"]:
                state["points"].pop()
        elif key == ord("r"):
            state["points"] = []
        elif key == 13 and len(state["points"]) == len(labels):
            break

    cv2.destroyWindow(win)
    if state["quit"] or len(state["points"]) != len(labels):
        raise RuntimeError(f"Point selection cancelled before all {len(labels)} points were confirmed.")
    return np.array(state["points"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def _prep_gray(frame, blur_ksize):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray


def _fill_point_gaps(points, point_idx, col_bad, hold_frames, interpolated_frame):
    """Replace long dropout runs (> hold_frames) with linear interpolation.

    Short runs (<= hold_frames) keep the held (frozen last-good) value that
    was already written during the forward pass. Every bad frame -- short or
    long -- is still marked in ``interpolated_frame`` since neither case is a
    direct fresh measurement.
    """
    n = len(col_bad)
    interpolated_frame |= col_bad

    i = 0
    while i < n:
        if col_bad[i]:
            j = i
            while j < n and col_bad[j]:
                j += 1
            run_len = j - i
            if run_len > hold_frames:
                before = i - 1
                after = j if j < n else None
                if before >= 0 and after is not None:
                    idx = np.arange(i, j)
                    for axis in range(2):
                        points[i:j, point_idx, axis] = np.interp(
                            idx, [before, after],
                            [points[before, point_idx, axis], points[after, point_idx, axis]],
                        )
                elif before >= 0:
                    points[i:j, point_idx, :] = points[before, point_idx, :]
            i = j
        else:
            i += 1


def track_all_frames(cap, initial_points, args):
    lk_params = dict(
        winSize=CONFIG["lk_win_size"],
        maxLevel=CONFIG["lk_max_level"],
        criteria=CONFIG["lk_criteria"],
    )

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not re-read first frame for tracking.")
    prev_gray = _prep_gray(frame, CONFIG["blur_ksize"])
    prev_pts = initial_points.reshape(-1, 1, 2).astype(np.float32)

    points_list = [initial_points.copy()]
    fb_err_list = [np.zeros(4, dtype=np.float32)]
    bad_mask_list = [np.zeros(4, dtype=bool)]

    hold_count = np.zeros(4, dtype=int)
    held_pts = initial_points.copy()
    recent_fb = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = _prep_gray(frame, CONFIG["blur_ksize"])

        next_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_pts, None, **lk_params)

        fb_err = np.linalg.norm((prev_pts - back_pts).reshape(-1, 2), axis=1)
        bad = (status.reshape(-1) == 0) | (status_b.reshape(-1) == 0) | (fb_err > args.fb_thresh)

        resolved = next_pts.reshape(-1, 2).copy()
        for i in range(4):
            if bad[i]:
                hold_count[i] += 1
                resolved[i] = held_pts[i]
            else:
                hold_count[i] = 0
                held_pts[i] = resolved[i]

        points_list.append(resolved.copy())
        fb_err_list.append(fb_err.astype(np.float32))
        bad_mask_list.append(bad.copy())

        prev_gray = gray
        prev_pts = resolved.reshape(-1, 1, 2).astype(np.float32)

        recent_fb.append(float(fb_err.mean()))
        if len(recent_fb) > 10:
            recent_fb.pop(0)
        if args.reanchor_interval and frame_idx % args.reanchor_interval == 0:
            mean_recent = float(np.mean(recent_fb))
            if mean_recent > CONFIG["drift_warn_px"]:
                print(
                    f"[warn] frame {frame_idx}: mean FB error {mean_recent:.2f}px over the last "
                    f"{len(recent_fb)} frames exceeds the drift-warn threshold "
                    f"({CONFIG['drift_warn_px']}px). Consider re-running with a re-click at this frame."
                )

    points = np.stack(points_list, axis=0)
    fb_err = np.stack(fb_err_list, axis=0)
    bad_mask = np.stack(bad_mask_list, axis=0)

    interpolated_frame = np.zeros(points.shape[0], dtype=bool)
    for i in range(4):
        _fill_point_gaps(points, i, bad_mask[:, i], args.hold_frames, interpolated_frame)

    tracking_ok = ~np.any(bad_mask, axis=1)

    return {
        "points": points.astype(np.float32),
        "tracking_ok": tracking_ok,
        "interpolated": interpolated_frame,
        "fb_err": fb_err,
    }


def _default_debug_path(cache_path):
    p = Path(cache_path)
    stem = p.stem
    if stem.endswith("_points"):
        stem = stem[: -len("_points")]
    return str(p.with_name(stem + "_tracking_debug.csv"))


def write_debug_csv(path, result):
    n = result["points"].shape[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_idx", "fb_err_base_px", "fb_err_mcp_px", "fb_err_pip_px",
            "fb_err_dip_px", "tracking_ok", "interpolated",
        ])
        for i in range(n):
            fb = result["fb_err"][i]
            w.writerow([i, *[f"{v:.3f}" for v in fb],
                        bool(result["tracking_ok"][i]), bool(result["interpolated"][i])])


# ---------------------------------------------------------------------------
# Absolute geometric tip derivation (replaces every prior approach -- the
# distal_anchor LK point, the color+PCA orientation attempt, and the
# rotation-search template matching + checkpoint chain). See the module
# docstring for the full diagnosis of why each of those failed on this rig,
# and why an absolute, per-frame, non-integrating estimate fixes it.
# ---------------------------------------------------------------------------

def derive_blue_range(frame0_bgr, dip0, tip0, config):
    """Auto-sample the distal link's blue in frame 0 -> (lo, hi) HSV bounds for
    cv2.inRange. Sampling per video (rather than a hand-tuned global constant)
    keeps this robust to lighting differences between recordings."""
    hsv0 = cv2.cvtColor(frame0_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv0.shape[:2]
    r = int(config["blue_sample_radius_px"])
    patches = []
    for a in config["blue_sample_fracs"]:
        p = dip0 + a * (tip0 - dip0)
        x, y = int(round(p[0])), int(round(p[1]))
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        patch = hsv0[y0:y1, x0:x1].reshape(-1, 3)
        if patch.size:
            patches.append(patch)
    if not patches:
        raise RuntimeError("Could not sample the distal link's blue color in frame 0.")
    samples = np.concatenate(patches, axis=0).astype(np.float32)
    med_h, med_s, med_v = np.median(samples, axis=0)
    lo = np.array([
        max(0.0, med_h - config["blue_hue_tol"]),
        max(config["blue_sat_floor"], med_s - config["blue_sat_tol"]),
        max(config["blue_val_floor"], med_v - config["blue_val_tol"]),
    ], dtype=np.uint8)
    hi = np.array([min(179.0, med_h + config["blue_hue_tol"]), 255, 255], dtype=np.uint8)
    return lo, hi


def blue_mask(frame_bgr, lo, hi, kernel):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m


def fit_circle(pts):
    """Least-squares (Kasa) circle fit to (N,2) points. Returns (center_xy, R),
    or (None, None) if the fit is degenerate."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None, None
    return np.array([cx, cy], dtype=np.float32), float(np.sqrt(r2))


def finger_component_mask(mask, dip_xy, link_len, config):
    """Keep only the finger's own blue: connected components of ``mask`` that
    reach within ``cc_dip_radius_frac`` x link_len of the tracked dip. Both
    finger links wrap the dip screw so both survive, but DISCONNECTED blue
    clutter is dropped -- critically the blue Arduino board the tip sweeps over,
    which is otherwise a false 'farthest' point. Returns a uint8 mask, or None
    if there is too little blue to trust."""
    ys, xs = np.nonzero(mask)
    if len(xs) < config["min_blue_pixels"]:
        return None
    num, labels = cv2.connectedComponents(mask)
    if num <= 1:
        return mask
    dx = xs.astype(np.float32) - float(dip_xy[0])
    dy = ys.astype(np.float32) - float(dip_xy[1])
    near = (dx * dx + dy * dy) <= (config["cc_dip_radius_frac"] * link_len) ** 2
    keep_labels = np.unique(labels[ys[near], xs[near]])
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return None
    return np.where(np.isin(labels, keep_labels), 255, 0).astype(np.uint8)


def measure_tip_apex(mask, dip_xy, pip_xy, d_prev, link_len, config):
    """Re-detect the tip's rounded edge arc from the blue ``mask`` this frame
    and return ``(apex_xy, unit_dir, corner_dist)`` -- the TRUE extreme corner
    of the distal link, via a circle fit to that arc. Returns None if too
    little of the arc is visible to trust (caller holds the last direction).

    No point is tracked across frames: the arc is re-detected from scratch every
    frame, so nothing can accumulate/drift. Gates (all using the reliably-tracked
    dip/pip) keep only the distal cap edge:
      - the finger's own connected blue (drops the Arduino board etc.);
      - contour points at ``cap_lo_frac``..``cap_hi_frac`` x link_len from the dip;
      - farther from the pip than the dip is (the distal side);
      - within ``cap_cone_deg`` of the previous frame's direction;
      - then the farthest ``100 - cap_far_pct`` percent of those (the rounded cap).
    A circle is fit to the cap; the apex is ``center + R * outward`` (the point on
    that circle farthest from the dip). If the fit is degenerate, it falls back
    to the farthest gated contour point.
    """
    finger = finger_component_mask(mask, dip_xy, link_len, config)
    if finger is None:
        return None
    cnts, _ = cv2.findContours(finger, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    C = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)

    v = C - dip_xy[None, :]
    dist = np.linalg.norm(v, axis=1)
    ann = (dist > config["cap_lo_frac"] * link_len) & (dist < config["cap_hi_frac"] * link_len)
    dip_pip = float(np.linalg.norm(dip_xy - pip_xy))
    distal = np.linalg.norm(C - pip_xy[None, :], axis=1) > (dip_pip * config["distal_pip_frac"])
    u = v / (dist[:, None] + 1e-6)
    cone = (u @ d_prev) > np.cos(np.radians(config["cap_cone_deg"]))
    keep = ann & distal & cone
    if int(keep.sum()) < config["min_cap_points"]:
        return None

    cap_pts, cap_dist = C[keep], dist[keep]
    thr = np.percentile(cap_dist, config["cap_far_pct"])
    sel = cap_dist >= thr
    if int(sel.sum()) >= config["min_cap_points"]:
        cap_pts = cap_pts[sel]

    center, R = fit_circle(cap_pts)
    if center is None or R < config["circle_min_radius_px"] or R > config["circle_max_radius_frac"] * link_len:
        apex = cap_pts[int(np.argmax(np.linalg.norm(cap_pts - dip_xy, axis=1)))]
    else:
        outward = center - dip_xy
        nrm = float(np.linalg.norm(outward))
        apex = center + R * (outward / nrm) if nrm > 1e-6 else center
    apex = np.asarray(apex, dtype=np.float32)
    d = apex - dip_xy
    dd = float(np.linalg.norm(d))
    if dd < 1e-6:
        return None
    return apex, (d / dd).astype(np.float32), dd


def derive_tip_trajectory(video_path, dip_track, pip_track, tip0, config, log=print):
    """Second pass over the video: derive the fingertip every frame as a rigid
    link off the tracked dip, pointed at the tip's true corner as re-detected
    from the blue mask that frame (see module docstring). Returns
    (tip_col (n,2) float32, stats dict).

    Two phases. Phase 1 measures, for every frame, the direction to the circle-
    fit tip apex and that apex's distance from the dip (holding the last
    direction when the arc isn't confidently visible, or when the direction
    would swing non-physically far -- a segmentation outlier). Phase 2 pins the
    radius to the MEDIAN measured corner distance across the accepted frames:
    the link is rigid, so the true corner is a constant distance from the dip;
    pinning to the measured median makes the dip->tip length constant (0% CV),
    kills residual radial measurement noise, and is independent of exactly where
    the frame-0 click landed (which was a few px inboard of the real corner).

    ``dip_track`` / ``pip_track`` are the finalized (n,2) LK trajectories;
    ``tip0`` is the frame-0 tip click. Frame indexing is aligned by reading the
    video from frame 0 and consuming one frame per row.
    """
    n = dip_track.shape[0]
    dip0 = np.asarray(dip_track[0], dtype=np.float32)
    tip0 = np.asarray(tip0, dtype=np.float32)
    link_len = float(np.linalg.norm(tip0 - dip0))  # frame-0 click length; only scales the gates
    if link_len < 1e-3:
        raise RuntimeError("Frame-0 tip and dip coincide -- cannot derive a link length/direction.")
    d_prev = ((tip0 - dip0) / link_len).astype(np.float32)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for tip derivation: {video_path}")
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first frame for blue-range sampling.")

    lo, hi = derive_blue_range(frame0, dip0, tip0, config)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config["morph_kernel_px"], config["morph_kernel_px"]))

    directions = np.zeros((n, 2), dtype=np.float32)
    directions[0] = d_prev
    corner_dist = np.full(n, np.nan, dtype=np.float32)

    n_held = 0
    hold_run = 0
    max_hold_run = 0
    warned_this_run = False

    for t in range(1, n):
        ok, frame_t = cap.read()
        res = None
        if ok:
            mask = blue_mask(frame_t, lo, hi, kernel)
            res = measure_tip_apex(mask, np.asarray(dip_track[t], np.float32),
                                    np.asarray(pip_track[t], np.float32),
                                    d_prev, link_len, config)

        # Reject a non-physical single-frame direction swing as a segmentation
        # outlier (widening the allowance while holding, so it self-recovers).
        reject = False
        if res is not None:
            _apex, d, _dist = res
            step_deg = abs(float(geometry.signed_angle(d_prev, d)))
            allowed = min(config["max_dir_step_deg"] + hold_run * config["dir_widen_per_frame"],
                          config["dir_widen_cap_deg"])
            reject = step_deg > allowed

        if res is None or reject:
            directions[t] = d_prev
            n_held += 1
            hold_run += 1
            max_hold_run = max(max_hold_run, hold_run)
            if hold_run == config["tip_hold_warn_frames"] and not warned_this_run:
                log(f"[warn] frame {t}: the blue tip apex could not be confidently re-detected "
                    f"for {hold_run} consecutive frames; holding the last confident direction.")
                warned_this_run = True
            continue

        _apex, d, dist = res
        d_prev = d
        directions[t] = d
        corner_dist[t] = dist
        hold_run = 0
        warned_this_run = False

    cap.release()

    # Phase 2: rigid pin to the median measured true-corner distance.
    measured = corner_dist[~np.isnan(corner_dist)]
    corner_len = float(np.median(measured)) if measured.size else link_len
    tip_col = (dip_track + corner_len * directions).astype(np.float32)

    stats = {"n_held": n_held, "max_hold_run": max_hold_run,
             "link_len": link_len, "corner_len": corner_len,
             "blue_lo": lo.tolist(), "blue_hi": hi.tolist()}
    return tip_col, stats


def main():
    args = parse_args()

    if args.from_cache:
        # Re-derive ONLY the tip column from an existing cache's already-good
        # base/mcp/pip/dip LK tracks + frame-0 tip click. No click UI, no LK.
        # Read everything out and close the npz before we (possibly) overwrite
        # the same path -- on Windows an open NpzFile locks the file.
        with np.load(args.from_cache) as old:
            old_pts = old["points"]
            if old_pts.shape[1] < 5:
                raise SystemExit(f"'{args.from_cache}' has < 5 point columns; cannot re-derive the tip.")
            lk_points = old_pts[:, :4, :].astype(np.float32)  # base, mcp, pip, dip
            tip0 = old_pts[0, 4, :].astype(np.float32)        # frame-0 tip click
            fps = float(old["fps"])
            tracking_ok = np.array(old["tracking_ok"])
            interpolated = np.array(old["interpolated"])
        result = {
            "points": lk_points,
            "tracking_ok": tracking_ok,
            "interpolated": interpolated,
        }
        print(f"Re-deriving tip from existing cache: {args.from_cache}")
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("Could not read the first frame.")

        initial_points = select_points(frame0, CONFIG)
        lk_initial = initial_points[:4]  # base, mcp, pip, dip -- LK-tracked
        tip0 = initial_points[4]         # frame-0 tip click, calibration only

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        result = track_all_frames(cap, lk_initial, args)
        cap.release()

    pip_idx, dip_idx = 2, 3
    dip_track = result["points"][:, dip_idx, :]
    pip_track = result["points"][:, pip_idx, :]

    tip_col, stats = derive_tip_trajectory(args.video, dip_track, pip_track, tip0, CONFIG)

    full_points = np.concatenate([result["points"], tip_col[:, None, :]], axis=1)

    Path(args.out_cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_cache,
        points=full_points,
        tracking_ok=result["tracking_ok"],
        interpolated=result["interpolated"],
        fps=np.float64(fps),
        n_frames=np.int64(full_points.shape[0]),
        schema_version=np.int64(7),
    )

    if "fb_err" in result:
        debug_path = args.out_debug_csv or _default_debug_path(args.out_cache)
        write_debug_csv(debug_path, result)
    else:
        debug_path = None

    n_frames = result["points"].shape[0]
    pct_interp = 100.0 * result["interpolated"].sum() / max(n_frames, 1)
    pct_held = 100.0 * stats["n_held"] / max(n_frames, 1)
    print(f"Processed {n_frames} frames @ {fps:.2f} fps.")
    print(f"LK interpolated/held frames (base/mcp/pip/dip): "
          f"{result['interpolated'].sum()} ({pct_interp:.1f}%)")
    print(f"Tip derivation: measured true-corner distance L={stats['corner_len']:.1f}px "
          f"(frame-0 click was {stats['link_len']:.1f}px), "
          f"blue HSV {stats['blue_lo']}..{stats['blue_hi']}")
    print(f"Tip apex held/fallback frames: {stats['n_held']} ({pct_held:.1f}%), "
          f"longest consecutive hold: {stats['max_hold_run']} frames.")
    print(f"Cache saved: {args.out_cache}")
    if debug_path:
        print(f"Debug CSV saved: {debug_path}")


if __name__ == "__main__":
    main()
