"""Shared geometry helpers for the finger-sweep video analysis pipeline.

Every script in this folder tracks the same 5 points per frame, in this fixed
order: base (fixed anchor), mcp, pip, dip, tip. ``tip`` is not tracked
directly (a smooth, textureless silhouette point causes optical-flow drift,
and no second reliable corner feature exists near it on this hardware either).
Instead it is derived each frame as an absolute geometric extremum: the tip is
placed at the frame-0 clicked link length off the reliably-tracked ``dip``,
pointed along the distal segment's direction measured that frame from the blue
mask -- no reference frame, no rotation integration, so nothing accumulates
(see ``track_points.py``'s module docstring for the full diagnosis of the four
earlier approaches that failed on this hardware, including why a rotation-search
template-matching + checkpoint-chain attempt drifted at the ends of the sweep).
Angles are signed, in degrees, flexion-positive, and zeroed against frame 0
of whichever video is being processed.
"""

from __future__ import annotations

import numpy as np

POINT_NAMES = ("base", "mcp", "pip", "dip", "tip")
JOINT_NAMES = ("mcp", "pip", "dip")


def wrap_deg(angle):
    """Wrap an angle (degrees), scalar or array, into the half-open (-180, 180]."""
    arr = np.asarray(angle, dtype=float)
    wrapped = (arr + 180.0) % 360.0 - 180.0
    wrapped = np.where(wrapped == -180.0, 180.0, wrapped)
    return float(wrapped) if arr.ndim == 0 else wrapped


def signed_angle(v_prev, v_next):
    """Signed bend (degrees) turning from v_prev to v_next.

    Uses atan2(cross, dot) rather than arccos(dot) so the sign of the turn
    (which way the joint is bending) is preserved. Broadcasts over the last
    axis, so it works on single vectors (shape (2,)) or batches (shape (N,2)).
    """
    v_prev = np.asarray(v_prev, dtype=float)
    v_next = np.asarray(v_next, dtype=float)
    cross = v_prev[..., 0] * v_next[..., 1] - v_prev[..., 1] * v_next[..., 0]
    dot = v_prev[..., 0] * v_next[..., 0] + v_prev[..., 1] * v_next[..., 1]
    return np.degrees(np.arctan2(cross, dot))


def compute_joint_angles(points_xy, flexion_sign=1):
    """Compute zeroed, flexion-signed MCP/PIP/DIP angles for every frame.

    Parameters
    ----------
    points_xy : ndarray, shape (n_frames, len(POINT_NAMES), 2)
        Per-frame tracked pixel coordinates, in POINT_NAMES order (base, mcp,
        pip, dip, tip). Any extra columns beyond these 5, should a future
        cache schema carry them, would be ignored here since columns are
        selected by name via POINT_NAMES.index(...).
    flexion_sign : {1, -1}
        Flip to -1 if flexion reads negative on a frame you can visually
        confirm is flexing (camera-side dependent).

    Returns
    -------
    dict mapping "mcp"/"pip"/"dip" -> (n_frames,) ndarray of angles (degrees).
    """
    pts = np.asarray(points_xy, dtype=float)
    idx = [POINT_NAMES.index(n) for n in ("base", "mcp", "pip", "dip", "tip")]
    base, mcp, pip_, dip, tip = (pts[:, i, :] for i in idx)

    v0 = mcp - base
    v1 = pip_ - mcp
    v2 = dip - pip_
    v3 = tip - dip

    raw = {
        "mcp": signed_angle(v0, v1),
        "pip": signed_angle(v1, v2),
        "dip": signed_angle(v2, v3),
    }
    zero = {k: v[0] for k, v in raw.items()}

    return {k: flexion_sign * wrap_deg(raw[k] - zero[k]) for k in JOINT_NAMES}


def detect_motion_window(angles, eps_deg=0.15, min_run=5):
    """Bound the sustained-motion region of a per-frame angle series.

    Parameters
    ----------
    angles : dict of "mcp"/"pip"/"dip" -> (n_frames,) ndarray (as returned by
        compute_joint_angles).
    eps_deg : float
        Frame-to-frame delta below this is treated as tracking noise, not motion.
    min_run : int
        Minimum number of consecutive above-threshold frames to count as a
        real motion run (guards against single-frame tracking jitter).

    Returns
    -------
    (motion_start_idx, motion_end_idx) : the frame-index bounds (inclusive)
    of the first-to-last sustained motion run. Falls back to the full range
    if no run of at least ``min_run`` frames is found.
    """
    stacked = np.stack([angles[k] for k in JOINT_NAMES], axis=0)  # (3, n)
    n = stacked.shape[1]
    if n < 2:
        return 0, max(n - 1, 0)

    delta = np.max(np.abs(np.diff(stacked, axis=1)), axis=0)  # (n-1,)
    active = delta > eps_deg

    run_starts, run_ends = [], []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) >= min_run:
                run_starts.append(i)
                run_ends.append(j - 1)
            i = j
        else:
            i += 1

    if not run_starts:
        return 0, n - 1

    motion_start = run_starts[0]
    motion_end = min(run_ends[-1] + 1, n - 1)
    return int(motion_start), int(motion_end)


def dedup_rows(frame_idx, times, angles, eps_deg=0.15):
    """Collapse consecutive near-duplicate rows, keeping only change points.

    Parameters
    ----------
    frame_idx, times : 1-D ndarrays, same length, already restricted to
        whatever window should be kept (e.g. the output of detect_motion_window).
    angles : dict of "mcp"/"pip"/"dip" -> 1-D ndarray, same length as frame_idx.
    eps_deg : float
        A row is kept only if it differs from the last *kept* row by more
        than this, in at least one joint. The first and last row of the
        window are always kept.

    Returns
    -------
    dict with keys "frame_idx", "time_s", "mcp_deg", "pip_deg", "dip_deg",
    each a 1-D ndarray of the kept rows.
    """
    n = len(frame_idx)
    empty = lambda: np.array([])  # noqa: E731
    if n == 0:
        return {"frame_idx": empty(), "time_s": empty(),
                "mcp_deg": empty(), "pip_deg": empty(), "dip_deg": empty()}

    stacked = np.stack([angles[k] for k in JOINT_NAMES], axis=1)  # (n, 3)
    keep = [0]
    last_kept = stacked[0]
    for i in range(1, n):
        if np.max(np.abs(stacked[i] - last_kept)) > eps_deg:
            keep.append(i)
            last_kept = stacked[i]
    if keep[-1] != n - 1:
        keep.append(n - 1)

    keep = np.array(keep, dtype=int)
    frame_idx = np.asarray(frame_idx)
    times = np.asarray(times)
    return {
        "frame_idx": frame_idx[keep],
        "time_s": times[keep],
        "mcp_deg": angles["mcp"][keep],
        "pip_deg": angles["pip"][keep],
        "dip_deg": angles["dip"][keep],
    }


def polygon_area(points_xy):
    """Shoelace-formula unsigned area (px^2) of a simple polygon.

    points_xy : ndarray (n, 2). Does not need to repeat the first point at
    the end -- the wraparound edge is added implicitly.
    """
    pts = np.asarray(points_xy, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


__all__ = [
    "POINT_NAMES",
    "JOINT_NAMES",
    "wrap_deg",
    "signed_angle",
    "compute_joint_angles",
    "detect_motion_window",
    "dedup_rows",
    "polygon_area",
]
