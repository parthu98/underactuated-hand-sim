#!/usr/bin/env python3
"""Flexion-plane CHECK utility for the PhaseSpace finger tracker.

The finger flexes in a FIXED horizontal plane (no abduction), so the flexion
plane is defined by construction from ``MOCAP_VERTICAL_AXIS`` /
``MOCAP_VERTICAL_SIGN`` (see mocap_config.py) — there is no calibration flex to
run any more. This script just helps you CONFIRM those two settings on the live
system: it streams for a few seconds and prints

  * the per-marker mean position + which axis varies least across the markers
    laid flat (that least-varying axis is your vertical / plane normal), and
  * the resulting plane basis (n, u_axis, u_perp) and the live per-segment angles.

Run:
    python mocap/calibrate.py --mock              # synthetic, for a dry run
    python mocap/calibrate.py --seconds 5         # real PhaseSpace, 5 s sample

Lay the finger STRAIGHT and flat while sampling. If the printed normal axis does
not match MOCAP_VERTICAL_AXIS, edit mocap_config.py accordingly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mocap_config as mcfg  # noqa: E402
from tracker import build_tracker  # noqa: E402

_AXES = ("X", "Y", "Z")


def main() -> int:
    p = argparse.ArgumentParser(description="PhaseSpace flexion-plane check")
    p.add_argument("--mock", action="store_true", help="synthetic tracker (dry run)")
    p.add_argument("--server", default=mcfg.MOCAP_SERVER_IP)
    p.add_argument("--no-slave", action="store_true")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="how long to sample (default 5 s)")
    p.add_argument("--rate", type=float, default=30.0, help="sample rate [Hz]")
    args = p.parse_args()

    tracker = build_tracker(
        mock=args.mock, server=args.server,
        segment_marker_ids=mcfg.MOCAP_SEGMENT_MARKER_IDS,
        calib_path=mcfg.MOCAP_CALIB_PATH,
        timeout_us=mcfg.MOCAP_EVENT_TIMEOUT_US,
        slave=(not args.no_slave) and mcfg.MOCAP_SLAVE,
        vertical_axis=mcfg.MOCAP_VERTICAL_AXIS,
        vertical_sign=mcfg.MOCAP_VERTICAL_SIGN,
    )

    print(f"Connecting to {'MOCK' if args.mock else args.server} ...")
    tracker.start()
    time.sleep(0.5)  # let the stream warm up

    print(f"Sampling for {args.seconds:.0f} s - hold the finger STRAIGHT and FLAT.")
    samples = {mid: [] for pair in mcfg.MOCAP_SEGMENT_MARKER_IDS for mid in pair}
    dt = 1.0 / max(1.0, args.rate)
    t_end = time.monotonic() + args.seconds
    while time.monotonic() < t_end:
        snap = tracker._snapshot()
        for mid in samples:
            m = snap.get(mid)
            if m is not None and m.cond > 0:
                samples[mid].append((m.x, m.y, m.z))
        time.sleep(dt)

    print("\nPer-marker mean position [mm]:")
    means = {}
    for mid, pts in sorted(samples.items()):
        if not pts:
            print(f"  id {mid:2d}: (never seen)")
            continue
        means[mid] = np.mean(pts, axis=0)
        x, y, z = means[mid]
        print(f"  id {mid:2d}: X={x:9.2f}  Y={y:9.2f}  Z={z:9.2f}  ({len(pts)} samples)")

    if len(means) >= 3:
        # Markers laid flat are ~coplanar; the axis with the SMALLEST spread
        # across all markers is the plane normal (your vertical axis).
        spread = np.std(np.array(list(means.values())), axis=0)
        normal_axis = int(np.argmin(spread))
        print(f"\nSpread across markers: X={spread[0]:.2f}  Y={spread[1]:.2f}  "
              f"Z={spread[2]:.2f}  ->  least-varying axis = {_AXES[normal_axis]}")
        print(f"Configured MOCAP_VERTICAL_AXIS = {mcfg.MOCAP_VERTICAL_AXIS} "
              f"({_AXES[mcfg.MOCAP_VERTICAL_AXIS]})", end="")
        if normal_axis == mcfg.MOCAP_VERTICAL_AXIS:
            print("  [OK match]")
        else:
            print(f"  [MISMATCH -> consider setting it to {normal_axis}]")

    det = tracker.detect()  # refines u_axis from the live base markers

    print("\nPlane basis in use:")
    print(f"  n      = {np.round(tracker.n, 4)}")
    print(f"  u_axis = {np.round(tracker.u_axis, 4)}")
    print(f"  u_perp = {np.round(tracker.u_perp, 4)}")

    print("\nLive per-segment in-plane angle phi [deg]:")
    for si, lbl in enumerate(mcfg.SEGMENT_LABELS):
        a = det["phi"].get(si)
        print(f"  {lbl:>4s}: {'(lost)' if a is None else f'{a:+7.2f}'}")

    tracker.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
