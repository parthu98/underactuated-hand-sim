#!/usr/bin/env python3
"""Standalone ΔL pull-cycle sweep for the finger servo(s).

No camera, no Qt, no load cell: opens the U2D2 bus, binds whichever of the two
finger servos (``config.FINGER_A_DXL_ID`` / ``FINGER_B_DXL_ID``) actually
answer -- one or both -- and repeatedly ramps ΔL between ``--min-delta`` and
``--max-delta`` mm, dwelling at each end. For tendon wear-in / mocap-capture
runs where no analytical comparison or per-trial capture is needed, just a
repeatable pull-release motion.

Run::

    python3 servo_sweep.py                       # auto: use whatever answers
    python3 servo_sweep.py --servo a              # only finger A
    python3 servo_sweep.py --servo both --cycles 10 --speed 8 --max-delta 40
    python3 servo_sweep.py --mock                 # no hardware
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import servo  # noqa: E402  (module import so the patch below sticks -- autodetect
              # / open_bus call the module-level enumerator, not an attribute copy)
from servo import MockServo, Servo, install_emergency_shutdown, open_bus  # noqa: E402


# --- cross-platform serial-port enumeration ---------------------------------
# servo.autodetect_servo / open_bus scan every port + baud, so port="auto" binds
# the U2D2 regardless of which port it landed on. The one gap is that
# servo.list_serial_ports() globs the Linux /dev/ttyUSB*/ttyACM* paths; on
# Windows the U2D2 enumerates as COMx. Patch the enumerator in-process (servo.py
# itself is untouched) to use pyserial's cross-platform comports(), falling back
# to the Linux globs.
def _list_serial_ports_xplat():
    try:
        from serial.tools import list_ports
        ports = sorted(p.device for p in list_ports.comports())
        if ports:
            return ports
    except Exception:  # noqa: BLE001
        pass
    import glob
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


servo.list_serial_ports = _list_serial_ports_xplat


FINGER_IDS = {"a": config.FINGER_A_DXL_ID, "b": config.FINGER_B_DXL_ID}
PULL_SIGNS = {config.FINGER_A_DXL_ID: config.FINGER_A_PULL_SIGN,
              config.FINGER_B_DXL_ID: config.FINGER_B_PULL_SIGN}
NAMES = {config.FINGER_A_DXL_ID: "A", config.FINGER_B_DXL_ID: "B"}


def connect(args):
    """Connect to whichever finger servo(s) ``--servo`` asks for.

    ``auto`` binds every finger id that answers on the bus (one or both);
    ``a``/``b`` bind just that one; ``both`` requires both to answer. Returns
    the list of connected ``Servo``/``MockServo`` instances.
    """
    wanted = (list(FINGER_IDS.values()) if args.servo in ("auto", "both")
              else [FINGER_IDS[args.servo]])

    if args.mock:
        # MockServo's synthetic current (120 + 90*|delta_mm|) is tuned against
        # its own 25 mm default cap; a wider sweep range needs a matching
        # headroom bump here or the demo nuisance-trips the (also synthetic)
        # over-current e-stop. Real Servo telemetry is unaffected -- its
        # overcurrent_warn_ma stays at the class default (a real safety limit).
        mock_warn_ma = max(3000.0, 120.0 + 90.0 * args.max_delta + 500.0)
        servos = [MockServo(dxl_id=i, spool_radius_m=config.SPOOL_RADIUS,
                            soft_delta_l_cap_mm=args.max_delta,
                            overcurrent_warn_ma=mock_warn_ma,
                            pull_sign=PULL_SIGNS[i])
                  for i in wanted]
        for s in servos:
            s.connect()
            print(f"finger {NAMES[s.dxl_id]} (id {s.dxl_id}) connected (mock)")
        return servos

    ok, ph, pk, port, baud, ids, msg = open_bus(
        port=args.port, baud=args.baud, expected_ids=None)
    if not ok:
        raise SystemExit(f"bus open failed: {msg}")

    present = [i for i in wanted if i in ids]
    if args.servo == "both" and len(present) < 2:
        missing = [NAMES[i] for i in wanted if i not in present]
        raise SystemExit(f"--servo both requested but {missing} not found on "
                         f"{port} @ {baud} baud (saw ids {ids})")
    if not present:
        raise SystemExit(f"no finger servo found on {port} @ {baud} baud "
                         f"(wanted ids {wanted}, saw {ids})")

    servos = []
    for i in present:
        s = Servo(dxl_id=i, spool_radius_m=config.SPOOL_RADIUS,
                 soft_delta_l_cap_mm=args.max_delta, pull_sign=PULL_SIGNS[i])
        s.attach_bus(ph, pk, port, baud)
        ok, cmsg = s.connect()
        if not ok:
            raise SystemExit(f"finger {NAMES[i]} connect failed: {cmsg}")
        print(f"finger {NAMES[i]} (id {i}) connected on {port} @ {baud} baud")
        servos.append(s)
    return servos


def _drive_to(servos, target_mm, speed_mm_s, dt=0.05):
    """Ramp every servo to ``target_mm`` together. False on refusal/e-stop."""
    for s in servos:
        if not s.start_ramp(target_mm, speed_mm_s):
            print(f"refused ramp to {target_mm:.1f} mm (e-stop or soft cap) -- stopping")
            return False
    while any(s.is_ramping() for s in servos):
        for s in servos:
            if s.service(dt)["estop"]:
                return False
        time.sleep(dt)
    for s in servos:
        s.service(dt)
    return True


def _hold(servos, hold_s, dt=0.05):
    elapsed = 0.0
    while elapsed < hold_s:
        for s in servos:
            if s.service(dt)["estop"]:
                return False
        time.sleep(dt)
        elapsed += dt
    return True


def sweep(servos, args):
    cycle = 0
    while args.cycles <= 0 or cycle < args.cycles:
        cycle += 1
        tag = f"{cycle}" if args.cycles <= 0 else f"{cycle}/{args.cycles}"
        print(f"cycle {tag}: -> {args.max_delta:.1f} mm")
        if not (_drive_to(servos, args.max_delta, args.speed)
                and _hold(servos, args.hold)):
            print("stopped: e-stop latched or ramp refused")
            return
        print(f"cycle {tag}: -> {args.min_delta:.1f} mm")
        if not (_drive_to(servos, args.min_delta, args.speed)
                and _hold(servos, args.hold)):
            print("stopped: e-stop latched or ramp refused")
            return


def main():
    p = argparse.ArgumentParser(description="delta-L pull-cycle sweep for the finger servo(s)")
    p.add_argument("--servo", choices=["auto", "a", "b", "both"], default="auto",
                   help="which finger servo(s) to drive (default: auto = whatever answers)")
    p.add_argument("--port", default="auto", help="U2D2 serial port (default: auto-scan)")
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--min-delta", type=float, default=0.0, help="delta-L low end [mm]")
    p.add_argument("--max-delta", type=float, default=config.GRIPPER_MAX_PULL_MM,
                   help=f"delta-L high end [mm] (default {config.GRIPPER_MAX_PULL_MM})")
    p.add_argument("--speed", type=float, default=5.0, help="ramp speed [mm/s]")
    p.add_argument("--hold", type=float, default=0.5, help="dwell at each end [s]")
    p.add_argument("--cycles", type=int, default=0,
                   help="number of pull/release cycles (0 = run until Ctrl-C)")
    p.add_argument("--mock", action="store_true", help="no hardware")
    args = p.parse_args()

    servos = connect(args)

    def _shutdown():
        for s in servos:
            try:
                s.disable()
                s.disconnect()
            except Exception:
                pass

    install_emergency_shutdown(_shutdown)

    try:
        sweep(servos, args)
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
