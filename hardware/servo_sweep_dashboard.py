#!/usr/bin/env python3
"""GUI counterpart to servo_sweep.py -- connect, jog-tension by hand, then run
the delta-L pull-cycle sweep, all from one PySide6 window.

No camera, no CSV logging, no analytical comparison -- just whichever finger
servo(s) answer (config.FINGER_A_DXL_ID / FINGER_B_DXL_ID, one or both), a
manual jog + SET ZERO to tension the tendon by hand, a manual delta-L GO, and
an AUTO SWEEP that cycles delta-L between --min-delta and --max-delta.

Torque is OFF until CONNECT is pressed -- launching this script alone never
moves the motor(s); nothing is auto-connected or auto-run.

Run::

    python3 servo_sweep_dashboard.py                # real hardware, auto-detect
    python3 servo_sweep_dashboard.py --servo a       # only finger A
    python3 servo_sweep_dashboard.py --mock          # no hardware
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
from dashboard import _btn, _dspin, _group, _JogKeyFilter  # noqa: E402  (reuse styling)
from servo import MockServo, Servo, install_emergency_shutdown, open_bus  # noqa: E402

from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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

TICK_MS = 50  # 20 Hz main loop

FINGER_IDS = {"a": config.FINGER_A_DXL_ID, "b": config.FINGER_B_DXL_ID}
PULL_SIGNS = {config.FINGER_A_DXL_ID: config.FINGER_A_PULL_SIGN,
              config.FINGER_B_DXL_ID: config.FINGER_B_PULL_SIGN}
NAMES = {config.FINGER_A_DXL_ID: "A", config.FINGER_B_DXL_ID: "B"}


class SweepDashboard(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.servos = []          # connected Servo/MockServo instances
        self._bus_handles = None  # (port_handler, packet_handler) for real hw
        self._jog_dir = 0
        self._t_prev = time.monotonic()

        # Auto-sweep state machine -- advanced every tick, never blocks the UI.
        self._sweep_active = False
        self._sweep_phase = "idle"   # idle | to_high | hold_high | to_low | hold_low
        self._sweep_hold_until = 0.0
        self._sweep_cycle = 0

        self.setWindowTitle("Finger Servo Sweep -- IITGN")
        self.setStyleSheet("background-color:#0d1117; color:#c9d1d9;")
        self.setMinimumSize(560, 700)
        self.setFocusPolicy(Qt.StrongFocus)

        self._build_ui()

        # App-wide key filter so the jog keys survive focus changes (reused
        # from dashboard.py -- it already skips number fields being edited).
        self._key_filter = _JogKeyFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self._key_filter)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(TICK_MS)

        # Kill the servo(s) on ANY exit (Ctrl-C / SIGTERM too), not just a
        # clicked window close. Idempotent: safe from closeEvent + atexit + signal.
        self._shutdown_done = False
        install_emergency_shutdown(self._shutdown)

    def _grab_focus(self):
        cw = self.centralWidget()
        if cw is not None:
            cw.setFocus()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        central.setFocusPolicy(Qt.StrongFocus)
        root = QVBoxLayout(central)

        # ---- connect ----
        g = _group("CONNECT", "#58a6ff")
        lay = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_connect = _btn("CONNECT", "#0d1117", "#58a6ff", "#58a6ff")
        self.btn_connect.clicked.connect(self._do_connect)
        row.addWidget(self.btn_connect)
        self.btn_estop = _btn("E-STOP", "white", "#da3633", "#da3633")
        self.btn_estop.clicked.connect(self._do_estop)
        row.addWidget(self.btn_estop)
        lay.addLayout(row)
        self.lbl_conn = QLabel(f"not connected -- torque is OFF  "
                               f"(--servo {self.args.servo}"
                               f"{' --mock' if self.args.mock else ''})")
        self.lbl_conn.setStyleSheet("color:#8b949e;")
        lay.addWidget(self.lbl_conn)
        g.setLayout(lay)
        root.addWidget(g)

        # ---- jog & zero ----
        g = _group("JOG  &  ZERO  (tension by hand)", "#ffd33d")
        lay = QVBoxLayout()
        hint = QLabel("←/A = CCW   →/D = CW (hold)   ↑/W = step+   "
                      "↓/S = step-   Space = E-STOP")
        hint.setStyleSheet("color:#8b949e; font-size:10px;")
        lay.addWidget(hint)
        row = QHBoxLayout()
        self.btn_ccw = _btn("◀ CCW")
        self.btn_cw = _btn("CW ▶")
        self.btn_ccw.pressed.connect(lambda: self._set_jog(-1))
        self.btn_ccw.released.connect(lambda: self._set_jog(0))
        self.btn_cw.pressed.connect(lambda: self._set_jog(+1))
        self.btn_cw.released.connect(lambda: self._set_jog(0))
        row.addWidget(self.btn_ccw)
        row.addWidget(self.btn_cw)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("step (rev)"))
        self.jog_step = _dspin(0.001, 0.5, 0.01, 0.001)
        row2.addWidget(self.jog_step)
        self.btn_zero = _btn("◎ SET ZERO", "#0d1117", "#ffd33d", "#ffd33d")
        self.btn_zero.clicked.connect(self._set_zero)
        row2.addWidget(self.btn_zero)
        lay.addLayout(row2)
        row3 = QHBoxLayout()
        self.btn_flip_a = _btn("flip A dir")
        self.btn_flip_b = _btn("flip B dir")
        self.btn_flip_a.clicked.connect(lambda: self._flip_dir(config.FINGER_A_DXL_ID))
        self.btn_flip_b.clicked.connect(lambda: self._flip_dir(config.FINGER_B_DXL_ID))
        row3.addWidget(self.btn_flip_a)
        row3.addWidget(self.btn_flip_b)
        lay.addLayout(row3)
        g.setLayout(lay)
        root.addWidget(g)

        # ---- manual delta-L go ----
        g = _group("MANUAL delta-L", "#79c0ff")
        lay = QHBoxLayout()
        lay.addWidget(QLabel("target"))
        self.delta_spin = _dspin(0.0, max(100.0, self.args.max_delta), 0.0, 1.0, " mm")
        lay.addWidget(self.delta_spin)
        lay.addWidget(QLabel("speed"))
        self.speed_spin = _dspin(0.2, 20.0, self.args.speed, 0.5, " mm/s")
        lay.addWidget(self.speed_spin)
        self.btn_go = _btn("▶ GO", "#0d1117", "#79c0ff", "#79c0ff")
        self.btn_go.clicked.connect(self._go)
        lay.addWidget(self.btn_go)
        g.setLayout(lay)
        root.addWidget(g)

        # ---- auto sweep ----
        g = _group("AUTO SWEEP", "#56d364")
        lay = QVBoxLayout()
        grid = QGridLayout()
        grid.addWidget(QLabel("min (mm)"), 0, 0)
        self.min_spin = _dspin(0.0, max(100.0, self.args.max_delta), self.args.min_delta, 1.0)
        grid.addWidget(self.min_spin, 0, 1)
        grid.addWidget(QLabel("max (mm)"), 0, 2)
        self.max_spin = _dspin(0.0, max(100.0, self.args.max_delta), self.args.max_delta, 1.0)
        grid.addWidget(self.max_spin, 0, 3)
        grid.addWidget(QLabel("step (mm)"), 1, 0)
        self.step_spin = _dspin(0.1, 50.0, self.args.step, 0.5, " mm")
        grid.addWidget(self.step_spin, 1, 1)
        grid.addWidget(QLabel("speed (mm/s)"), 1, 2)
        self.sweep_speed_spin = _dspin(0.2, 20.0, self.args.speed, 0.5)
        grid.addWidget(self.sweep_speed_spin, 1, 3)
        grid.addWidget(QLabel("stop time (s)"), 2, 0)
        self.stop_spin = _dspin(0.0, 60.0, self.args.hold, 0.1, " s")
        grid.addWidget(self.stop_spin, 2, 1)
        grid.addWidget(QLabel("cycles (0=inf)"), 2, 2)
        self.cycles_spin = QSpinBox()
        self.cycles_spin.setRange(0, 100000)
        self.cycles_spin.setValue(self.args.cycles)
        self.cycles_spin.setStyleSheet("QSpinBox{background:#0d1117;color:#c9d1d9;"
                                       "border:1px solid #30363d;padding:2px;}")
        grid.addWidget(self.cycles_spin, 2, 3)
        lay.addLayout(grid)
        hint = QLabel("steps delta-L up from min to max in 'step' increments, "
                      "dwelling 'stop time' at each stop, then back down")
        hint.setStyleSheet("color:#8b949e; font-size:10px;")
        lay.addWidget(hint)
        row = QHBoxLayout()
        self.btn_sweep_start = _btn("▶ START SWEEP", "#0d1117", "#56d364", "#56d364")
        self.btn_sweep_stop = _btn("■ STOP SWEEP", "#0d1117", "#f85149", "#f85149")
        self.btn_sweep_start.clicked.connect(self._start_sweep)
        self.btn_sweep_stop.clicked.connect(self._stop_sweep)
        row.addWidget(self.btn_sweep_start)
        row.addWidget(self.btn_sweep_stop)
        lay.addLayout(row)
        self.lbl_sweep = QLabel("sweep idle")
        self.lbl_sweep.setStyleSheet("color:#8b949e;")
        lay.addWidget(self.lbl_sweep)
        g.setLayout(lay)
        root.addWidget(g)

        # ---- readouts ----
        g = _group("READOUTS", "#58a6ff")
        self.readout_lay = QVBoxLayout()
        self.lbl_no_servo = QLabel("(connect to see readouts)")
        self.lbl_no_servo.setStyleSheet("color:#6e7681;")
        self.readout_lay.addWidget(self.lbl_no_servo)
        g.setLayout(self.readout_lay)
        root.addWidget(g)

        root.addStretch()
        self.setCentralWidget(central)

        self.ro_labels = {}
        self.lbl_estop = None
        self._set_controls_enabled(False)

    def _set_controls_enabled(self, on):
        for w in (self.btn_ccw, self.btn_cw, self.btn_zero, self.btn_go,
                 self.btn_sweep_start, self.btn_sweep_stop,
                 self.btn_flip_a, self.btn_flip_b):
            w.setEnabled(on)

    # -----------------------------------------------------------------
    # connect
    # -----------------------------------------------------------------
    def _wanted_ids(self):
        if self.args.servo in ("auto", "both"):
            return list(FINGER_IDS.values())
        return [FINGER_IDS[self.args.servo]]

    def _do_connect(self):
        if self.servos:
            return
        wanted = self._wanted_ids()
        cap = max(self.args.max_delta, 100.0)

        if self.args.mock:
            mock_warn_ma = max(3000.0, 120.0 + 90.0 * cap + 500.0)
            servos = [MockServo(dxl_id=i, spool_radius_m=config.SPOOL_RADIUS,
                                soft_delta_l_cap_mm=cap,
                                overcurrent_warn_ma=mock_warn_ma,
                                pull_sign=PULL_SIGNS[i])
                     for i in wanted]
            for s in servos:
                s.connect()
            self.servos = servos
            self._finish_connect()
            return

        ok, ph, pk, port, baud, ids, msg = open_bus(
            port=self.args.port, baud=self.args.baud, expected_ids=None)
        if not ok:
            QMessageBox.critical(self, "Connect", msg)
            return

        present = [i for i in wanted if i in ids]
        if self.args.servo == "both" and len(present) < 2:
            missing = [NAMES[i] for i in wanted if i not in present]
            QMessageBox.critical(self, "Connect",
                f"--servo both requested but {missing} not found on "
                f"{port} @ {baud} baud (saw ids {ids})")
            try:
                ph.closePort()
            except Exception:
                pass
            return
        if not present:
            QMessageBox.critical(self, "Connect",
                f"no finger servo found on {port} @ {baud} baud "
                f"(wanted ids {wanted}, saw {ids})")
            try:
                ph.closePort()
            except Exception:
                pass
            return

        servos = []
        for i in present:
            s = Servo(dxl_id=i, spool_radius_m=config.SPOOL_RADIUS,
                     soft_delta_l_cap_mm=cap, pull_sign=PULL_SIGNS[i])
            s.attach_bus(ph, pk, port, baud)
            ok, cmsg = s.connect()
            if not ok:
                QMessageBox.critical(self, "Connect",
                    f"finger {NAMES[i]} connect failed: {cmsg}")
                for done in servos:
                    done.disconnect()
                try:
                    ph.closePort()
                except Exception:
                    pass
                return
            servos.append(s)

        self._bus_handles = (ph, pk)
        self.servos = servos
        self._finish_connect()

    def _finish_connect(self):
        names = ", ".join(NAMES.get(s.dxl_id, str(s.dxl_id)) for s in self.servos)
        self.lbl_conn.setText(f"connected: {names}  -- torque ON")
        self.btn_connect.setText(f"CONNECTED ({names})")
        self.btn_connect.setEnabled(False)
        self._build_readout_rows()
        self._set_controls_enabled(True)
        self._grab_focus()

    def _build_readout_rows(self):
        while self.readout_lay.count():
            item = self.readout_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.ro_labels = {}
        for s in self.servos:
            name = NAMES.get(s.dxl_id, str(s.dxl_id))
            lab = QLabel(f"{name}: -- mm   -- mA")
            lab.setStyleSheet("color:#c9d1d9; font-family:Consolas;")
            self.readout_lay.addWidget(lab)
            self.ro_labels[s.dxl_id] = lab
        self.lbl_estop = QLabel("")
        self.lbl_estop.setStyleSheet("color:#f85149; font-weight:bold;")
        self.readout_lay.addWidget(self.lbl_estop)

    # -----------------------------------------------------------------
    # jog / zero / manual go
    # -----------------------------------------------------------------
    def _set_jog(self, d):
        self._jog_dir = d

    def _jog_step_up(self):
        self.jog_step.setValue(min(self.jog_step.value() * 2, self.jog_step.maximum()))

    def _jog_step_down(self):
        self.jog_step.setValue(max(self.jog_step.value() / 2, self.jog_step.minimum()))

    def _set_zero(self):
        for s in self.servos:
            if s.get_state().get("connected"):
                s.set_zero()

    def _flip_dir(self, dxl_id):
        s = next((s for s in self.servos if s.dxl_id == dxl_id), None)
        if s is None:
            return
        new_sign = -1 if getattr(s, "pull_sign", 1) > 0 else +1
        s.set_pull_direction(new_sign)
        self.statusBar().showMessage(
            f"{NAMES.get(dxl_id, dxl_id)} pull_sign = {new_sign} (re-jog to confirm)", 3000)

    def _go(self):
        target = float(self.delta_spin.value())
        speed = float(self.speed_spin.value())
        for s in self.servos:
            s.start_ramp(target, speed)

    def _do_estop(self):
        for s in self.servos:
            s.e_stop()
        self._jog_dir = 0
        self._sweep_active = False
        self._sweep_phase = "idle"
        self.lbl_sweep.setText("sweep stopped: E-STOP")
        self.lbl_sweep.setStyleSheet("color:#f85149; font-weight:bold;")

    # -----------------------------------------------------------------
    # auto sweep (non-blocking state machine, advanced from _tick)
    # -----------------------------------------------------------------
    @staticmethod
    def _build_waypoints(min_mm, max_mm, step_mm):
        """Staircase min -> max in ``step_mm`` increments, then back down.

        E.g. min=0, max=40, step=5 -> [0,5,...,40,35,...,5,0]. The last
        ascending increment is clamped to land exactly on ``max_mm`` even if
        it doesn't divide evenly by ``step_mm``.
        """
        lo, hi = min(min_mm, max_mm), max(min_mm, max_mm)
        step = max(abs(step_mm), 1e-6)
        up = []
        v = lo
        while v < hi - 1e-9:
            up.append(round(v, 6))
            v += step
        up.append(hi)
        down = list(reversed(up[:-1]))
        return up + down

    def _start_sweep(self):
        if not self.servos:
            return
        self._sweep_waypoints = self._build_waypoints(
            float(self.min_spin.value()), float(self.max_spin.value()),
            float(self.step_spin.value()))
        self._sweep_speed = float(self.sweep_speed_spin.value())
        self._sweep_stop_s = float(self.stop_spin.value())
        self._sweep_cycles_target = int(self.cycles_spin.value())
        self._sweep_cycle = 0
        self._sweep_idx = 0
        target = self._sweep_waypoints[0]
        for s in self.servos:
            s.start_ramp(target, self._sweep_speed)
        self._sweep_phase = "ramp"
        self._sweep_active = True
        self.lbl_sweep.setText(
            f"cycle 1, stop 1/{len(self._sweep_waypoints)}: -> {target:.1f} mm")
        self.lbl_sweep.setStyleSheet("color:#56d364; font-weight:bold;")

    def _stop_sweep(self):
        self._sweep_active = False
        self._sweep_phase = "idle"
        self.lbl_sweep.setText("sweep idle")
        self.lbl_sweep.setStyleSheet("color:#8b949e;")

    def _advance_sweep(self, now):
        if not self._sweep_active:
            return
        if any(s.get_state().get("estop") for s in self.servos):
            self._sweep_active = False
            self._sweep_phase = "idle"
            self.lbl_sweep.setText("sweep stopped: E-STOP")
            self.lbl_sweep.setStyleSheet("color:#f85149; font-weight:bold;")
            return

        ramping = any(s.is_ramping() for s in self.servos)
        n = len(self._sweep_waypoints)
        if self._sweep_phase == "ramp" and not ramping:
            self._sweep_phase = "hold"
            self._sweep_hold_until = now + self._sweep_stop_s
        elif self._sweep_phase == "hold" and now >= self._sweep_hold_until:
            self._sweep_idx += 1
            if self._sweep_idx >= n:
                self._sweep_idx = 0
                self._sweep_cycle += 1
                if (self._sweep_cycles_target > 0
                        and self._sweep_cycle >= self._sweep_cycles_target):
                    self._sweep_active = False
                    self._sweep_phase = "idle"
                    self.lbl_sweep.setText(f"sweep done: {self._sweep_cycle} cycles")
                    self.lbl_sweep.setStyleSheet("color:#8b949e;")
                    return
            target = self._sweep_waypoints[self._sweep_idx]
            for s in self.servos:
                s.start_ramp(target, self._sweep_speed)
            self._sweep_phase = "ramp"
            self.lbl_sweep.setText(
                f"cycle {self._sweep_cycle + 1}, stop {self._sweep_idx + 1}/{n}: "
                f"-> {target:.1f} mm")

    # -----------------------------------------------------------------
    # main loop
    # -----------------------------------------------------------------
    def _tick(self):
        now = time.monotonic()
        dt = now - self._t_prev
        self._t_prev = now

        any_estop = False
        for s in self.servos:
            try:
                st = s.service(dt)
            except Exception:
                st = s.get_state()
            if self._jog_dir and st.get("connected") and not st.get("estop"):
                s.jog(self._jog_dir, step_rev=float(self.jog_step.value()))
            if st.get("estop"):
                any_estop = True
            lab = self.ro_labels.get(s.dxl_id)
            if lab is not None:
                lab.setText(
                    f"{NAMES.get(s.dxl_id, s.dxl_id)}: "
                    f"{st.get('delta_L_mm', float('nan')):.2f} mm   "
                    f"{st.get('current_ma', float('nan')):.0f} mA"
                    f"{'  [E-STOP]' if st.get('estop') else ''}")

        if self.lbl_estop is not None:
            self.lbl_estop.setText("E-STOP LATCHED" if any_estop else "")

        self._advance_sweep(now)

    # -----------------------------------------------------------------
    # shutdown
    # -----------------------------------------------------------------
    def _shutdown(self):
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        for s in self.servos:
            try:
                s.disable()
                s.disconnect()
            except Exception:
                pass

    def closeEvent(self, e):
        self._shutdown()
        super().closeEvent(e)


def main():
    p = argparse.ArgumentParser(
        description="Finger servo connect + jog-tension + delta-L sweep dashboard")
    p.add_argument("--servo", choices=["auto", "a", "b", "both"], default="auto",
                   help="which finger servo(s) to bind on CONNECT (default: auto)")
    p.add_argument("--port", default="auto", help="U2D2 serial port (default: auto-scan)")
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--min-delta", type=float, default=0.0, help="sweep low end [mm]")
    p.add_argument("--max-delta", type=float, default=config.GRIPPER_MAX_PULL_MM,
                   help=f"sweep high end [mm] (default {config.GRIPPER_MAX_PULL_MM})")
    p.add_argument("--speed", type=float, default=5.0, help="default ramp speed [mm/s]")
    p.add_argument("--step", type=float, default=5.0,
                   help="default delta-L increment per stop [mm] (default 5.0)")
    p.add_argument("--hold", type=float, default=0.5, help="default stop time at each step [s]")
    p.add_argument("--cycles", type=int, default=0,
                   help="default cycle count (0 = run until STOP SWEEP)")
    p.add_argument("--mock", action="store_true", help="no hardware")
    args = p.parse_args()

    app = QApplication(sys.argv)
    win = SweepDashboard(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
