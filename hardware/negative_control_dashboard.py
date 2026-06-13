#!/usr/bin/env python3
"""Springs-OUT negative-control dashboard for the tendon-driven 3R finger.

This is the main validation dashboard (``dashboard.py``) with ONE behavioural
change for the ρ→0 negative control: every trial is an *independent* sweep.
After a full ΔL pass the rig ramps back to the origin and **waits for the
operator to re-zero** (press ◎ SET ZERO) before the next sweep starts. That
makes each trial a fresh, independently-zeroed measurement, so the trial-to-
trial spread of M₁₂ / M₃₂ is the headline number — exactly what the control
needs (with stiffness removed, posture should stop being repeatable).

What is the SAME as dashboard.py: camera/ArUco, jog/zero, ramp+settle, capture,
CSV logging, plots, analytical overlay. Run identically, e.g.::

    python3 negative_control_dashboard.py                 # real cam + servo
    python3 negative_control_dashboard.py --mock          # no hardware
    python3 negative_control_dashboard.py --delta-max 45  # sweep up to 45 mm

What is DIFFERENT:
  • the ΔL ramp + sweep reach up to ``--delta-cap`` (default 45 mm), set live
    in the dashboard via the "sweep max ΔL" field and the target spinbox;
  • AUTO SWEEP returns to the origin and pauses for a re-zero between trials.

Springs-out protocol: install no springs (or kᵢ→0), set k≈0 in the spring
panel so the analytical overlay reflects the degenerate corner, use ~10 trials,
and sweep to the same max each time. The variance across trials is the result.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# --- repo single-source-of-truth ---------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import predictor  # noqa: E402
from camera import MockCamera, RealSenseAruco  # noqa: E402
from dashboard import Dashboard, _btn, _dspin  # noqa: E402  (reuse everything)
from servo import MockServo, Servo  # noqa: E402
from state_machine import AutoSweep, State  # noqa: E402

from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
)

# ΔL sweep grid step (mm). Points are 5, 10, …, up to the chosen max.
_SWEEP_STEP_MM = 5.0

# Springs-out floor: the energy-minimisation solution θᵢ ∝ rᵢ/kᵢ divides by k,
# so a literal k=0 yields NaN and aborts the capture. The limit is well-defined
# (when all kᵢ are equal — including →0 — k cancels and θᵢ = rᵢΔL/Σrⱼ², the
# degenerate-uniform pose), so we let the UI read a true 0 but floor it to this
# tiny ε wherever the analytical model is evaluated. ε is k-independent for the
# uniform case, so the logged/displayed prediction is unaffected by its value.
_K_FLOOR = 1e-6


class NegativeControlDashboard(Dashboard):
    """Dashboard variant that re-zeros between sweeps and ramps to a high ΔL cap."""

    def __init__(self, camera, servo, *, delta_cap_mm, delta_max_mm, **kw):
        # These must exist BEFORE super().__init__ runs, because _delta_group()
        # (called from _build_ui) reads them.
        self.delta_cap = float(delta_cap_mm)
        self.delta_max = float(min(delta_max_mm, delta_cap_mm))
        self._await_rezero = False        # paused at origin, waiting for SET ZERO
        self._sweep_trial_seen = 0        # last trial index we opened a sweep for
        super().__init__(camera, servo, **kw)

        # Let the manual GO target reach the full cap (base class hard-codes 25).
        self.delta_spin.setMaximum(self.delta_cap)
        self.setWindowTitle(
            "Springs-OUT Negative Control — re-zero between sweeps — IITGN")
        self.lbl_sweep.setText(
            "negative control: AUTO SWEEP re-zeros between trials")

    # ---- springs-out: allow k = 0 in the spring panel --------------------
    def _spring_group(self):
        g = super()._spring_group()
        # This rig is run with no springs installed: let each kᵢ go to 0 and
        # default the panel to the springs-out state.
        for s in (self.k1, self.k2, self.k3):
            s.setMinimum(0.0)
            s.setValue(0.0)
        self.lbl_label.setCurrentText("springs_out")
        btn = _btn("k → 0  (springs out)", "#0d1117", "#d2a8ff", "#d2a8ff")
        btn.clicked.connect(self._zero_springs)
        # _spring_group lays out on a QGridLayout; drop the button on a new row.
        g.layout().addWidget(btn, 4, 0, 1, 4)
        return g

    def _zero_springs(self):
        for s in (self.k1, self.k2, self.k3):
            s.setValue(0.0)

    def _k_vec(self):
        """Spring vector with non-positive (springs-out) entries floored to ε so
        the analytical model stays finite (a literal 0 yields NaN). The spinboxes
        still read a true 0; the CSV logs ε (≈0, prints as 0.000) since the row's
        k comes from this vector, and ρ reads ≈1 (the degenerate-uniform corner)."""
        k = super()._k_vec()
        return np.where(k <= 0.0, _K_FLOOR, k)

    # ---- UI: add a "sweep max ΔL" field to the ΔL CONTROL group ----------
    def _delta_group(self):
        g = super()._delta_group()
        row = QHBoxLayout()
        row.addWidget(QLabel("sweep max ΔL"))
        self.sweep_max_spin = _dspin(
            _SWEEP_STEP_MM, self.delta_cap, self.delta_max, 1.0, " mm")
        row.addWidget(self.sweep_max_spin)
        g.layout().addLayout(row)
        return g

    def _sweep_list(self):
        """ΔL points 5,10,… up to (and including) the chosen sweep max [mm]."""
        mx = float(self.sweep_max_spin.value())
        pts = list(np.arange(_SWEEP_STEP_MM, mx + 1e-9, _SWEEP_STEP_MM))
        if not pts or abs(pts[-1] - mx) > 1e-9:
            pts.append(mx)
        return [round(float(p), 3) for p in pts]

    # ---- AUTO SWEEP: independent, re-zeroed trials -----------------------
    def _toggle_sweep(self):
        if self.auto_active or self._await_rezero:
            self.auto_active = False
            self._await_rezero = False
            self.btn_sweep.setText("AUTO SWEEP")
            self.lbl_sweep.setText("sweep stopped")
            return
        if not self.joints.is_zeroed() or not self.servo.get_state().get("connected"):
            QMessageBox.warning(self, "Auto sweep",
                                "Connect servo and Set Zero first.")
            return
        self.sweep = AutoSweep(self._sweep_list(), n_trials=self.trial_spin.value())
        self._sweep_trial_seen = 0
        self.auto_active = True
        self.btn_sweep.setText("STOP SWEEP")
        self._sweep_next()

    def _sweep_next(self):
        tgt = self.sweep.current_target()
        if tgt is None:
            self.auto_active = False
            self.btn_sweep.setText("AUTO SWEEP")
            self.lbl_sweep.setText("sweep complete")
            return

        # New trial just started (after the first): return to origin and wait
        # for the operator to re-zero before sweeping again. auto_active is
        # cleared so the SETTLED tick does NOT auto-capture the origin.
        if self.sweep.trial_idx > self._sweep_trial_seen:
            self._sweep_trial_seen = self.sweep.trial_idx
            self._await_rezero = True
            self.auto_active = False
            self.servo.start_ramp(0.0, speed_mm_s=float(self.speed_spin.value()))
            self.delta_spin.setValue(0.0)
            self.target_mm = 0.0
            self._set_state(State.RAMP)
            self.lbl_sweep.setText(
                f"↩ trial {self.sweep.trial_idx} done — returning to origin; "
                f"press ◎ SET ZERO to start trial {self.sweep.trial_idx + 1}/"
                f"{self.sweep.n_trials}")
            return

        # Normal point within the current trial.
        self.delta_spin.setValue(tgt)
        self.target_mm = tgt
        self.servo.start_ramp(tgt, speed_mm_s=float(self.speed_spin.value()))
        self.lbl_sweep.setText(self.sweep.progress())
        self._set_state(State.RAMP)

    def _set_zero(self):
        super()._set_zero()
        # A successful re-zero (state -> ZEROED) during a between-trial pause
        # resumes the sweep with the next, independently-zeroed trial.
        if self._await_rezero and self.state == State.ZEROED:
            self._await_rezero = False
            self.auto_active = True
            self._sweep_next()


# =====================================================================
# entry point
# =====================================================================
def _build_devices(args):
    if args.mock or args.mock_camera:
        cam = MockCamera()
    else:
        cam = RealSenseAruco(width=args.width, height=args.height, fps=args.fps,
                             serial=args.rs_serial)
    if args.mock or args.mock_servo:
        servo = MockServo(spool_radius_m=args.spool_radius,
                          soft_delta_l_cap_mm=args.delta_cap)
    else:
        servo = Servo(port=args.port, baud=args.baud, dxl_id=args.id,
                      spool_radius_m=args.spool_radius,
                      soft_delta_l_cap_mm=args.delta_cap)
    return cam, servo


def main():
    p = argparse.ArgumentParser(
        description="Springs-out negative-control dashboard (re-zero per sweep)")
    p.add_argument("--mock", action="store_true", help="no hardware (mock cam+servo)")
    p.add_argument("--mock-camera", action="store_true")
    p.add_argument("--mock-servo", action="store_true")
    p.add_argument("--port", default="auto",
                   help="servo serial port; 'auto' scans ttyUSB*/ttyACM* + bauds")
    p.add_argument("--id", type=int, default=15)
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--rs-serial", default=None,
                   help="pin a specific RealSense by serial (default: any port)")
    p.add_argument("--spool-radius", type=float, default=config.SPOOL_RADIUS,
                   help=f"tendon spool RADIUS [m] (default {config.SPOOL_RADIUS})")
    p.add_argument("--delta-cap", type=float, default=45.0,
                   help="hard soft-cap / max ΔL the ramp + target field allow [mm]")
    p.add_argument("--delta-max", type=float, default=45.0,
                   help="initial AUTO-SWEEP max ΔL [mm] (editable in the UI)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    # geometry (same source of truth as dashboard.py / validation.py).
    geom_note = ""
    link_lengths = None
    try:
        r, link_lengths = predictor.get_geometry()
        geom_note = (f"r (moment arms) from high-fidelity model: "
                     f"[{r[0]*1000:.2f}, {r[1]*1000:.2f}, {r[2]*1000:.2f}] mm")
    except Exception as e:  # noqa: BLE001
        r = np.full(3, config.SHEATH_MOMENT_ARM)
        geom_note = (f"⚠ mujoco/finger.xml unavailable ({type(e).__name__}); "
                     f"using constant r = {config.SHEATH_MOMENT_ARM*1000:.1f} mm.")

    app = QApplication(sys.argv)
    app.setFont(QFont("Consolas", 10))
    cam, servo = _build_devices(args)
    win = NegativeControlDashboard(
        cam, servo, delta_cap_mm=args.delta_cap, delta_max_mm=args.delta_max,
        geom_r=r, geom_note=geom_note, geom_link_lengths=link_lengths)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
