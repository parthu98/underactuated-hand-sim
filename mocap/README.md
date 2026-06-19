# Mocap rig — PhaseSpace finger tracking

PhaseSpace (OWL2) optical motion capture as an alternative joint-angle source for
the tendon-driven 3R finger, replacing the RealSense + ArUco camera. The servo,
analytical predictor, CSV logging, auto-sweep, and plots are reused verbatim from
the `hardware/` rig — only the joint-angle *measurement* changes.

## How it works

- **Markers, not rigid bodies.** Two LED markers per segment on four segments
  (base / proximal / middle / distal = 8 LEDs). Each segment's direction vector
  is `far − near`; a joint angle is the in-plane angle between consecutive
  segment vectors. (A per-segment rigid body is impossible — the OWL SDK needs
  ≥4 markers for a 6-DOF body, and we only need a direction.)
- **Flexion-plane projection.** Markers are not mounted perfectly (pitch/yaw off
  the link axis), so each 3D segment vector is projected onto the flexion plane
  before its angle is taken. Residual constant offset is cancelled by the
  straight-pose **Set Zero**. Because the finger flexes in a **fixed horizontal
  plane** (no abduction), the plane normal is just the lab **vertical axis** —
  set once via `MOCAP_VERTICAL_AXIS` / `MOCAP_VERTICAL_SIGN`, no calibration flex.
  Only the normal affects joint angles (they difference consecutive segment
  angles, which are invariant to the in-plane long-axis reference).
- **Drop-in.** `tracker.py` emits a per-segment in-plane angle `phi` keyed
  `0..3`, so `hardware/joints.py` computes `mcp/pip/dip` exactly as before.

## Files

| file | purpose |
|------|---------|
| `owl.py` | vendored PhaseSpace OWL2 Python SDK (pure stdlib sockets) |
| `mocap_config.py` | mocap-only config: server IP, LED-id→segment map, paths |
| `tracker.py` | `PhaseSpaceTracker` (+ `MockTracker`) + fixed-plane projection |
| `dashboard.py` | PySide6 dashboard (subclasses the hardware one) |
| `calibrate.py` | CLI to CHECK the plane / confirm which axis is vertical |
| `diagnose.py` | live dump of raw markers + segment vectors/phi (mapping/plane check) |
| `results/` | CSV validation logs land here (gitignored) |

## Setup

1. In the PhaseSpace **Master Client**, assign stable ids to the 8 LEDs and note
   which id is the palm-side (`near`) and fingertip-side (`far`) on each segment.
2. Edit `MOCAP_SEGMENT_MARKER_IDS` in `mocap_config.py` to match, ordered
   `[base, prox, mid, dist]` as `(near_id, far_id)`. Set `MOCAP_SERVER_IP`.
3. Set `MOCAP_VERTICAL_AXIS` / `MOCAP_VERTICAL_SIGN` to the lab axis that points
   up. Confirm it with `python mocap/calibrate.py` (lay the finger flat — it
   reports the least-varying axis and flags a mismatch).

## Run

```bash
python mocap/dashboard.py --mock        # no hardware: synthetic mocap + servo
python mocap/calibrate.py --seconds 5   # CHECK the plane / confirm vertical axis
python mocap/dashboard.py               # PhaseSpace + Dynamixel
```

Workflow in the GUI: **Connect Mocap** → **Connect Servo** → straighten and
**Set Zero** → use the manual ΔL controls for testing / base tension, or **Auto
Sweep** for a full validation run. Rows are logged to
`mocap/results/mocap_validation_*.csv`. (No calibration flex — the flexion plane
is fixed and known from `MOCAP_VERTICAL_AXIS`.)

By default the tracker connects as an OWL **slave**, so the Master Client can
stay open at the same time (`--no-slave` to be the primary client).

**Servo auto-detection:** the servo port/baud/id default to auto — *Connect
Servo* scans every COM/tty port across the common baud rates, broadcast-pings,
and binds whatever Dynamixel answers, so you don't need to know the COM port or
the motor id. Override with `--port COM5 --id 15 --baud 57600` if you ever need
to pin them.
