# Underactuated Finger — tendon-driven finger simulation & validation

CAD-accurate MuJoCo simulations of an anthropomorphic **3-joint (MCP / PIP / DIP)
tendon-driven underactuated finger**, together with the hardware and motion-capture
rigs that validate the model against a real prototype. The whole project studies one
relationship:

$$\Delta L \;\longrightarrow\; (\theta_{\text{MCP}},\, \theta_{\text{PIP}},\, \theta_{\text{DIP}})$$

A single flexor-tendon displacement $\Delta L$ pulls the finger, and the three joint
angles emerge **passively** from quasi-static equilibrium (return springs vs. tendon).
The high-fidelity finger model is the foundation for the two-finger gripper work.

---

## Where to start (taking this over?)

1. **This file** — the repository map and how to run things.
2. **`AGENTS.md`** — the canonical deep-dive: architecture, the load-bearing technical
   decisions (unified spool radius, angle-dependent moment-arm curve, empirical
   load-cell calibration, mocap plane convention, …), and project conventions. Every
   AI/dev tool reads it; treat it as the source of truth for *why* things are the way
   they are. (The per-tool files `CLAUDE.md` / `CODEX.md` / `GEMINI.md` /
   `.github/copilot-instructions.md` only hold tool mechanics and defer to `AGENTS.md`.)
3. **`config.py` is the single source of truth for every physical & numerical
   parameter** — geometry, joint limits, spring stiffnesses, friction, the validation
   ΔL, solver tolerances. Change a value there and it propagates everywhere:

   ```text
   config.py ─┬─▶ finger_model.py ──▶ high_fidelity/  (viewer, validation suite)
              ├─▶ analytical_model.py  (closed-form morphology law)
              ├─▶ gripper/             (XML builders + stiffness/capstan sweeps)
              └─▶ hardware/ + mocap/   (rigs read the same physical constants)
   ```

4. **`CONTEXT.md`** — a short rolling "what was I doing last" note, refreshed per session.

---

## Repository layout

Folders with their own `README.md` (`gripper/`, `hardware/`, `mocap/`, `legacy/`) carry
the in-folder detail; the tree below is the one-line map.

```text
underactuated_finger/
├── config.py                # SINGLE SOURCE OF TRUTH — all physical/numerical params
├── finger_model.py          # Builds the physics-faithful MuJoCo finger from config;
│                            #   rewrites angle-dependent tendon moment arms each step
├── analytical_model.py      # Closed-form morphology law (energy minimisation)
├── scene.xml                # Base MuJoCo scene (floor/lighting) for generated models
├── requirements.txt / .envrc
│
├── high_fidelity/           # CAD-accurate finger: viewer, validation, analysis
│   ├── interactive_viewer.py    # Regenerate finger.xml from config + viewer w/ ΔL slider
│   ├── static_viewer.py         # Load an existing finger.xml and view (optional ramp)
│   ├── validation.py            # Analytical-vs-MuJoCo stiffness-ratio sweep → plots/CSVs
│   ├── analyze_hw_validation.py # Clean + analyse a hardware-validation CSV (M12/M32, errors)
│   ├── generate_tables.py       # Tables II/III/IV for the RAL 2026 paper
│   ├── chaos_study.py           # Zero-stiffness chaos / non-determinism study
│   ├── meshes/ · params.json    # CAD STL geometry + inertial properties
│   └── validation_results/ · chaos_results/   # Generated plots + CSVs
│
├── hardware/                # Physical validation rig (PySide6 + Dynamixel + RealSense)
│   ├── dashboard.py             # Main validation dashboard (ArUco angles + servo)
│   ├── load_test_dashboard.py   # Pull-out (load-carrying) test dashboard + Futek load cell
│   ├── negative_control_dashboard.py  # Springs-OUT negative-control variant of dashboard.py
│   ├── servo.py                 # Safety-wrapped Dynamixel interface (shared U2D2, e-stop)
│   ├── servo_sweep.py · servo_sweep_dashboard.py  # ΔL pull-cycle sweep (headless / GUI)
│   ├── joints.py                # MCP/PIP/DIP angle extraction (planar + plane-free 3D)
│   ├── camera.py                # RealSense + ArUco in-plane tracking
│   ├── load_cell.py             # Futek LCM300 axial load cell via USB220 serial
│   ├── logger.py · predictor.py · state_machine.py  # CSV log, analytical overlay, FSM
│   └── results/                 # Committed load-test trial CSVs (research data)
│
├── mocap/                   # PhaseSpace (OWL2) optical tracking — alternative angle source
│   ├── dashboard.py             # Subclasses the hardware Dashboard, swaps camera for mocap
│   ├── tracker.py               # PhaseSpaceTracker / MockTracker → per-joint angles
│   ├── calibrate.py             # Flexion-plane check utility
│   ├── diagnose.py              # Live raw-marker / segment-vector dump
│   ├── mocap_config.py          # Mocap-only knobs (physics still from root config.py)
│   ├── owl.py                   # Vendored PhaseSpace OWL SDK
│   └── results/                 # Committed mocap validation CSVs (research data)
│
├── gripper/                 # Two-finger gripper development
│   ├── build_gripper.py         # Builds gripper.xml (plain gripper) + headless self-test
│   ├── interactive_gripper.py   # Tk control panel for the plain gripper
│   ├── interactive_load_test.py # Self-contained pull-out rig: builds load_test.xml + dashboard
│   ├── stiffness_sweep.py       # Headless ρ1×ρ3 stiffness sweep → heatmap/CSV (--combos/--replot)
│   ├── capstan/capstan_load_test.py  # Capstan-spool load-test variant (+ mu-sweep)
│   └── sweep_results/ · capstan/results/   # Generated heatmaps + CSVs
│
├── legacy/                  # Initial low-fidelity model + early research outputs (reference)
│
└── .dual-graph/             # Local dual-graph MCP context store (cross-session memory);
                             #   the regenerated graph + per-session logs are git-ignored
```

Generated model files (`finger.xml`, `gripper.xml`, `load_test.xml`, `_finger_fidelity.xml`)
are rebuilt from `config.py` on every run and are **git-ignored — do not hand-edit them.**

---

## Setup

```bash
python3 -m venv mujoco_env && source mujoco_env/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)        # so scripts can `import config`
```

An `.envrc` is included that activates the env and sets `PYTHONPATH` (use with `direnv`).

---

## How to run

**Simulation**

```bash
python3 high_fidelity/interactive_viewer.py   # visualise the finger, drag the ΔL slider
python3 high_fidelity/validation.py           # MuJoCo vs analytical sweep → plots + CSVs
```

**Analysis**

```bash
python3 high_fidelity/analyze_hw_validation.py  # latest HW-validation CSV: agreement/errors
python3 high_fidelity/generate_tables.py        # paper Tables II / III / IV
python3 high_fidelity/chaos_study.py            # zero-stiffness non-determinism study
```

**Hardware / mocap rigs** — every dashboard takes `--mock` to run with no hardware attached:

```bash
python3 hardware/dashboard.py --mock            # validation rig (RealSense + Dynamixel)
python3 hardware/load_test_dashboard.py --mock  # pull-out load test (servos + load cell)
python3 mocap/dashboard.py --mock               # PhaseSpace optical tracking + servo
python3 mocap/calibrate.py --seconds 5          # check which lab axis is "up" (flexion plane)
```

**Gripper simulations**

```bash
python3 gripper/interactive_gripper.py          # plain two-finger gripper control panel
python3 gripper/interactive_load_test.py        # simulated pull-out load test
python3 gripper/stiffness_sweep.py              # full ρ1×ρ3 sweep → heatmap  (--combos: 3 spring sets)
python3 gripper/capstan/capstan_load_test.py    # capstan-spool load-test variant
```

---

## The two models

1. **Analytical** (`analytical_model.py`) — quasi-static joint angles from minimum
   elastic energy under the inextensible-tendon constraint,
   $\theta_i = (r_i/k_i)\,\Delta L / \sum_j (r_j^2/k_j)$, with cascaded redistribution
   when a joint hits its mechanical stop, and an angle-dependent tendon moment arm.

2. **High-fidelity MuJoCo** (`finger_model.py` + generated `finger.xml`) — the CAD
   geometry with a hardware-faithful flexor: an angle-dependent sheath moment arm, a
   near-inextensible steel-string tendon, and near-rigid joint limits. Moment arms and
   link lengths are read back from the loaded model so both halves share one geometric
   source of truth.

The validation suite sweeps the stiffness ratios $\rho_1=k_1/k_2$, $\rho_3=k_3/k_2$
(anchored on the three measured hardware springs) and compares MuJoCo equilibrium
against the analytical morphology law, reporting the morphology metrics
$M_{12}=\theta_1/\theta_2$ and $M_{32}=\theta_3/\theta_2$ and their error.
