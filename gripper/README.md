# Two-Finger Gripper

A full gripper built from **two** physics-faithful tendon-driven fingers facing
each other, with a live Tk control panel. It reuses the validated single-finger
model and the project's single source of truth, `../config.py`.

```
gripper/
├── build_gripper.py        # generates gripper.xml from config.py (+ headless self-test)
├── interactive_gripper.py  # the control-panel app (run this)
└── gripper.xml             # auto-generated, git-ignored
```

## Run it

```bash
python interactive_gripper.py
```

A **control panel** opens next to the MuJoCo viewer. Everything is live — no
restarts, no recompiles.

## What the panel controls

| Group | Control | Notes |
|-------|---------|-------|
| **Movement mode** | *Simultaneous (link both fingers)* | On → both fingers move together. Off → drive Finger A and Finger B **independently**. |
| **Finger drive** | Finger A / Finger B **pull ΔL** [mm] (slider **or** entry) | Shortens the tendon by ΔL (`tendon_lengthspring = L_rest − ΔL`) — the validated actuation. More ΔL → more flexion / grip. The string is stiff, so creep it. |
| **Joint stiffness** | live **MCP / PIP / DIP** sliders | Override the joint stiffness on the fly (shared by both fingers) to study how the stiffness ratio shapes the grip. **Not saved** — each launch loads `MCP/PIP/DIP_STIFFNESS` from `config.py`. |
| **Aperture** | centre-to-centre gap (mm) | Slides the two finger bases apart/together live (up to 200 mm). |
| **Probe object** | Enabled · Shape · Size · Length · **Depth** | A **static** target the fingers grip. Radius up to **80 mm**. Depth slides it between the fingers: **low = enveloping grasp, high = pinch grasp**. The cylinder lies along X so the fingers wrap its **circular cross-section**. |
| **Scene** | Gravity · Reset · Quit | Gravity defaults **off** (matches the tested model). |
| **Readout** | per-finger angles + tendon tension, **grip force**, and a live plot | Grip force = total contact force on the object — your handle on gripping ability vs. ΔL / stiffness. |

**Hotkeys** (panel focused): `m` link · `1`/`2` pick finger · `↑`/`↓` jog finger ΔL ·
`←`/`→` aperture · `[` / `]` object depth · `g` gravity · `o` object · `r` reset · `q` quit.

## Design notes

* **Same finger physics as the tested high-fidelity model.** Each finger keeps the
  constant sheath moment arm (`coef = -SHEATH_MOMENT_ARM`), hard joint limits
  (`LIMIT_SOLREF/SOLIMP`), `SIM_TIMESTEP`, and the shared joint springs — identical
  morphology to `finger_model.build_fidelity_xml` / `validation.py`.
* **ΔL drive (the validated actuation).** Each flexor is the stiff
  (`TENDON_STIFFNESS`) string whose spring rest length we shorten by ΔL
  (`tendon_lengthspring = L_rest − ΔL`), identical to `validation.py`. Driving a
  near-inextensible tendon by *position* against a *rigid* object is stiff and was
  unstable at a coarse timestep (forces exploded); the **small `GRIPPER_TIMESTEP`
  below tames it** — grip now rises smoothly with ΔL and stays finite. Creep the
  slider: the string is stiff (~100 N per mm past contact).
* **Live joint stiffness.** The MCP/PIP/DIP sliders write `model.jnt_stiffness`
  for both fingers every step — softer MCP curls the base joint first, stiffer MCP
  pushes the curl out to PIP/DIP. Loaded from `config.py` at launch; the slider
  override is runtime-only (not saved).
* **Near-rigid contacts (so the fingers conform, not clip).** This is the recipe
  from the MuJoCo docs + Menagerie grasping threads, all in `config.py`:
  - **Small `GRIPPER_TIMESTEP` (0.00025 s)** — a contact's stiffness is capped at
    ≈ 2× timestep, so a coarse step lets the finger punch *transiently* into the
    object mid-close (the "cutting in"). The smaller step + `GRIPPER_CONTACT_SOLREF
    ≈ 0.0005` makes contacts near-rigid: **peak penetration drops from ~3–14 mm to
    ≤ ~0.35 mm** even at 200+ N. The viewer sub-steps (~33×) to stay real-time.
  - **`GRIPPER_CONTACT_SOLIMP` with `dmin→1`** — rigid from first contact (no "mush").
  - **`cone="elliptic"` + `GRIPPER_IMPRATIO`** — stiff friction so objects don't slip;
    raise impratio (→50–200) for firmer free-object load tests.
  - **`condim=6` + torsional/rolling friction** — round objects don't spin/roll out.
* **Object must fit the gap.** The open finger half-thickness is ≈ 13 mm, so keep
  `aperture/2 − 13 mm ≥ object radius`. Otherwise the fingers start *embedded* in the
  object at rest and contact forces explode. The panel lets you widen the aperture
  live. Grip needs ~150–250 N tension (the joint return springs are stiff) — sweep
  the tension slider and read the grip-force plot.
* **Shared stiffness.** Both fingers read the same `MCP/PIP/DIP_STIFFNESS` from
  `config.py`, so a single edit there changes **both** fingers. Studying how the
  stiffness *ratio* changes the grip is the whole point — watch the grip-force
  readout vs. tension while you change those values and re-run.
* **Static probe object.** The object is welded in space (no joint), so the
  fingers press against an immovable target and nothing gets flung. Only its
  depth (world Z) is adjustable — in for enveloping, out for pinch.
* **Layout.** Fingers point up, separated along Y, palms facing the centre
  (Finger A `Ry(90°)`, Finger B `Rz(180°)·Ry(90°)`). Pulling either flexor curls
  that finger toward the centre.
* **Everything from `config.py`.** Layout and object defaults live in the
  `GRIPPER_*` block; geometry/stiffness/tendon/limits come from the existing
  sections. Edit there and re-run.

## Headless checks (no display)

```bash
python build_gripper.py                 # compile + step + grip the fixed object
python interactive_gripper.py --selftest  # exercise the apply/readout path
```
