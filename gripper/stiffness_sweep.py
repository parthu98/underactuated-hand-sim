#!/home/namit/iitgn/mujoco_env/bin/python3
"""
stiffness_sweep.py
==================
Headless **stiffness-ratio sweep** of the two-finger gripper's load-carrying
(pull-out) capacity, for comparison against the hardware rig.

For every cell of a ρ1 × ρ3 grid this script:

    1. builds the SAME validated load-test scene as
       ``gripper/interactive_load_test.py`` (it reuses that file's XML builder +
       ``_apply`` / ``_readout`` so the physics is identical — single source of
       truth),
    2. sets the three joint springs from the cell's ratios,
    3. closes both fingers onto the object with the servo-capped flexor,
    4. ramps the external pull tension T until the object breaks free, and
    5. records **Tmax** — the tension at break-free = the holding capacity.

Convention (matches ``high_fidelity/validation.py`` and the paper):

    k1 = MCP (proximal),  k2 = PIP (middle, REFERENCE),  k3 = DIP (distal)
    ρ1 = k1 / k2          (x-grid is independent of k2 scale)
    ρ3 = k3 / k2
    k2 is pinned to SPRING_2 — the whole theory rests on the *ratios*, so k2 is
    the fixed reference and we sweep ρ1, ρ3 about it.

The grid is log-spaced across the three measured hardware spring ratios
(SPRING_3/k2 … SPRING_1/k2) so S3 / S1 land on the grid edges (and a point near
S2 = 1.0 sits in the middle).

Outputs (in ``gripper/sweep_results/``):
    stiffness_sweep.csv         — one row per cell (ρ1, ρ3, k's, Tmax, status…)
    heatmap_Tmax.png            — ρ1 (y) vs ρ3 (x), cells coloured/annotated Tmax

Run:
    python stiffness_sweep.py              # full grid (SWEEP.n_rho²)
    python stiffness_sweep.py --quick      # 3×3 smoke grid (fast sanity check)

Everything you would want to change for a different sweep is in the ``SWEEP``
namespace below; genuinely general / hardware facts (servo torque, spool radius,
hardware aperture, springs) live in the top-level ``config.py``.
"""
import argparse
import csv
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import config                       # noqa: E402  — single source of truth
# Reuse the validated load-test physics (XML builder + step application). This
# import pulls in Tk/matplotlib-TkAgg at module load, but only defines classes —
# it opens no window until main() is called, so it is safe headless.
import interactive_load_test as ilt  # noqa: E402

# Force a non-interactive backend for OUR plotting (ilt set TkAgg on import).
import matplotlib                   # noqa: E402
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt     # noqa: E402

MM = 1e-3
OUT_DIR = os.path.join(HERE, "sweep_results")

# =====================================================================
#  SWEEP PARAMETERS — edit here for a different sweep.
#  (Hardware/general facts come from config.py and are only referenced.)
# =====================================================================
SWEEP = SimpleNamespace(
    # ---- ρ grid ------------------------------------------------------
    n_rho=10,                 # grid points per axis  → n_rho² cells
    # ρ axis bounds are the measured spring ratios (k2 = SPRING_2 reference):
    #   ρ_low  = SPRING_3 / SPRING_2,  ρ_high = SPRING_1 / SPRING_2
    # (computed below from config so they always track the measured springs).

    # ---- Object (the thing being grasped) ----------------------------
    # Hardware-matched: a vertical Ø80 mm cylinder (the duct-tape-wrapped object),
    # standing on its axis (+Z), as logged by the load-test rig. All numbers come
    # from config so the sim scene tracks the real fixture exactly.
    object_shape="cylinder",  # box | cylinder | sphere
    object_rotated=True,      # 90° about Y → cylinder stands VERTICAL (axis +Z),
                              #   fingers wrap its Ø80 mm cross-section, pull is ⟂.
    object_diameter_mm=config.LOAD_TEST_OBJECT_DIAMETER_MM,  # 80 mm → radius 40 mm
    object_length_mm=config.LOAD_TEST_OBJECT_HEIGHT_MM,      # 80 mm full height (half = 40 mm)
    object_mass_kg=config.LOAD_TEST_OBJECT_MASS_KG,

    # ---- Scene geometry ----------------------------------------------
    # aperture_mm here is the CENTRE-TO-CENTRE base-link separation (from config),
    # i.e. inner aperture (77 mm) + one base-link width. The Ø80 mm object is wider
    # than the 77 mm inner gap, so the fingers must be pre-splayed at rest
    # (LOAD_TEST_INIT_SPLAY_DEG, applied in measure_cell) — otherwise straight
    # fingers start embedded and the cell blows up.
    aperture_mm=config.LOAD_TEST_SEPARATION * 1000.0,  # centre-to-centre sep (mm); inner aperture = 77 mm
    # Mount height raised so the rotated (vertical) cylinder clears the floor:
    # object bottom = mount_height - object_half_length must be > 0.
    mount_height_m=config.LOAD_TEST_MOUNT_HEIGHT,   # 0.060 m — finger-base/object-centre height
    object_depth_x_m=config.LOAD_TEST_OBJECT_DEPTH_X,  # X anchor of the grasp (~0.088 m)
    gravity=True,             # load test runs upright, with gravity (like hw)

    # ---- Actuation / servo cap ---------------------------------------
    # Close command: large enough to fully envelop AND saturate the flexor; the
    # servo force cap (config.LOAD_TEST_MAX_TENDON_FORCE) then limits grip force,
    # so the holding capacity is set by the servo torque, not by ΔL.
    close_delta_l_mm=config.LOAD_TEST_CLOSE_DELTA_L_MM,   # 15 mm command (force-capped)
    max_flexor_force_n=config.LOAD_TEST_MAX_TENDON_FORCE,  # per-finger servo cap

    # ---- Tension ramp / break-free detection -------------------------
    tension_max_n=config.LOAD_TEST_MAX_TENSION,    # 300 N ramp ceiling (Tmax cap)
    close_time_s=1.2,         # settle time for the closing grasp
    ramp_time_s=3.5,          # 0 → tension_max linear ramp duration
    slip_threshold_mm=8.0,    # net object slide that counts as "broken free"

    # ---- Heatmap rendering -------------------------------------------
    # The raw grid is coarse (n_rho²) and the break-free detection adds a little
    # speckle, so render a SMOOTH field: denoise lightly, then upsample in log-ρ
    # space (cell index = log-ρ here) with cubic interpolation + filled contours.
    smooth_sigma=0.7,         # Gaussian denoise in CELL units (0 = off)
    smooth_upsample=24,       # cubic-zoom factor for the displayed field
    heatmap_contours=12,      # number of overlaid contour lines (0 = none)
    heatmap_annotate=False,   # overlay the raw per-cell numbers (clutters smooth)

    # ---- Output ------------------------------------------------------
    out_dir=OUT_DIR,
    csv_name="stiffness_sweep.csv",
    heatmap_name="heatmap_Tmax.png",
)

# ρ axis anchors from the measured springs (k2 = SPRING_2 reference).
K2_REF = config.SPRING_2
RHO_LOW = config.SPRING_3 / K2_REF    # ρ at the soft spring  (S3)
RHO_MID = config.SPRING_2 / K2_REF    # = 1.0                  (S2)
RHO_HIGH = config.SPRING_1 / K2_REF   # ρ at the stiff spring (S1)


# =====================================================================
#  Grid + labels
# =====================================================================
def build_rho_grid(n):
    """n log-spaced ρ values across [RHO_LOW, RHO_HIGH] (S3 … S1 on the edges)."""
    return 10 ** np.linspace(np.log10(RHO_LOW), np.log10(RHO_HIGH), n)


def rho_tick_label(rho):
    """Format a ρ value, marking the spring anchors S1/S2/S3."""
    for anchor, tag in ((RHO_LOW, "S3"), (RHO_MID, "S2"), (RHO_HIGH, "S1")):
        if abs(rho - anchor) < 1e-2:
            return f"{rho:.2f}\n({tag})"
    return f"{rho:.2f}"


# =====================================================================
#  One cell: close on the object, ramp the pull, return Tmax
# =====================================================================
def _snap(tension, k_mcp, k_pip, k_dip):
    """A panel-state snapshot for ilt._apply (sweep geometry + this cell's k's)."""
    return SimpleNamespace(
        reset=False, link=True,
        dL={"a": SWEEP.close_delta_l_mm * MM, "b": SWEEP.close_delta_l_mm * MM},
        stiffness={"mcp": k_mcp, "pip": k_pip, "dip": k_dip},
        aperture=SWEEP.aperture_mm * MM,
        obj_enabled=True, obj_shape=SWEEP.object_shape,
        obj_rotated=SWEEP.object_rotated,
        obj_size_mm=SWEEP.object_diameter_mm / 2.0,
        obj_len_mm=SWEEP.object_length_mm / 2.0,
        obj_depth_x=SWEEP.object_depth_x_m,
        gravity=SWEEP.gravity,
        tension=tension, max_force=SWEEP.max_flexor_force_n)


def measure_cell(model, data, ids, k_mcp, k_pip, k_dip):
    """Close the grip, ramp T to break-free, return (Tmax_N, status, grip_N, slip_mm)."""
    dt = model.opt.timestep
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Pre-splay both MCPs to the config init angle so the straight fingers start
    # OUTSIDE the Ø80 mm object: it is wider than the 77 mm inner aperture, so
    # un-splayed fingers begin embedded and the contact solver blows up.
    for f in "ab":
        data.qpos[ids.qadr[f]["mcp"]] = math.radians(config.LOAD_TEST_INIT_SPLAY_DEG)
    mujoco.mj_forward(model, data)

    cur = {"a": 0.0, "b": 0.0}

    # Phase 1 — close on the object (force-capped flexor).
    for _ in range(int(SWEEP.close_time_s / dt)):
        ilt._apply(model, data, ids, cur, _snap(0.0, k_mcp, k_pip, k_dip))
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        return math.nan, "blewup_close", math.nan, math.nan

    ro = ilt._readout(model, data, ids, _snap(0.0, k_mcp, k_pip, k_dip))
    grip_close = ro["grip"]
    x0 = float(data.qpos[ids.obj_qadr])

    # Phase 2 — ramp the pull tension; record T at break-free.
    n_ramp = int(SWEEP.ramp_time_s / dt)
    slip_m = SWEEP.slip_threshold_mm * MM
    Tmax, status = SWEEP.tension_max_n, "held(censored)"
    for i in range(n_ramp):
        T = SWEEP.tension_max_n * (i / n_ramp)
        ilt._apply(model, data, ids, cur, _snap(T, k_mcp, k_pip, k_dip))
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return math.nan, "blewup_pull", grip_close, math.nan
        if float(data.qpos[ids.obj_qadr]) - x0 > slip_m:
            Tmax, status = T, "slip"
            break
    slip_mm = (float(data.qpos[ids.obj_qadr]) - x0) * 1000.0
    return Tmax, status, grip_close, slip_mm


# =====================================================================
#  Heatmap (ρ1 on y, ρ3 on x — matches validation.py convention)
# =====================================================================
def _smooth_field(Tmax):
    """NaN-safe denoise + cubic upsample of the Tmax grid for smooth display.

    The grid is log-spaced in ρ, so the cell index axis IS log-ρ — interpolating
    in index space is therefore the physically correct (geometric) interpolation.
    Returns the upsampled field (NaNs preserved as a mask)."""
    finite = np.isfinite(Tmax)
    filled = np.where(finite, Tmax, np.nanmean(Tmax))

    sigma = SWEEP.smooth_sigma
    if sigma:
        # Normalised Gaussian smoothing so masked/edge cells don't bias the blur.
        w = gaussian_filter(finite.astype(float), sigma, mode="nearest")
        filled = gaussian_filter(filled * finite, sigma, mode="nearest")
        filled = np.divide(filled, w, out=np.full_like(filled, np.nanmean(Tmax)),
                           where=w > 1e-6)

    z = max(1, int(SWEEP.smooth_upsample))
    hi = zoom(filled, z, order=3, mode="nearest")
    # Carry the missing-data mask up to the fine grid (nearest, so blocks stay).
    mask_hi = zoom((~finite).astype(float), z, order=0, mode="nearest") > 0.5
    return np.ma.array(hi, mask=mask_hi)


def plot_heatmap(Tmax, rho_grid, path):
    n = len(rho_grid)
    logr = np.log10(rho_grid)
    ext = [logr[0], logr[-1], logr[0], logr[-1]]   # x = ρ3, y = ρ1 (log space)

    fig, ax = plt.subplots(figsize=(7.6, 6.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")

    field = _smooth_field(Tmax)
    finite = np.ma.masked_invalid(Tmax).compressed()
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (None, None)

    im = ax.imshow(field, origin="lower", cmap=cmap, aspect="auto",
                   extent=ext, interpolation="bicubic", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$T_{\max}$  holding capacity  (N)")

    if SWEEP.heatmap_contours and finite.size:
        xs = np.linspace(logr[0], logr[-1], field.shape[1])
        ys = np.linspace(logr[0], logr[-1], field.shape[0])
        levels = np.linspace(vmin, vmax, int(SWEEP.heatmap_contours))
        cs = ax.contour(xs, ys, field, levels=levels, colors="white",
                        linewidths=0.6, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")

    ax.set_xticks(logr)
    ax.set_yticks(logr)
    ax.set_xticklabels([rho_tick_label(r) for r in rho_grid], fontsize=8)
    ax.set_yticklabels([rho_tick_label(r) for r in rho_grid], fontsize=8)
    ax.set_xlabel(r"$\rho_3 = k_3 / k_2$   (DIP / PIP)")
    ax.set_ylabel(r"$\rho_1 = k_1 / k_2$   (MCP / PIP)")
    ax.set_title(
        f"Gripper holding capacity vs stiffness ratios\n"
        f"object Ø{SWEEP.object_diameter_mm:.0f} mm · inner aperture "
        f"{config.LOAD_TEST_APERTURE_INNER_MM:.0f} mm · servo cap "
        f"{SWEEP.max_flexor_force_n:.0f} N/finger · $k_2$={K2_REF:.3f} N·m/rad",
        fontsize=10)

    if SWEEP.heatmap_annotate:
        thr = 0.5 * (vmin + vmax) if finite.size else 0.0
        for i in range(n):
            for j in range(n):
                v = Tmax[i, j]
                txt = f"{v:.0f}" if np.isfinite(v) else "—"
                ax.text(logr[j], logr[i], txt, ha="center", va="center",
                        fontsize=6,
                        color="white" if (np.isfinite(v) and v < thr) else "black")
    fig.savefig(path, dpi=300)
    plt.close(fig)


# =====================================================================
#  Driver
# =====================================================================
def run_sweep(quick=False):
    os.makedirs(SWEEP.out_dir, exist_ok=True)
    if quick:
        SWEEP.n_rho = 3
    rho_grid = build_rho_grid(SWEEP.n_rho)
    n = SWEEP.n_rho

    # Build the load-test scene ONCE with the sweep geometry; only the joint
    # springs change per cell (set live through ilt._apply).
    xml = ilt.generate_load_test_xml(
        separation=SWEEP.aperture_mm * MM,
        mount_height=SWEEP.mount_height_m,
        object_shape=SWEEP.object_shape,
        object_size_mm=SWEEP.object_diameter_mm / 2.0,
        object_length_mm=SWEEP.object_length_mm / 2.0,
        object_depth_x=SWEEP.object_depth_x_m,
        gravity_on=SWEEP.gravity)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = ilt._make_ids(model)
    ids.Lrest = {f: float(data.ten_length[ids.tendon[f]]) for f in "ab"}

    print(f"[sweep] grid {n}×{n} = {n*n} cells   "
          f"ρ ∈ [{rho_grid[0]:.3f}, {rho_grid[-1]:.3f}]   "
          f"k2(ref)={K2_REF:.3f} N·m/rad")
    print(f"[sweep] object Ø{SWEEP.object_diameter_mm:.0f} mm "
          f"({'rotated/vertical' if SWEEP.object_rotated else 'flat'}) · "
          f"inner aperture {config.LOAD_TEST_APERTURE_INNER_MM:.0f} mm · "
          f"mount {SWEEP.mount_height_m*1000:.0f} mm")
    print(f"[sweep] servo {config.LOAD_TEST_SERVO_STALL_KGFCM:.0f} kgf·cm × "
          f"{config.LOAD_TEST_SERVO_SAFETY_FACTOR:.2f} → "
          f"F_max {SWEEP.max_flexor_force_n:.0f} N/finger · "
          f"close ΔL {SWEEP.close_delta_l_mm:.0f} mm (force-capped, "
          f"ΔL@Fmax≈{config.LOAD_TEST_DELTA_L_AT_FMAX*1000:.1f} mm)")

    Tmax = np.full((n, n), np.nan)        # rows = ρ1 (MCP), cols = ρ3 (DIP)
    rows = []
    t0 = time.time()
    for i, rho1 in enumerate(rho_grid):       # ρ1 = k1/k2  (MCP)
        for j, rho3 in enumerate(rho_grid):   # ρ3 = k3/k2  (DIP)
            k_mcp, k_pip, k_dip = rho1 * K2_REF, K2_REF, rho3 * K2_REF
            T, status, grip, slip = measure_cell(model, data, ids,
                                                 k_mcp, k_pip, k_dip)
            Tmax[i, j] = T
            rows.append({
                "rho1": rho1, "rho3": rho3,
                "k_mcp": k_mcp, "k_pip": k_pip, "k_dip": k_dip,
                "Tmax_N": T, "status": status,
                "grip_close_N": grip, "final_slip_mm": slip})
            print(f"  [{i*n+j+1:3d}/{n*n}] ρ1={rho1:5.2f} ρ3={rho3:5.2f} "
                  f"k=({k_mcp:.3f},{k_pip:.3f},{k_dip:.3f})  "
                  f"Tmax={T:6.1f} N  {status}")

    # ---- CSV ---------------------------------------------------------
    csv_path = os.path.join(SWEEP.out_dir, SWEEP.csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Heatmap -----------------------------------------------------
    png_path = os.path.join(SWEEP.out_dir, SWEEP.heatmap_name)
    plot_heatmap(Tmax, rho_grid, png_path)

    n_blew = sum(1 for r in rows if str(r["status"]).startswith("blewup"))
    print(f"\n[sweep] done in {time.time()-t0:.1f}s   "
          f"({n_blew} cell(s) non-finite)" if n_blew else
          f"\n[sweep] done in {time.time()-t0:.1f}s")
    print(f"[sweep] CSV     → {csv_path}")
    print(f"[sweep] heatmap → {png_path}")
    if n_blew:
        print("[sweep] WARNING: some cells blew up at rest — the open fingers are "
              "likely embedded in the object (aperture too tight for the radius). "
              "Increase SWEEP.aperture_mm slightly or reduce object_diameter_mm.")
    return csv_path, png_path


def replot_from_csv():
    """Re-render the heatmap from an existing CSV (no re-simulation)."""
    csv_path = os.path.join(SWEEP.out_dir, SWEEP.csv_name)
    rho_set, cells = set(), {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            r1, r3 = float(row["rho1"]), float(row["rho3"])
            rho_set.add(round(r1, 6))
            cells[(round(r1, 6), round(r3, 6))] = float(row["Tmax_N"])
    rho_grid = np.array(sorted(rho_set))
    n = len(rho_grid)
    keys = [round(r, 6) for r in rho_grid]
    Tmax = np.array([[cells.get((keys[i], keys[j]), np.nan)
                      for j in range(n)] for i in range(n)])
    png_path = os.path.join(SWEEP.out_dir, SWEEP.heatmap_name)
    plot_heatmap(Tmax, rho_grid, png_path)
    print(f"[sweep] re-rendered {n}×{n} heatmap → {png_path}")


def run_combos():
    """Run the simulated load test for the 3 hardware-matched stiffness combos.

    Builds the hardware-matched scene once, then for each combo in
    config.LOAD_TEST_STIFFNESS_COMBOS closes the grip and ramps the pull to the
    break-free tension (Tmax holding capacity), via the same measure_cell used by
    the full sweep. Writes a small CSV + a Tmax-per-combo bar chart.
    """
    os.makedirs(SWEEP.out_dir, exist_ok=True)
    xml = ilt.generate_load_test_xml(
        separation=SWEEP.aperture_mm * MM,
        mount_height=SWEEP.mount_height_m,
        object_shape=SWEEP.object_shape,
        object_size_mm=SWEEP.object_diameter_mm / 2.0,
        object_length_mm=SWEEP.object_length_mm / 2.0,
        object_depth_x=SWEEP.object_depth_x_m,
        gravity_on=SWEEP.gravity)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = ilt._make_ids(model)
    ids.Lrest = {f: float(data.ten_length[ids.tendon[f]]) for f in "ab"}

    print(f"[combos] object Ø{SWEEP.object_diameter_mm:.0f} mm · inner aperture "
          f"{config.LOAD_TEST_APERTURE_INNER_MM:.0f} mm · close ΔL "
          f"{SWEEP.close_delta_l_mm:.0f} mm · servo cap "
          f"{SWEEP.max_flexor_force_n:.0f} N/finger")

    rows = []
    for name in config.LOAD_TEST_STIFFNESS_COMBOS:
        k_mcp, k_pip, k_dip = config.LOAD_TEST_STIFFNESS_CONFIGS[name]
        T, status, grip, slip = measure_cell(model, data, ids, k_mcp, k_pip, k_dip)
        rows.append({"combo": name, "k_mcp": k_mcp, "k_pip": k_pip, "k_dip": k_dip,
                     "Tmax_N": T, "status": status, "grip_close_N": grip,
                     "final_slip_mm": slip})
        print(f"  {name:32s} k=({k_mcp:.3f},{k_pip:.3f},{k_dip:.3f})  "
              f"Tmax={T:6.1f} N  {status}")

    csv_path = os.path.join(SWEEP.out_dir, "load_test_combos.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    png_path = os.path.join(SWEEP.out_dir, "load_test_combos_Tmax.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    labels = [r["combo"] for r in rows]
    tvals = [r["Tmax_N"] for r in rows]
    bars = ax.bar(range(len(rows)), tvals, color="#3b78b8")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"$T_{\max}$  holding capacity  (N)")
    ax.set_title(f"Load-test holding capacity — 3 stiffness combos\n"
                 f"object Ø{SWEEP.object_diameter_mm:.0f} mm · inner aperture "
                 f"{config.LOAD_TEST_APERTURE_INNER_MM:.0f} mm · close ΔL "
                 f"{SWEEP.close_delta_l_mm:.0f} mm", fontsize=10)
    for b, r in zip(bars, rows):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{r['Tmax_N']:.0f}\n{r['status']}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"[combos] CSV     → {csv_path}")
    print(f"[combos] bar     → {png_path}")
    return csv_path, png_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gripper stiffness-ratio holding-capacity sweep.")
    ap.add_argument("--quick", action="store_true", help="3×3 smoke grid (fast).")
    ap.add_argument("--replot", action="store_true",
                    help="re-render the heatmap from the existing CSV (no sims).")
    ap.add_argument("--combos", action="store_true",
                    help="run the 3 hardware-matched stiffness combos (CSV + Tmax bar chart).")
    args = ap.parse_args()
    if args.combos:
        run_combos()
    elif args.replot:
        replot_from_csv()
    else:
        run_sweep(quick=args.quick)
