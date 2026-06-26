#!/usr/bin/env python3
"""
capstan_load_test.py
====================
Gripper load-test with **capstan (tendon-sheath) friction**.

Physics:  T_out = T_in * exp(-mu * Phi) at each joint.
Implementation: external opposing torques via qfrc_applied, using the ACTUAL
spring tension (not the servo cap).

Run:
    python capstan_load_test.py              # 3 combos at default mu
    python capstan_load_test.py --mu 0.10    # override mu
    python capstan_load_test.py --sweep      # sweep mu, best fit to HW
"""
import argparse
import csv
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(_REPO_ROOT, "gripper"), _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import config
import finger_model
import interactive_load_test as ilt

MM = 1e-3
JOINT_NAMES = config.JOINT_NAMES
OUT_DIR = os.path.join(HERE, "results")

# Capstan parameters — PLA sheath + steel cable
CAPSTAN_MU = getattr(config, "FRICTION_MU_TENDON", 0.15)

# Routing wrap at zero flexion (from CAD geometry)
CAPSTAN_PHI0_DEG = {"mcp": 30.0, "pip": 20.0, "dip": 15.0}


def capstan_wrap(joint_name, theta_rad):
    """Total wrap angle [rad]: routing + |flexion|."""
    return math.radians(CAPSTAN_PHI0_DEG[joint_name]) + abs(theta_rad)


def apply_capstan_friction(model, data, ids, mu):
    """Apply capstan friction torques at finger A joints via qfrc_applied.

    The tension used is the ACTUAL spring tension from the MuJoCo tendon model
    (not the 306 N servo force cap).  This keeps torques physically correct
    and numerically stable.
    """
    t = ids.capstan_tendon
    # Actual spring tension (how MuJoCo computes it)
    actual_T = max(0.0, float(model.tendon_stiffness[t]) * max(
        0.0, float(data.ten_length[t]) - float(model.tendon_lengthspring[t, 0])))

    T_entry = actual_T
    for idx, jn in enumerate(JOINT_NAMES):
        jid = ids.capstan_jnt[jn]
        theta = float(data.qpos[model.jnt_qposadr[jid]])
        phi = capstan_wrap(jn, theta)
        T_exit = T_entry * math.exp(-mu * phi)

        # Moment arm from finger_model's per-step update
        r = abs(float(model.wrap_prm[model.tendon_adr[t] + idx]))

        # Friction torque opposes force transmission
        tau = (T_entry - T_exit) * r
        data.qfrc_applied[model.jnt_dofadr[jid]] -= tau

        T_entry = T_exit


def build_model():
    xml = ilt.generate_load_test_xml()
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = ilt._make_ids(model)
    ids.Lrest = {f: float(data.ten_length[ids.tendon[f]]) for f in "ab"}
    ids.capstan_tendon = ids.tendon["a"]
    ids.capstan_jnt = {n: ids.jnt["a"][n] for n in JOINT_NAMES}
    return model, data, ids


def _snap(dL_m, tension, k_mcp, k_pip, k_dip):
    return SimpleNamespace(
        reset=False, link=True,
        dL={"a": dL_m, "b": dL_m},
        stiffness={"mcp": k_mcp, "pip": k_pip, "dip": k_dip},
        aperture=config.LOAD_TEST_SEPARATION,
        obj_enabled=True, obj_shape="cylinder", obj_rotated=True,
        obj_size_mm=config.LOAD_TEST_OBJECT_DIAMETER_MM / 2.0,
        obj_len_mm=config.LOAD_TEST_OBJECT_HEIGHT_MM / 2.0,
        obj_depth_x=config.LOAD_TEST_OBJECT_DEPTH_X,
        gravity=True,
        tension=tension, max_force=config.LOAD_TEST_MAX_TENDON_FORCE)


def measure_cell(model, data, ids, k_mcp, k_pip, k_dip,
                 mu=CAPSTAN_MU, capstan=True):
    dt = model.opt.timestep
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    for f in "ab":
        data.qpos[ids.qadr[f]["mcp"]] = math.radians(config.LOAD_TEST_INIT_SPLAY_DEG)
    mujoco.mj_forward(model, data)

    cur = {"a": 0.0, "b": 0.0}

    # Phase 1: close
    snap = _snap(config.LOAD_TEST_CLOSE_DELTA_L_MM * MM, 0.0, k_mcp, k_pip, k_dip)
    for _ in range(int(1.2 / dt)):
        ilt._apply(model, data, ids, cur, snap)
        if capstan:
            apply_capstan_friction(model, data, ids, mu)
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)):
        return math.nan, "blewup"

    x0 = float(data.qpos[ids.obj_qadr])

    # Phase 2: ramp pull
    slip_m = 8.0 * MM
    Tmax, status = 300.0, "held"
    for i in range(int(3.5 / dt)):
        T = 300.0 * (i / (3.5 / dt))
        snap = _snap(config.LOAD_TEST_CLOSE_DELTA_L_MM * MM, T, k_mcp, k_pip, k_dip)
        ilt._apply(model, data, ids, cur, snap)
        if capstan:
            apply_capstan_friction(model, data, ids, mu)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return math.nan, "blewup"
        if float(data.qpos[ids.obj_qadr]) - x0 > slip_m:
            Tmax, status = T, "slip"
            break
    return Tmax, status


def run_combos(mu=CAPSTAN_MU):
    model, data, ids = build_model()
    combos = config.LOAD_TEST_STIFFNESS_COMBOS
    configs = config.LOAD_TEST_STIFFNESS_CONFIGS

    print(f"{'='*72}")
    print(f"CAPSTAN FRICTION LOAD TEST — mu = {mu:.3f}")
    print(f"{'='*72}")
    print(f"  PLA sheath + steel cable")
    print(f"  wrap offsets: MCP={CAPSTAN_PHI0_DEG['mcp']:.0f}deg "
          f"PIP={CAPSTAN_PHI0_DEG['pip']:.0f}deg "
          f"DIP={CAPSTAN_PHI0_DEG['dip']:.0f}deg")
    print()

    # Hardware reference
    hw = {"Uniform": 12.8, "stiff→soft": 3.0, "soft→stiff": 6.6}

    rows = []
    for name in combos:
        k_mcp, k_pip, k_dip = configs[name]
        T_no, s_no = measure_cell(model, data, ids, k_mcp, k_pip, k_dip, capstan=False)
        T_cap, s_cap = measure_cell(model, data, ids, k_mcp, k_pip, k_dip, mu=mu, capstan=True)

        hw_val = next(v for k, v in hw.items() if k in name)
        err = abs(T_cap - hw_val)
        ratio = T_cap / T_no if T_no > 0 else math.nan

        print(f"  {name}")
        print(f"    frictionless: {T_no:6.1f} N   capstan: {T_cap:6.1f} N   "
              f"HW: {hw_val:.1f} N   err: {err:.1f} N")

        rows.append({"combo": name, "k_mcp": k_mcp, "k_pip": k_pip, "k_dip": k_dip,
                      "Tmax_frictionless": T_no, "Tmax_capstan": T_cap,
                      "hw_peak": hw_val, "error": err, "mu": mu})
    print()

    # CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"capstan_mu{mu:.2f}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  CSV -> {csv_path}")

    # Chart
    _plot_comparison(rows, mu)
    return rows


def sweep_mu():
    model, data, ids = build_model()
    combos = config.LOAD_TEST_STIFFNESS_COMBOS
    configs = config.LOAD_TEST_STIFFNESS_CONFIGS
    mu_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    hw = {"Uniform": 12.8, "stiff→soft": 3.0, "soft→stiff": 6.6}

    print(f"{'='*72}")
    print(f"MU SWEEP — 3 combos x {len(mu_values)} mu values")
    print(f"{'='*72}")

    all_rows = []
    for mu in mu_values:
        print(f"\n  mu = {mu:.2f}")
        for name in combos:
            k_mcp, k_pip, k_dip = configs[name]
            T_no, _ = measure_cell(model, data, ids, k_mcp, k_pip, k_dip, capstan=False)
            T_cap, _ = measure_cell(model, data, ids, k_mcp, k_pip, k_dip, mu=mu, capstan=True)
            hw_key = next(k for k in hw if k in name)
            hw_val = hw[hw_key]
            err = abs(T_cap - hw_val)
            print(f"    {name:35s}  no_fric={T_no:6.1f}  capstan={T_cap:6.1f}  HW={hw_val:.1f}  err={err:.1f}")
            all_rows.append({"mu": mu, "combo": name, "Tmax_frictionless": T_no,
                             "Tmax_capstan": T_cap, "hw_peak": hw_val, "error_vs_hw": err,
                             "k_mcp": k_mcp, "k_pip": k_pip, "k_dip": k_dip})

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "mu_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    print(f"\n{'='*72}")
    print("BEST FIT")
    print(f"{'='*72}")
    for name in combos:
        subset = [r for r in all_rows if r["combo"] == name]
        best = min(subset, key=lambda r: r["error_vs_hw"])
        short = name.split("(")[0].strip()
        print(f"  {short:25s}  HW={best['hw_peak']:5.1f}  mu={best['mu']:.2f}  "
              f"sim={best['Tmax_capstan']:6.1f}  err={best['error_vs_hw']:.1f}")

    _plot_sweep(all_rows, combos, hw)
    return all_rows


def _plot_comparison(rows, mu):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        x = np.arange(len(rows))
        w = 0.25
        t_no = [r["Tmax_frictionless"] for r in rows]
        t_cap = [r["Tmax_capstan"] for r in rows]
        hw_vals = [r["hw_peak"] for r in rows]

        ax.bar(x - w, t_no, w, label="Frictionless", color="#3b78b8", alpha=0.85)
        ax.bar(x, t_cap, w, label=f"Capstan μ={mu:.2f}", color="#e74c3c", alpha=0.85)
        ax.bar(x + w, hw_vals, w, label="Hardware", color="#2ecc71", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([r["combo"].split("(")[0].strip() for r in rows], fontsize=9)
        ax.set_ylabel("Tmax (N)")
        ax.set_title(f"Capstan vs Frictionless vs Hardware — μ={mu:.2f}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        for i, (tn, tc, hv) in enumerate(zip(t_no, t_cap, hw_vals)):
            ax.text(i - w, tn, f"{tn:.1f}", ha="center", va="bottom", fontsize=7, color="#3b78b8")
            ax.text(i, tc, f"{tc:.1f}", ha="center", va="bottom", fontsize=7, color="#e74c3c")
            ax.text(i + w, hv, f"{hv:.1f}", ha="center", va="bottom", fontsize=7, color="#2ecc71")

        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUT_DIR, f"capstan_mu{mu:.2f}.png"), dpi=200)
        plt.close(fig)
        print(f"  chart -> {OUT_DIR}/capstan_mu{mu:.2f}.png")
    except Exception:
        pass


def _plot_sweep(all_rows, combos, hw):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
        for ax, name in zip(axes, combos):
            sub = [r for r in all_rows if r["combo"] == name]
            mus = [r["mu"] for r in sub]
            t_caps = [r["Tmax_capstan"] for r in sub]
            hw_val = sub[0]["hw_peak"]
            best = min(sub, key=lambda r: r["error_vs_hw"])

            ax.plot(mus, t_caps, "o-", color="#3b78b8", lw=2, label="Capstan")
            ax.axhline(sub[0]["Tmax_frictionless"], color="#3b78b8", ls="--", alpha=0.4, label="Frictionless")
            ax.axhline(hw_val, color="red", ls="--", lw=1.5, label=f"HW ({hw_val:.1f})")
            ax.plot(best["mu"], best["Tmax_capstan"], "r*", ms=15, label=f"Best μ={best['mu']:.2f}")
            ax.set_title(name.split("(")[0].strip(), fontsize=10)
            ax.set_xlabel("μ"); ax.set_ylabel("Tmax (N)")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

        os.makedirs(OUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUT_DIR, "mu_sweep.png"), dpi=200)
        plt.close(fig)
        print(f"  chart -> {OUT_DIR}/mu_sweep.png")
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=CAPSTAN_MU)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        sweep_mu()
    else:
        run_combos(mu=args.mu)
