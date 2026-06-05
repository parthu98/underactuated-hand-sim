#!/usr/bin/env python3
"""Generate values for Tables II, III, and IV of the RAL 2026 paper.

Table II:  Absolute joint angles θ1,θ2,θ3 (ana vs sim) at ΔL = 20 mm
Table III: Dimensionless morphology metrics M12, M32 (ana vs sim) at ΔL = 20 mm
Table IV:  Morphology metric error (%) vs ΔL for ΔL = 10, 15, 20, 25 mm

This script reuses existing results where available (metric_table.csv at ΔL=10mm)
and runs new simulations only for the missing ΔL values.
"""
import csv
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
import finger_model
from analytical_model import (
    analytical_angles_deg,
    extract_kinematics_from_model,
    morphology_metrics,
)

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "finger.xml")
OUT_DIR = os.path.join(HERE, "validation_results")

# Hardware springs
SPRING_1 = config.SPRING_1
SPRING_2 = config.SPRING_2
SPRING_3 = config.SPRING_3
K2_BASE = SPRING_2

EQUIL_MAX_TIME = config.EQUIL_MAX_TIME
VEL_TOL = config.VEL_TOL
SATURATION_TOL = config.SATURATION_TOL

# Three named regimes matching the paper
REGIMES = [
    ("Proximal-dominant", SPRING_3, SPRING_2, SPRING_2),
    ("Uniform",           SPRING_2, SPRING_2, SPRING_2),
    ("Distal-dominant",   SPRING_2, SPRING_2, SPRING_3),
]

# ΔL values for Table IV
DELTA_L_VALUES_MM = [10, 15, 20, 25]
DELTA_L_VALUES_M = [v / 1000.0 for v in DELTA_L_VALUES_MM]


def pct_err(sim, ana):
    """Absolute percentage error: |sim - ana|/ana * 100."""
    if not (np.isfinite(sim) and np.isfinite(ana)) or abs(ana) < 1e-9:
        return np.nan
    return abs(sim - ana) / ana * 100.0


def _load_model():
    """Load the physics-faithful finger model."""
    if not os.path.exists(XML_PATH):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interactive_viewer", os.path.join(HERE, "interactive_viewer.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._build_xml()
    return finger_model.load_fidelity_model(XML_PATH)


def _mujoco_equilibrium(base_model, k_vec, delta_L):
    """Run MuJoCo to equilibrium for given stiffness and ΔL."""
    import mujoco

    model = finger_model.load_fidelity_model(XML_PATH)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("mcp", "pip", "dip")]
    for idx, jid in enumerate(jids):
        model.jnt_stiffness[jid] = k_vec[idx]
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    L_rest = data.ten_length[0]
    model.tendon_lengthspring[0] = [L_rest - delta_L, L_rest - delta_L]

    dt = model.opt.timestep
    n_max = int(EQUIL_MAX_TIME / dt)
    POS_WIN = 200
    POS_TOL = np.radians(0.02)
    q_hist = np.zeros((POS_WIN, 3))

    for step in range(n_max):
        mujoco.mj_step(model, data)
        q_now = np.array([data.qpos[jid] for jid in jids])
        q_hist[step % POS_WIN] = q_now
        if step > 200:
            vels = np.array([data.qvel[jid] for jid in jids])
            pos_drift = np.max(np.abs(q_hist.max(axis=0) - q_hist.min(axis=0)))
            if np.linalg.norm(vels) < VEL_TOL or pos_drift < POS_TOL:
                break

    angles = np.array([np.degrees(data.qpos[jid]) for jid in jids])
    lo = np.degrees(np.array([model.jnt_range[jid, 0] for jid in jids]))
    hi = np.degrees(np.array([model.jnt_range[jid, 1] for jid in jids]))
    angles = np.clip(angles, lo, hi)
    return angles


def try_load_existing_10mm():
    """Try to load existing metric_table.csv for ΔL=10mm data."""
    csv_path = os.path.join(OUT_DIR, 'metric_table.csv')
    if not os.path.exists(csv_path):
        return None
    
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            regime = row['regime']
            data[regime] = {
                'theta_ana': np.array([float(row['theta1_ana_deg']),
                                        float(row['theta2_ana_deg']),
                                        float(row['theta3_ana_deg'])]),
                'theta_sim': np.array([float(row['theta1_sim_deg']),
                                        float(row['theta2_sim_deg']),
                                        float(row['theta3_sim_deg'])]),
                'M12_ana': float(row['M12_ana']),
                'M12_sim': float(row['M12_sim']),
                'M32_ana': float(row['M32_ana']),
                'M32_sim': float(row['M32_sim']),
                'M12_err_pct': float(row['M12_err_pct']),
                'M32_err_pct': float(row['M32_err_pct']),
            }
    return data


def main():
    print("=" * 72)
    print("  GENERATING TABLE II, III, IV VALUES")
    print("=" * 72)

    # Extract geometry from the model
    base_model = _load_model()
    r_ext, link_lengths = extract_kinematics_from_model(base_model)
    print(f"  Moment arms: r = [{r_ext[0]*1000:.3f}, {r_ext[1]*1000:.3f}, {r_ext[2]*1000:.3f}] mm")
    print(f"  Link lengths: L = [{link_lengths[0]*1000:.2f}, {link_lengths[1]*1000:.2f}, {link_lengths[2]*1000:.2f}] mm")

    # Check if ΔL=10mm data already exists
    existing_10mm = try_load_existing_10mm()
    if existing_10mm:
        print("\n  ✓ Found existing metric_table.csv — reusing ΔL=10mm data")
    else:
        print("\n  ✗ No existing data — will run all ΔL values fresh")

    # Map regime short names from existing CSV to our labels
    regime_name_map = {
        'proximal': 'Proximal-dominant',
        'uniform': 'Uniform',
        'distal': 'Distal-dominant',
    }

    # ─── Run simulations for all needed ΔL values ───
    # Structure: results[dL_mm][regime_name] = {...}
    results = {}

    for dL_m, dL_mm in zip(DELTA_L_VALUES_M, DELTA_L_VALUES_MM):
        results[dL_mm] = {}

        # Check if we can reuse existing data for 10mm
        if dL_mm == 10 and existing_10mm:
            print(f"\n  [ΔL = {dL_mm} mm] Reusing existing data")
            for regime_label, k1, k2, k3 in REGIMES:
                short = regime_label.split('-')[0].lower() if '-' in regime_label else regime_label.lower()
                if short in existing_10mm:
                    results[dL_mm][regime_label] = existing_10mm[short]
                    d = existing_10mm[short]
                    print(f"    {regime_label}: θ_ana=[{d['theta_ana'][0]:.2f}, {d['theta_ana'][1]:.2f}, {d['theta_ana'][2]:.2f}]°")
                    continue
                # Fallback: run simulation
                print(f"    {regime_label}: not found in CSV, running simulation...")
                k = np.array([k1, k2, k3])
                th_a = analytical_angles_deg(dL_m, r_ext, k)
                th_s = _mujoco_equilibrium(base_model, k, dL_m)
                M12_a, M32_a = morphology_metrics(th_a)
                M12_s, M32_s = morphology_metrics(th_s)
                results[dL_mm][regime_label] = {
                    'theta_ana': th_a, 'theta_sim': th_s,
                    'M12_ana': M12_a, 'M12_sim': M12_s,
                    'M32_ana': M32_a, 'M32_sim': M32_s,
                    'M12_err_pct': pct_err(M12_s, M12_a),
                    'M32_err_pct': pct_err(M32_s, M32_a),
                }
        else:
            print(f"\n  [ΔL = {dL_mm} mm] Running simulations...")
            for regime_label, k1, k2, k3 in REGIMES:
                k = np.array([k1, k2, k3])
                th_a = analytical_angles_deg(dL_m, r_ext, k)
                th_s = _mujoco_equilibrium(base_model, k, dL_m)
                M12_a, M32_a = morphology_metrics(th_a)
                M12_s, M32_s = morphology_metrics(th_s)
                results[dL_mm][regime_label] = {
                    'theta_ana': th_a, 'theta_sim': th_s,
                    'M12_ana': M12_a, 'M12_sim': M12_s,
                    'M32_ana': M32_a, 'M32_sim': M32_s,
                    'M12_err_pct': pct_err(M12_s, M12_a),
                    'M32_err_pct': pct_err(M32_s, M32_a),
                }
                d = results[dL_mm][regime_label]
                print(f"    {regime_label}: θ_ana=[{th_a[0]:.2f}, {th_a[1]:.2f}, {th_a[2]:.2f}]°  "
                      f"θ_sim=[{th_s[0]:.2f}, {th_s[1]:.2f}, {th_s[2]:.2f}]°")

    # ═══════════════════════════════════════════════════════════════════
    # TABLE II — Absolute Joint Angles at ΔL = 20 mm
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  TABLE II: ANALYTICAL MODEL VS. MUJOCO: ABSOLUTE JOINT ANGLES (ΔL = 20 mm)")
    print("=" * 72)
    dL_ref = 20
    header = f"  {'Case':<20} {'θ1_ana':>8} {'θ1_sim':>8} {'θ2_ana':>8} {'θ2_sim':>8} {'θ3_ana':>8} {'θ3_sim':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    max_e_theta = [0.0, 0.0, 0.0]  # max |eθi| for each joint

    for regime_label, _, _, _ in REGIMES:
        d = results[dL_ref][regime_label]
        th_a = d['theta_ana']
        th_s = d['theta_sim']
        print(f"  {regime_label:<20} {th_a[0]:8.2f} {th_s[0]:8.2f} {th_a[1]:8.2f} {th_s[1]:8.2f} {th_a[2]:8.2f} {th_s[2]:8.2f}")
        for i in range(3):
            err = abs(th_s[i] - th_a[i])
            max_e_theta[i] = max(max_e_theta[i], err)

    print(f"\n  Max |eθ1|: {max_e_theta[0]:.2f}°   Max |eθ2|: {max_e_theta[1]:.2f}°   Max |eθ3|: {max_e_theta[2]:.2f}°")

    # ═══════════════════════════════════════════════════════════════════
    # TABLE III — Dimensionless Morphology Metrics at ΔL = 20 mm
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  TABLE III: DIMENSIONLESS MORPHOLOGY METRICS (ΔL = 20 mm)")
    print("=" * 72)
    header3 = f"  {'Case':<20} {'M12_ana':>8} {'M12_sim':>8} {'M32_ana':>8} {'M32_sim':>8}"
    print(header3)
    print("  " + "-" * (len(header3) - 2))

    max_e12_abs = 0.0
    max_e32_abs = 0.0
    for regime_label, _, _, _ in REGIMES:
        d = results[dL_ref][regime_label]
        print(f"  {regime_label:<20} {d['M12_ana']:8.3f} {d['M12_sim']:8.3f} {d['M32_ana']:8.3f} {d['M32_sim']:8.3f}")
        e12 = abs(pct_err(d['M12_sim'], d['M12_ana']))
        e32 = abs(pct_err(d['M32_sim'], d['M32_ana']))
        max_e12_abs = max(max_e12_abs, e12)
        max_e32_abs = max(max_e32_abs, e32)

    print(f"\n  Max |e12|: {max_e12_abs:.2f}%   Max |e32|: {max_e32_abs:.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # TABLE IV — Error (%) vs ΔL
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  TABLE IV: MORPHOLOGY METRIC ERROR (%) VS. TENDON DISPLACEMENT ΔL")
    print("=" * 72)

    # e12 section
    dl_header = "  " + f"{'Case':<20}" + "".join(f"{'ΔL='+str(dl)+'mm':>10}" for dl in DELTA_L_VALUES_MM)
    print("\n  e12 (%):")
    print(dl_header)
    print("  " + "-" * (len(dl_header) - 2))
    
    max_e12_all = 0.0
    for regime_label, _, _, _ in REGIMES:
        row_str = f"  {regime_label:<20}"
        for dL_mm in DELTA_L_VALUES_MM:
            d = results[dL_mm][regime_label]
            e12 = pct_err(d['M12_sim'], d['M12_ana'])
            row_str += f"{e12:10.2f}"
            max_e12_all = max(max_e12_all, abs(e12))
        print(row_str)

    # e32 section
    print(f"\n  e32 (%):")
    print(dl_header)
    print("  " + "-" * (len(dl_header) - 2))

    max_e32_all = 0.0
    for regime_label, _, _, _ in REGIMES:
        row_str = f"  {regime_label:<20}"
        for dL_mm in DELTA_L_VALUES_MM:
            d = results[dL_mm][regime_label]
            e32 = pct_err(d['M32_sim'], d['M32_ana'])
            row_str += f"{e32:10.2f}"
            max_e32_all = max(max_e32_all, abs(e32))
        print(row_str)

    print(f"\n  Max |e12| across all ΔL: {max_e12_all:.2f}%")
    print(f"  Max |e32| across all ΔL: {max_e32_all:.2f}%")

    # ═══════════════════════════════════════════════════════════════════
    # Save all raw results to a CSV for post-processing
    # ═══════════════════════════════════════════════════════════════════
    csv_path = os.path.join(OUT_DIR, 'tables_II_III_IV.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'delta_L_mm', 'regime',
            'theta1_ana_deg', 'theta2_ana_deg', 'theta3_ana_deg',
            'theta1_sim_deg', 'theta2_sim_deg', 'theta3_sim_deg',
            'M12_ana', 'M12_sim', 'M12_err_pct',
            'M32_ana', 'M32_sim', 'M32_err_pct',
        ])
        for dL_mm in DELTA_L_VALUES_MM:
            for regime_label, _, _, _ in REGIMES:
                d = results[dL_mm][regime_label]
                th_a = d['theta_ana']
                th_s = d['theta_sim']
                writer.writerow([
                    dL_mm, regime_label,
                    f"{th_a[0]:.4f}", f"{th_a[1]:.4f}", f"{th_a[2]:.4f}",
                    f"{th_s[0]:.4f}", f"{th_s[1]:.4f}", f"{th_s[2]:.4f}",
                    f"{d['M12_ana']:.6f}", f"{d['M12_sim']:.6f}", f"{d['M12_err_pct']:.4f}",
                    f"{d['M32_ana']:.6f}", f"{d['M32_sim']:.6f}", f"{d['M32_err_pct']:.4f}",
                ])
    print(f"\n  [SAVED] {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
