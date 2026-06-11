#!/usr/bin/env python3
"""Clean + analyse a hardware-validation CSV (one spring-set run).

Pick a dataset by SUBSTRING, exact filename, full path, or 'latest':

    python3 high_fidelity/analyze_hw_validation.py 195012
    python3 high_fidelity/analyze_hw_validation.py latest
    python3 high_fidelity/analyze_hw_validation.py --list

Cleaning (all tunable, see --help):
  * drop ΔL = 0 rows (analytical morphology metric is undefined: ~0/0);
  * require markers_all_visible == True;
  * require a real flexed pose: theta_pip_exp > --min-pip deg (metric
    denominator), which rejects half-actuated / failed captures;
  * robust per-ΔL outlier rejection: drop rows whose M12_exp or M32_exp is
    beyond a modified z-score of --z (MAD-based) from that ΔL group's median;
  * report the ENGAGED regime (ΔL >= --engaged mm) separately as the
    trustworthy set — low-ΔL steps are a proximal-first engagement artifact
    where the joints do not yet move proportionally.

Reports: run metadata, cleaning summary, per-joint angle error (exp − ana),
exp-vs-ana correlation, M12/M32 agreement (all-clean and engaged), sweep
repeatability, and a per-ΔL table. With --recompute-arm it also shows what the
angle-dependent linear moment arm (config.MOMENT_ARM_*) would predict.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

RESULTS_DIR = os.path.join(_HERE, "validation_results")
JOINTS = ("mcp", "pip", "dip")


# --------------------------------------------------------------------------- #
# dataset resolution
# --------------------------------------------------------------------------- #
def list_datasets():
    return sorted(glob.glob(os.path.join(RESULTS_DIR, "hw_validation_*.csv")),
                  key=os.path.getmtime)


def resolve_dataset(selector):
    """Return the absolute path of the dataset named by ``selector``.

    Accepts a full path, an exact filename, 'latest'/None for the newest run,
    or any substring that uniquely matches one CSV in validation_results/.
    """
    files = list_datasets()
    if not files:
        sys.exit(f"No hw_validation_*.csv found in {RESULTS_DIR}")

    if selector in (None, "", "latest"):
        return files[-1]                                  # newest by mtime

    if os.path.isfile(selector):
        return os.path.abspath(selector)

    cand = os.path.join(RESULTS_DIR, selector)
    if os.path.isfile(cand):
        return cand

    matches = [f for f in files if selector in os.path.basename(f)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        names = "\n  ".join(os.path.basename(f) for f in files)
        sys.exit(f"No dataset matches '{selector}'. Available:\n  {names}")
    names = "\n  ".join(os.path.basename(f) for f in matches)
    sys.exit(f"'{selector}' is ambiguous — matches several:\n  {names}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    try:
        return float(row.get(key, ""))
    except (ValueError, TypeError):
        return np.nan


def col(rows, key):
    return np.array([fnum(r, key) for r in rows], float)


def mod_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.zeros_like(x) if mad == 0 else 0.6745 * (x - med) / mad


def fmt(x, n=2):
    return "nan" if (x is None or not np.isfinite(x)) else f"{x:.{n}f}"


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #
def clean_rows(rows, min_pip, zthresh):
    valid = [r for r in rows
             if fnum(r, "delta_L_mm") > 0
             and str(r.get("markers_all_visible")).lower() == "true"
             and fnum(r, "theta_pip_exp") > min_pip
             and np.isfinite(fnum(r, "M12_exp"))
             and np.isfinite(fnum(r, "M32_exp"))]
    clean, n_out = [], 0
    for dl in sorted({fnum(r, "delta_L_mm") for r in valid}):
        g = [r for r in valid if fnum(r, "delta_L_mm") == dl]
        z12 = mod_z(col(g, "M12_exp"))
        z32 = mod_z(col(g, "M32_exp"))
        for r, a, b in zip(g, z12, z32):
            if abs(a) <= zthresh and abs(b) <= zthresh:
                clean.append(r)
            else:
                n_out += 1
    return valid, clean, n_out


# --------------------------------------------------------------------------- #
# report sections
# --------------------------------------------------------------------------- #
def agreement(rows, met):
    exp, ana = col(rows, f"{met}_exp"), col(rows, f"{met}_ana")
    m = np.isfinite(exp) & np.isfinite(ana) & (np.abs(ana) > 1e-9)
    exp, ana = exp[m], ana[m]
    if not len(exp):
        print(f"  {met}: (no comparable rows)")
        return
    e = exp - ana
    mape = 100 * np.mean(np.abs(e) / np.abs(ana))
    const_ana = np.allclose(ana, ana[0])
    ana_str = f"{ana[0]:.3f} (const)" if const_ana else f"{ana.mean():.3f} (mean)"
    print(f"  {met}:  ana={ana_str}  exp={exp.mean():.3f} ± {exp.std(ddof=1):.3f}"
          f"   bias={e.mean():+.3f}  MAE={np.abs(e).mean():.3f}  "
          f"RMSE={np.sqrt(np.mean(e**2)):.3f}")
    print(f"        mean abs error = {mape:.1f}%   ->  agreement ≈ {100 - mape:.1f}%")


def angle_errors(rows):
    print(f"\n-- Per-joint angle error (exp − ana), n={len(rows)} --")
    print(f"   {'joint':5} {'bias':>8} {'MAE':>8} {'RMSE':>8} {'std':>8}  [deg]")
    alle = []
    for j in JOINTS:
        e = col(rows, f"err_{j}")
        e = e[np.isfinite(e)]
        alle.append(e)
        print(f"   {j:5} {fmt(e.mean()):>8} {fmt(np.abs(e).mean()):>8} "
              f"{fmt(np.sqrt(np.mean(e**2))):>8} {fmt(e.std(ddof=1)):>8}")
    ae = np.concatenate(alle)
    print(f"   {'ALL':5} {fmt(ae.mean()):>8} {fmt(np.abs(ae).mean()):>8} "
          f"{fmt(np.sqrt(np.mean(ae**2))):>8} {fmt(ae.std(ddof=1)):>8}")
    ex = np.concatenate([col(rows, f"theta_{j}_exp") for j in JOINTS])
    an = np.concatenate([col(rows, f"theta_{j}_ana") for j in JOINTS])
    m = np.isfinite(ex) & np.isfinite(an)
    if m.sum() > 2 and np.ptp(an[m]) > 0:
        r = np.corrcoef(ex[m], an[m])[0, 1]
        s, b = np.polyfit(an[m], ex[m], 1)
        print(f"   exp-vs-ana:  Pearson r={r:.3f}  R²={r**2:.3f}  "
              f"fit exp={s:.2f}·ana{b:+.2f}")


def repeatability(rows):
    print(f"\n-- Repeatability across repeated sweeps (std of θ_exp at fixed ΔL) --")
    print(f"   {'ΔL':>4} {'n':>3} | {'σθ_mcp':>7} {'σθ_pip':>7} {'σθ_dip':>7} "
          f"| {'σM12':>6} {'σM32':>6}")
    for dl in sorted({fnum(r, "delta_L_mm") for r in rows}):
        g = [r for r in rows if fnum(r, "delta_L_mm") == dl]
        if len(g) < 2:
            continue
        s = [np.nanstd(col(g, f"theta_{j}_exp"), ddof=1) for j in JOINTS]
        sM12 = np.nanstd(col(g, "M12_exp"), ddof=1)
        sM32 = np.nanstd(col(g, "M32_exp"), ddof=1)
        print(f"   {int(dl):>4} {len(g):>3} | {fmt(s[0]):>7} {fmt(s[1]):>7} "
              f"{fmt(s[2]):>7} | {fmt(sM12):>6} {fmt(sM32):>6}")


def per_dl_table(rows):
    print(f"\n-- Cleaned per-ΔL means (exp ± std) --")
    print(f"   {'ΔL':>4} {'n':>3} | {'M12_exp':>14} | {'M32_exp':>14}")
    for dl in sorted({fnum(r, "delta_L_mm") for r in rows}):
        g = [r for r in rows if fnum(r, "delta_L_mm") == dl]
        m12, m32 = col(g, "M12_exp"), col(g, "M32_exp")
        print(f"   {int(dl):>4} {len(g):>3} | {m12.mean():>6.3f} ± "
              f"{m12.std(ddof=1) if len(g) > 1 else 0:<5.3f} | "
              f"{m32.mean():>6.3f} ± {m32.std(ddof=1) if len(g) > 1 else 0:<5.3f}")


def recompute_arm(clean):
    """Show what the angle-dependent linear moment arm would predict, using
    each file's own baseline 0° arm backed out from the logged theta_ana."""
    import config
    import analytical_model as am

    rs = []
    for r in clean:
        dl = fnum(r, "delta_L_mm") / 1000.0
        s = sum(np.radians(fnum(r, f"theta_{j}_ana")) for j in JOINTS)
        if s > 1e-6:
            rs.append(dl / s)
    r0 = float(np.median(rs)) if rs else config.SHEATH_MOMENT_ARM

    def predict(angle_dep):
        am._MA_ANGLE_DEP = angle_dep
        out = {j: [] for j in JOINTS}
        for r in clean:
            dl = fnum(r, "delta_L_mm") / 1000.0
            k = np.array([fnum(r, f"k_{j}") for j in JOINTS])
            th = am.analytical_angles_deg(dl, np.array([r0] * 3), k)
            for i, j in enumerate(JOINTS):
                out[j].append(th[i])
        return {j: np.array(v) for j, v in out.items()}

    exp = {j: col(clean, f"theta_{j}_exp") for j in JOINTS}
    old = {j: col(clean, f"theta_{j}_ana") for j in JOINTS}   # CSV (const arm)
    new = predict(True)
    print(f"\n-- Angle-dependent arm recompute (baseline r0={r0*1e3:.2f} mm, "
          f"linear to {config.MOMENT_ARM_FULL_FLEXION*1e3:.0f} mm) --")
    print(f"   {'joint':5} {'bias const':>11} {'bias linear':>12} "
          f"{'MAE const':>10} {'MAE linear':>11}  [deg]")
    for j in JOINTS:
        bo, bn = np.nanmean(exp[j] - old[j]), np.nanmean(exp[j] - new[j])
        mo, mn = np.nanmean(np.abs(exp[j] - old[j])), np.nanmean(np.abs(exp[j] - new[j]))
        print(f"   {j:5} {bo:>+11.2f} {bn:>+12.2f} {mo:>10.2f} {mn:>11.2f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="latest",
                    help="substring / filename / path / 'latest' (default: latest)")
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--engaged", type=float, default=10.0,
                    help="ΔL (mm) threshold for the trustworthy 'engaged' regime")
    ap.add_argument("--min-pip", type=float, default=1.0,
                    help="reject captures with theta_pip_exp below this (deg)")
    ap.add_argument("--z", type=float, default=3.5,
                    help="modified z-score cutoff for per-ΔL outlier rejection")
    ap.add_argument("--recompute-arm", action="store_true",
                    help="also show the angle-dependent linear-arm prediction")
    args = ap.parse_args()

    if args.list:
        print("Datasets in", RESULTS_DIR, "(oldest → newest):")
        for f in list_datasets():
            print("  ", os.path.basename(f))
        return

    path = resolve_dataset(args.dataset)
    rows = load(path)
    valid, clean, n_out = clean_rows(rows, args.min_pip, args.z)
    engaged = [r for r in clean if fnum(r, "delta_L_mm") >= args.engaged]

    print("=" * 80)
    print(os.path.basename(path))
    print("=" * 80)
    k = tuple(rows[0].get(f"k_{j}") for j in JOINTS)
    vis = np.mean([str(r.get("markers_all_visible")).lower() == "true" for r in rows])
    print(f"rows={len(rows)}  k(mcp,pip,dip)={k} N·m/rad  "
          f"rho1={rows[0].get('rho1')} rho3={rows[0].get('rho3')}  "
          f"markers_visible={vis*100:.0f}%")
    print(f"cleaning: {len(rows)} total -> {len(valid)} valid "
          f"(−{len(rows)-len(valid)} zero-pull/invalid) -> {len(clean)} clean "
          f"(−{n_out} outliers @ z>{args.z})  | engaged ΔL≥{args.engaged:g}mm: "
          f"n={len(engaged)}")

    angle_errors(engaged)

    print(f"\n== M12 / M32 agreement — ALL cleaned rows (n={len(clean)}) ==")
    agreement(clean, "M12")
    agreement(clean, "M32")
    print(f"\n== M12 / M32 agreement — ENGAGED ΔL≥{args.engaged:g}mm "
          f"(n={len(engaged)}) — trustworthy ==")
    agreement(engaged, "M12")
    agreement(engaged, "M32")

    repeatability(rows)
    per_dl_table(clean)

    if args.recompute_arm:
        try:
            recompute_arm(engaged)
        except Exception as exc:                       # pragma: no cover
            print(f"\n(recompute-arm skipped: {exc})")
    print()


if __name__ == "__main__":
    main()
