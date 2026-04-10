"""
validate_candidate.py
---------------------
Physics validation for a specific optimizer-output design candidate.

Bypasses Stage 1 kinematic synthesis (which would re-compute r from l and e)
and feeds the exact 10-dimensional geometry directly into the physics engine.
This mirrors exactly what generate.py does after Stage 2, using the same dict
assembly, mass-properties call, and engine evaluation.

Usage
-----
  python scripts/validate_candidate.py
  python scripts/validate_candidate.py --config configs/generate/baseline.yaml

The Rank 1 candidate is hard-coded below. Edit CANDIDATE to test other designs.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from mech390.config import load_config
from mech390.physics import engine, mass_properties as mp
from mech390.physics._utils import get_or_warn

logging.basicConfig(
    level=logging.WARNING,           # suppress routine INFO noise
    format="%(levelname)-8s  %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Rank 1 optimizer candidate (continuous-space values, no manufacturing rounding)
# ---------------------------------------------------------------------------
CANDIDATE = {
    'r':              0.114726,  # crank radius (m)
    'l':              0.330855,  # rod length (m)
    'e':              0.189765,  # eccentricity (m)
    'width_r':        0.005311,  # crank link width (m)
    'thickness_r':    0.004865,  # crank link thickness (m)
    'width_l':        0.004596,  # rod link width (m)
    'thickness_l':    0.001523,  # rod link thickness (m)
    'd_shaft_A':      0.003000,  # motor output shaft diameter (m) — was pin_diameter_A
    'pin_diameter_B': 0.001514,  # crank–rod lug pin (m)
    'pin_diameter_C': 0.001695,  # rod–slider lug pin (m)
}

# Surrogate prediction for reference
_SURROGATE_PASS_PROB   = 0.900
_SURROGATE_SCORE       = 0.7587


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate_candidate",
        description="Physics validation for the Rank 1 optimizer candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", metavar="PATH",
        default=str(PROJECT_ROOT / "configs" / "generate" / "baseline.yaml"),
        help="Path to generation config YAML (source of material, operating, limits).",
    )
    return p


def _kinematic_check(d: dict) -> dict:
    """
    Check kinematic feasibility: for all crank angles, the rod must be able
    to connect pin B to pin C.  This requires l > r + e (offset mechanism).

    Returns dict with 'margin' (positive = feasible) and 'feasible' flag.
    """
    r, l, e = d['r'], d['l'], d['e']
    margin = l - (r + e)   # must be > 0 for a complete rotation
    return {'margin': margin, 'feasible': margin > 0}


def _net_section_check(d: dict, min_net: float) -> dict:
    """Return net-section margin for each of the 4 pin pairs (positive = OK)."""
    pairs = {
        "width_r – pin_A": ('width_r', 'd_shaft_A'),
        "width_r – pin_B": ('width_r', 'pin_diameter_B'),
        "width_l – pin_B": ('width_l', 'pin_diameter_B'),
        "width_l – pin_C": ('width_l', 'pin_diameter_C'),
    }
    return {
        label: d[w] - d[p] - min_net
        for label, (w, p) in pairs.items()
    }


def run(config_path: str) -> None:
    config = load_config(config_path)

    # -------------------------------------------------------------------------
    # Extract constants from config (mirrors generate.py:334-398)
    # -------------------------------------------------------------------------
    _ctx       = 'validate_candidate'
    operating  = config.get('operating', {})
    limits_cfg = config.get('limits', {})
    material   = config.get('material', {})
    sa_cfg     = config.get('stress_analysis', {})

    rpm            = float(get_or_warn(operating, 'RPM',            30,   context=_ctx))
    sweep_step_deg = float(get_or_warn(operating, 'sweep_step_deg', 15.0, context=_ctx))
    omega          = rpm * 2.0 * math.pi / 60.0
    m_block       = float(get_or_warn(operating,  'm_block',       0.0,   context=_ctx))
    mu            = float(get_or_warn(operating,  'mu',            0.0,   context=_ctx))
    g             = float(get_or_warn(operating,  'g',             9.81,  context=_ctx))
    safety_factor = float(get_or_warn(limits_cfg, 'safety_factor', 1.0,   context=_ctx))
    n_buck_min    = float(get_or_warn(limits_cfg, 'n_buck_min',    3.0,   context=_ctx))
    utilization_max  = float(get_or_warn(limits_cfg, 'utilization_max',   1.0, context=_ctx))
    n_static_min     = float(get_or_warn(limits_cfg, 'n_static_min',      1.0, context=_ctx))
    n_fatigue_min    = float(get_or_warn(limits_cfg, 'n_fatigue_min',     1.0, context=_ctx))
    d_miner_max      = float(get_or_warn(limits_cfg, 'D_miner_max',       1.0, context=_ctx))
    total_cycles     = float(get_or_warn(operating, 'TotalCycles', 18720000,   context=_ctx))

    S_y = float(get_or_warn(material, 'S_y', 345e6, context=_ctx))
    sigma_limit = S_y / safety_factor
    tau_limit   = 0.577 * sigma_limit

    delta      = float(get_or_warn(sa_cfg, 'diametral_clearance_m', 1e-4, context=_ctx))
    min_wall   = float(get_or_warn(sa_cfg, 'min_wall_mm', 0.5e-3, context=_ctx))
    min_net    = delta + 2.0 * min_wall

    _mat = {
        'E':    float(get_or_warn(material, 'E',    73.1e9, context=_ctx)),
        'S_ut': float(get_or_warn(material, 'S_ut', 483e6,  context=_ctx)),
        'S_y':  S_y,
        'Sn':   float(get_or_warn(material, 'Sn',   133e6,  context=_ctx)),
    }
    _sa = {
        'delta':           delta,
        'Kt_lug':          float(get_or_warn(sa_cfg, 'Kt_lug',          2.34,   context=_ctx)),
        'Kt_hole_torsion': float(get_or_warn(sa_cfg, 'Kt_hole_torsion', 4.0,    context=_ctx)),
        'n_buck_target':   n_buck_min,
        'basquin_A':       float(get_or_warn(sa_cfg, 'basquin_A', 924e6,  context=_ctx)),
        'basquin_b':       float(get_or_warn(sa_cfg, 'basquin_b', -0.086, context=_ctx)),
        'C_sur':           float(get_or_warn(sa_cfg, 'C_sur',     0.88,   context=_ctx)),
        'C_st':            float(get_or_warn(sa_cfg, 'C_st',      1.0,    context=_ctx)),
        'C_R':             float(get_or_warn(sa_cfg, 'C_R',       0.81,   context=_ctx)),
        'C_f':             float(get_or_warn(sa_cfg, 'C_f',       1.0,    context=_ctx)),
        'C_m':             float(get_or_warn(sa_cfg, 'C_m',       1.0,    context=_ctx)),
        'L_bearing':       float(get_or_warn(sa_cfg, 'L_bearing', 0.010,  context=_ctx)),
        'alpha_r':         0.0,
    }

    n_shaft_min = float(get_or_warn(limits_cfg, 'n_shaft_min', 2.0, context=_ctx))

    # -------------------------------------------------------------------------
    # Assemble design_eval dict (mirrors generate.py:418-465)
    # -------------------------------------------------------------------------
    design_eval = dict(CANDIDATE)
    design_eval['omega']   = omega
    design_eval['m_block'] = m_block
    design_eval['mu']      = mu
    design_eval['g']       = g

    # Mass properties
    mass_props = mp.compute_design_mass_properties(design_eval, config)
    design_eval.update(mass_props)

    design_eval['total_mass'] = (
        design_eval.get('mass_crank',  0.0)
        + design_eval.get('mass_rod',   0.0)
        + design_eval.get('mass_slider', 0.0)
    )

    # Volume envelope (bounding box)
    _slider_cfg = config.get('geometry', {}).get('slider', {})
    _s_h = float(_slider_cfg.get('height', 0.02))
    _s_l = float(_slider_cfg.get('length', 0.02))
    _r, _l, _e = CANDIDATE['r'], CANDIDATE['l'], CANDIDATE['e']
    _tr = CANDIDATE['thickness_r']
    _tl = CANDIDATE['thickness_l']
    # Bounding-box: see instructions.md Section 3.4 for derivation.
    _T = (_tl + _s_h) / 2.0 + max(_tr, (_s_h - _tl) / 2.0)
    _H = _r + max(_r, _e + _s_h / 2.0)
    _L = _r + float(np.sqrt(max((_r + _l)**2 - _e**2, 0.0))) + _s_l / 2.0
    design_eval['volume_envelope'] = _T * _H * _L

    design_eval.update(_mat)
    design_eval.update(_sa)
    design_eval['n_shaft_min']  = n_shaft_min
    design_eval['n_rpm']          = rpm
    design_eval['sweep_step_deg'] = sweep_step_deg
    design_eval['total_cycles']   = total_cycles

    W = 62   # report width (used throughout)

    # -------------------------------------------------------------------------
    # Pre-check: kinematic feasibility (must satisfy before calling engine)
    # -------------------------------------------------------------------------
    kin_check = _kinematic_check(CANDIDATE)
    if not kin_check['feasible']:
        print()
        print("=" * W)
        print("  Physics Validation — Rank 1 Optimizer Candidate")
        print("=" * W)
        print(f"  Surrogate pass_prob : {_SURROGATE_PASS_PROB:.1%}  |  score : {_SURROGATE_SCORE:.3f}")
        print()
        print("  Geometry:")
        print(f"    r={CANDIDATE['r']*1e3:.1f} mm  l={CANDIDATE['l']*1e3:.1f} mm  "
              f"e={CANDIDATE['e']*1e3:.1f} mm")
        print()
        print("  KINEMATIC INFEASIBILITY DETECTED")
        print(f"  Condition required: l > r + e")
        print(f"  Actual:             l={CANDIDATE['l']*1e3:.1f} mm,  "
              f"r+e={( CANDIDATE['r']+CANDIDATE['e'])*1e3:.1f} mm")
        print(f"  Violation margin:   {kin_check['margin']*1e3:+.2f} mm  (must be > 0)")
        print()
        print("  Explanation: The crank-slider mechanism cannot complete a full rotation")
        print("  because the rod (l) is too short to bridge pin B to pin C at some")
        print("  crank angles (θ≈75°, where r*sin(θ)+e exceeds l).")
        print()
        print("  The optimizer found this geometry because it treats r, l, e as")
        print("  independent variables, but the surrogate was trained only on designs")
        print("  where l > r + e was enforced by Stage 1 kinematic synthesis.")
        print("  A kinematic penalty must be added to the optimizer score function.")
        print("=" * W)
        print("  OVERALL VERDICT: FAIL  (kinematically infeasible)")
        print("=" * W)
        print()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Physics evaluation
    # -------------------------------------------------------------------------
    metrics = engine.evaluate_design(design_eval)

    if not metrics.get('valid_physics', False):
        print("\n  RESULT: INVALID PHYSICS (numerical singularity or NaN in engine)")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Evaluate pass/fail checks (mirrors generate.py:_compute_checks)
    # -------------------------------------------------------------------------
    checks: dict = {}

    # Static utilization
    s_max = metrics.get('sigma_max', 0.0)
    t_max = metrics.get('tau_max',   0.0)
    utilization = max(s_max / sigma_limit if sigma_limit > 0 else 0.0,
                      t_max / tau_limit   if tau_limit   > 0 else 0.0)
    checks['check_static'] = utilization <= utilization_max

    # Static FoS per component
    for comp, key in (('rod', 'sigma_rod_peak'), ('crank', 'sigma_crank_peak'), ('pin', 'sigma_pin_peak')):
        peak = float(metrics.get(key, 0.0) or 0.0)
        n = sigma_limit / peak if peak > 0.0 else float('inf')
        checks[f'n_static_{comp}'] = n
        checks[f'check_static_{comp}'] = n >= n_static_min

    # Buckling
    n_buck = metrics.get('n_buck', float('inf'))
    checks['n_buck'] = n_buck
    checks['check_buck'] = n_buck >= n_buck_min if math.isfinite(n_buck) else True

    # Fatigue Goodman
    for comp in ('rod', 'crank', 'pin'):
        n = metrics.get(f'n_{comp}', float('inf'))
        checks[f'n_fatigue_{comp}'] = n
        checks[f'check_fatigue_{comp}'] = n >= n_fatigue_min if math.isfinite(n) else True

    # Miner's damage
    for comp in ('rod', 'crank', 'pin'):
        D = metrics.get(f'D_{comp}')
        checks[f'D_{comp}'] = D
        checks[f'check_miner_{comp}'] = (D < d_miner_max) if D is not None else True

    # Shaft A (Mott 12-24)
    n_shaft = metrics.get('n_shaft', float('inf'))
    checks['n_shaft'] = n_shaft
    checks['check_shaft'] = bool(n_shaft >= n_shaft_min if math.isfinite(n_shaft) else True)

    # Overall
    all_checks = [
        checks['check_static'],
        checks['check_static_rod'], checks['check_static_crank'], checks['check_static_pin'],
        checks['check_buck'],
        checks['check_fatigue_rod'], checks['check_fatigue_crank'], checks['check_fatigue_pin'],
        checks['check_miner_rod'], checks['check_miner_crank'], checks['check_miner_pin'],
        checks['check_shaft'],
    ]
    passed = all(all_checks)

    # Net-section margins
    net_margins = _net_section_check(CANDIDATE, min_net)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    def _fmt(v, unit=""):
        if v is None:           return "  N/A"
        if not math.isfinite(v): return "  inf"
        return f"{v:>8.4f}{unit}"

    def _ok(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    print()
    print("=" * W)
    print("  Physics Validation — Rank 1 Optimizer Candidate")
    print("=" * W)
    print(f"  Surrogate pass_prob : {_SURROGATE_PASS_PROB:.1%}  |  score : {_SURROGATE_SCORE:.3f}")
    print()
    print("  Geometry (as evaluated — no manufacturing rounding)")
    print(f"    r={CANDIDATE['r']*1e3:.1f} mm  l={CANDIDATE['l']*1e3:.1f} mm  "
          f"e={CANDIDATE['e']*1e3:.1f} mm")
    print(f"    width_r={CANDIDATE['width_r']*1e3:.1f} mm  thickness_r={CANDIDATE['thickness_r']*1e3:.1f} mm")
    print(f"    width_l={CANDIDATE['width_l']*1e3:.1f} mm  thickness_l={CANDIDATE['thickness_l']*1e3:.1f} mm")
    print(f"    pin_A={CANDIDATE['d_shaft_A']*1e3:.1f} mm  "
          f"pin_B={CANDIDATE['pin_diameter_B']*1e3:.1f} mm  "
          f"pin_C={CANDIDATE['pin_diameter_C']*1e3:.1f} mm")
    print()
    print(f"  Mass summary")
    print(f"    total_mass     : {design_eval.get('total_mass', 0.0)*1e3:>8.2f} g")
    print(f"    volume_envelope: {design_eval['volume_envelope']*1e6:>8.2f} cm³")
    print(f"    E_rev          : {metrics.get('E_rev', float('nan'))*1e3:>8.2f} mJ")
    print(f"    tau_A_max      : {metrics.get('tau_A_max', float('nan')):>8.2f} N·m")
    print()
    print(f"  {'Check':<36} {'Value':>8}   {'Limit':>8}   Status")
    print("  " + "-" * (W - 2))

    def row(label, val, limit, ok):
        return f"  {label:<36} {_fmt(val):>8}   {_fmt(limit):>8}   {_ok(ok)}"

    print(row("Static utilization",      utilization, utilization_max, checks['check_static']))
    print(row("n_static  rod  (>= min)", checks['n_static_rod'],   n_static_min, checks['check_static_rod']))
    print(row("n_static  crank(>= min)", checks['n_static_crank'], n_static_min, checks['check_static_crank']))
    print(row("n_static  pin  (>= min)", checks['n_static_pin'],   n_static_min, checks['check_static_pin']))
    print(row("n_buckling     (>= min)", checks['n_buck'],         n_buck_min,   checks['check_buck']))
    print(row("n_fatigue rod  (>= min)", checks['n_fatigue_rod'],  n_fatigue_min, checks['check_fatigue_rod']))
    print(row("n_fatigue crank(>= min)", checks['n_fatigue_crank'],n_fatigue_min, checks['check_fatigue_crank']))
    print(row("n_fatigue pin  (>= min)", checks['n_fatigue_pin'],  n_fatigue_min, checks['check_fatigue_pin']))
    print(row("D_miner   rod  (<= max)", checks['D_rod'],          d_miner_max,  checks['check_miner_rod']))
    print(row("D_miner   crank(<= max)", checks['D_crank'],        d_miner_max,  checks['check_miner_crank']))
    print(row("D_miner   pin  (<= max)", checks['D_pin'],          d_miner_max,  checks['check_miner_pin']))
    print(row("n_shaft  A     (>= min)", checks['n_shaft'],        n_shaft_min,  checks['check_shaft']))
    print()
    print(f"  Net-section margins (positive = OK, min = {min_net*1e3:.2f} mm)")
    for label, margin in net_margins.items():
        flag = "OK  " if margin > 0 else "FAIL"
        print(f"    {label:<22}: {margin*1e3:>+7.2f} mm   [{flag}]")
    print()
    print("=" * W)
    verdict = "PASS" if passed else "FAIL"
    print(f"  OVERALL VERDICT: {verdict}")
    print("=" * W)
    print()

    sys.exit(0 if passed else 1)


def main() -> None:
    args = _build_parser().parse_args()
    try:
        run(args.config)
    except Exception as exc:
        logging.getLogger("validate_candidate").exception("Unexpected error: %s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
