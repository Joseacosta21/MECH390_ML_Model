"""
summarize_results.py
--------------------
Read the top-N optimizer candidates from data/results/candidates.json,
run full physics validation on each, and print a manufacturing-ready
terminal report.

The report answers the key question: "Is Rank 1 suitable for manufacture?"
For each candidate it shows:
  - Exact mechanism dimensions (r, l, e, widths, thicknesses, pin diameters)
  - Surrogate predictions (pass_prob, predicted mass, torque, etc.)
  - Physics engine results (ROM, QRR, static FoS, fatigue FoS, buckling FoS)
  - Pass / fail verdict per check, plus overall PASS / FAIL

Project specs (encoded in baseline.yaml):
  ROM target: 250 mm ±0.5 mm  |  QRR: 1.5–2.5  |  Al 2024-T3
  Load: 500 g at slider  |  RPM: 30  |  n_static ≥ 1  |  n_buck ≥ 3

Usage
-----
  python scripts/summarize_results.py
  python scripts/summarize_results.py --candidates data/results/candidates.json \\
                                      --config     configs/generate/baseline.yaml
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from mech390.config import load_config
from mech390.physics import engine, kinematics as _kinematics, mass_properties as mp
from mech390.physics._utils import get_or_warn

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("summarize_results")

# Report column width
W = 72


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="summarize_results",
        description=(
            "Physics-validate the top optimizer candidates and print a "
            "manufacturing-ready terminal report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--candidates", "-c",
        default=str(PROJECT_ROOT / "data" / "results" / "candidates.json"),
        metavar="PATH",
        help="JSON file written by optimize_design.py (top-N candidate dicts).",
    )
    p.add_argument(
        "--config", "-g",
        default=str(PROJECT_ROOT / "configs" / "generate" / "baseline.yaml"),
        metavar="PATH",
        help="Generation config YAML (source of material, operating, limits).",
    )
    p.add_argument(
        "--top", "-n",
        type=int,
        default=3,
        metavar="N",
        help="Number of ranked candidates to validate and report.",
    )
    return p


# ---------------------------------------------------------------------------
# Physics validation helpers (shared logic with validate_candidate.py)
# ---------------------------------------------------------------------------

def _load_constants(config: dict) -> dict:
    """Extract all physics constants from config into a flat dict."""
    _ctx       = "summarize_results"
    operating  = config.get("operating", {})
    limits_cfg = config.get("limits", {})
    material   = config.get("material", {})
    sa_cfg     = config.get("stress_analysis", {})

    rpm            = float(get_or_warn(operating,  "RPM",            30,       context=_ctx))
    sweep_step_deg = float(get_or_warn(operating,  "sweep_step_deg", 15.0,     context=_ctx))
    omega          = rpm * 2.0 * math.pi / 60.0
    m_block        = float(get_or_warn(operating,  "m_block",        0.0,      context=_ctx))
    mu             = float(get_or_warn(operating,  "mu",             0.0,      context=_ctx))
    g              = float(get_or_warn(operating,  "g",              9.81,     context=_ctx))
    safety_factor  = float(get_or_warn(limits_cfg, "safety_factor",  1.0,      context=_ctx))
    n_buck_min     = float(get_or_warn(limits_cfg, "n_buck_min",     3.0,      context=_ctx))
    utilization_max   = float(get_or_warn(limits_cfg, "utilization_max",  1.0, context=_ctx))
    n_static_min      = float(get_or_warn(limits_cfg, "n_static_min",     1.0, context=_ctx))
    n_fatigue_min     = float(get_or_warn(limits_cfg, "n_fatigue_min",    1.0, context=_ctx))
    d_miner_max       = float(get_or_warn(limits_cfg, "D_miner_max",      1.0, context=_ctx))
    total_cycles      = float(get_or_warn(operating, "TotalCycles", 18720000,  context=_ctx))
    n_shaft_min       = float(get_or_warn(limits_cfg, "n_shaft_min",  2.0,     context=_ctx))

    S_y         = float(get_or_warn(material, "S_y",   345e6, context=_ctx))
    sigma_limit = S_y / safety_factor
    tau_limit   = 0.577 * sigma_limit

    delta      = float(get_or_warn(sa_cfg, "diametral_clearance_m", 1e-4,  context=_ctx))
    min_wall   = float(get_or_warn(sa_cfg, "min_wall_mm",           0.5e-3, context=_ctx))

    return dict(
        rpm=rpm, omega=omega, sweep_step_deg=sweep_step_deg,
        m_block=m_block, mu=mu, g=g,
        safety_factor=safety_factor, n_buck_min=n_buck_min,
        utilization_max=utilization_max, n_static_min=n_static_min,
        n_fatigue_min=n_fatigue_min, d_miner_max=d_miner_max,
        total_cycles=total_cycles, n_shaft_min=n_shaft_min,
        S_y=S_y, sigma_limit=sigma_limit, tau_limit=tau_limit,
        delta=delta, min_wall=min_wall, min_net=delta + 2.0 * min_wall,
        material=dict(
            E=float(get_or_warn(material, "E",    73.1e9, context=_ctx)),
            S_ut=float(get_or_warn(material, "S_ut", 483e6,  context=_ctx)),
            S_y=S_y,
            Sn=float(get_or_warn(material, "Sn",   133e6,  context=_ctx)),
        ),
        sa=dict(
            delta=delta,
            Kt_lug=float(get_or_warn(sa_cfg, "Kt_lug",          2.34,   context=_ctx)),
            Kt_hole_torsion=float(get_or_warn(sa_cfg, "Kt_hole_torsion", 4.0,    context=_ctx)),
            n_buck_target=n_buck_min,
            basquin_A=float(get_or_warn(sa_cfg, "basquin_A", 924e6,  context=_ctx)),
            basquin_b=float(get_or_warn(sa_cfg, "basquin_b", -0.086, context=_ctx)),
            C_sur=float(get_or_warn(sa_cfg, "C_sur",  0.88, context=_ctx)),
            C_st=float(get_or_warn(sa_cfg, "C_st",   1.0,  context=_ctx)),
            C_R=float(get_or_warn(sa_cfg, "C_R",    0.81, context=_ctx)),
            C_f=float(get_or_warn(sa_cfg, "C_f",    1.0,  context=_ctx)),
            C_m=float(get_or_warn(sa_cfg, "C_m",    1.0,  context=_ctx)),
            L_bearing=float(get_or_warn(sa_cfg, "L_bearing", 0.010, context=_ctx)),
            alpha_r=0.0,
        ),
    )


def _validate_candidate(
    candidate: Dict[str, Any],
    config: dict,
    constants: dict,
) -> Dict[str, Any]:
    """
    Run full physics validation on a single geometry candidate.

    Args:
        candidate:  Geometry dict (keys: r, l, e, width_r, ..., pin_diameter_C)
                    plus optional surrogate predictions (pred_*, pass_prob, weighted_score).
        config:     Parsed baseline.yaml.
        constants:  Pre-extracted constants from _load_constants().

    Returns:
        Dict with:
            valid_physics: bool
            passed:        bool (all checks pass)
            checks:        dict of individual check results
            metrics:       raw engine output dict
            design_eval:   full design dict (includes mass props, volume_envelope)
    """
    # --- Geometry keys used by the physics engine ---
    GEO_KEYS = (
        "r", "l", "e",
        "width_r", "thickness_r",
        "width_l", "thickness_l",
        "d_shaft_A", "pin_diameter_B", "pin_diameter_C",
    )
    geo = {k: float(candidate[k]) for k in GEO_KEYS}

    # --- Kinematic feasibility pre-check ---
    kin_margin = geo["l"] - (geo["r"] + geo["e"])
    if kin_margin <= 0:
        return {
            "valid_physics": False,
            "kinematic_fail": True,
            "kin_margin": kin_margin,
            "passed": False,
            "checks": {},
            "metrics": {},
            "design_eval": geo,
        }

    # --- Build design_eval dict (mirrors generate.py logic) ---
    c = constants
    design_eval = dict(geo)
    design_eval["omega"]   = c["omega"]
    design_eval["m_block"] = c["m_block"]
    design_eval["mu"]      = c["mu"]
    design_eval["g"]       = c["g"]

    mass_props = mp.compute_design_mass_properties(design_eval, config)
    design_eval.update(mass_props)
    design_eval["total_mass"] = (
        design_eval.get("mass_crank",  0.0)
        + design_eval.get("mass_rod",   0.0)
        + design_eval.get("mass_slider", 0.0)
    )

    # Volume envelope (bounding box) — Section 3.4 of instructions.md
    _slider_cfg = config.get("geometry", {}).get("slider", {})
    _s_h = float(_slider_cfg.get("height", 0.02))
    _s_l = float(_slider_cfg.get("length", 0.02))
    r, l, e = geo["r"], geo["l"], geo["e"]
    tr, tl  = geo["thickness_r"], geo["thickness_l"]
    _T = (tl + _s_h) / 2.0 + max(tr, (_s_h - tl) / 2.0)
    _H = r + max(r, e + _s_h / 2.0)
    _L = r + float(np.sqrt(max((r + l) ** 2 - e ** 2, 0.0))) + _s_l / 2.0
    design_eval["volume_envelope"] = _T * _H * _L

    design_eval.update(c["material"])
    design_eval.update(c["sa"])
    design_eval["n_shaft_min"]    = c["n_shaft_min"]
    design_eval["n_rpm"]          = c["rpm"]
    design_eval["sweep_step_deg"] = c["sweep_step_deg"]
    design_eval["total_cycles"]   = c["total_cycles"]

    # --- Run physics engine ---
    metrics = engine.evaluate_design(design_eval)

    if not metrics.get("valid_physics", False):
        return {
            "valid_physics": False,
            "kinematic_fail": False,
            "passed": False,
            "checks": {},
            "metrics": metrics,
            "design_eval": design_eval,
        }

    # --- Evaluate pass/fail checks ---
    sigma_limit = c["sigma_limit"]
    tau_limit   = c["tau_limit"]

    checks: dict = {}

    s_max = metrics.get("sigma_max", 0.0)
    t_max = metrics.get("tau_max",   0.0)
    utilization = max(
        s_max / sigma_limit if sigma_limit > 0 else 0.0,
        t_max / tau_limit   if tau_limit   > 0 else 0.0,
    )
    checks["utilization"]      = utilization
    checks["check_static"]     = utilization <= c["utilization_max"]

    for comp, key in (
        ("rod",   "sigma_rod_peak"),
        ("crank", "sigma_crank_peak"),
        ("pin",   "sigma_pin_peak"),
    ):
        peak = float(metrics.get(key, 0.0) or 0.0)
        n    = sigma_limit / peak if peak > 0.0 else float("inf")
        checks[f"n_static_{comp}"]      = n
        checks[f"check_static_{comp}"]  = n >= c["n_static_min"]

    n_buck = metrics.get("n_buck", float("inf"))
    checks["n_buck"]       = n_buck
    checks["check_buck"]   = (n_buck >= c["n_buck_min"]) if math.isfinite(n_buck) else True

    for comp in ("rod", "crank", "pin"):
        n = metrics.get(f"n_{comp}", float("inf"))
        checks[f"n_fatigue_{comp}"]      = n
        checks[f"check_fatigue_{comp}"]  = (n >= c["n_fatigue_min"]) if math.isfinite(n) else True

    for comp in ("rod", "crank", "pin"):
        D = metrics.get(f"D_{comp}")
        checks[f"D_{comp}"]           = D
        checks[f"check_miner_{comp}"] = (D < c["d_miner_max"]) if D is not None else True

    n_shaft = metrics.get("n_shaft", float("inf"))
    checks["n_shaft"]      = n_shaft
    checks["check_shaft"]  = bool(
        (n_shaft >= c["n_shaft_min"]) if math.isfinite(n_shaft) else True
    )

    all_checks = [
        checks["check_static"],
        checks["check_static_rod"], checks["check_static_crank"], checks["check_static_pin"],
        checks["check_buck"],
        checks["check_fatigue_rod"], checks["check_fatigue_crank"], checks["check_fatigue_pin"],
        checks["check_miner_rod"], checks["check_miner_crank"], checks["check_miner_pin"],
        checks["check_shaft"],
    ]

    return {
        "valid_physics": True,
        "kinematic_fail": False,
        "passed": all(all_checks),
        "checks": checks,
        "metrics": metrics,
        "design_eval": design_eval,
    }


# ---------------------------------------------------------------------------
# Net-section margin helper
# ---------------------------------------------------------------------------

def _net_section_margins(geo: dict, min_net: float) -> dict:
    pairs = {
        "width_r – pin_A": ("width_r",  "d_shaft_A"),
        "width_r – pin_B": ("width_r",  "pin_diameter_B"),
        "width_l – pin_B": ("width_l",  "pin_diameter_B"),
        "width_l – pin_C": ("width_l",  "pin_diameter_C"),
    }
    return {
        label: float(geo[w]) - float(geo[p]) - min_net
        for label, (w, p) in pairs.items()
    }


# ---------------------------------------------------------------------------
# Formatted printing
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 4, unit: str = "") -> str:
    if v is None:              return "     N/A"
    if not math.isfinite(v):   return "     inf"
    return f"{v:>8.{decimals}f}{unit}"


def _ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _row(label: str, val: Optional[float], limit: Optional[float], ok: bool,
         decimals: int = 4) -> str:
    return (
        f"  {label:<38} {_fmt(val, decimals):>10}   "
        f"{_fmt(limit, decimals):>10}   {_ok(ok)}"
    )


def _print_candidate_report(
    rank:      int,
    candidate: Dict[str, Any],
    result:    Dict[str, Any],
    constants: dict,
    config:    dict,
) -> None:
    checks     = result["checks"]
    metrics    = result["metrics"]
    design     = result["design_eval"]
    c          = constants
    geo        = {k: float(candidate[k])
                  for k in ("r", "l", "e", "width_r", "thickness_r",
                             "width_l", "thickness_l",
                             "d_shaft_A", "pin_diameter_B", "pin_diameter_C")}

    pass_prob      = candidate.get("pass_prob",      float("nan"))
    weighted_score = candidate.get("weighted_score", float("nan"))

    print()
    print("=" * W)
    print(f"  RANK {rank}  —  Surrogate score: {weighted_score:.4f}  "
          f"|  pass_prob: {pass_prob:.1%}")
    print("=" * W)

    # --- Geometry ---
    print()
    print("  Mechanism dimensions")
    print(f"    r (crank radius)     : {geo['r']*1e3:>7.2f} mm")
    print(f"    l (rod length)       : {geo['l']*1e3:>7.2f} mm")
    print(f"    e (eccentricity)     : {geo['e']*1e3:>7.2f} mm")
    print(f"    ROM (l+r-e to l-r+e) : {(geo['l']+geo['r']-geo['e'])*1e3:>7.2f} "
          f"→ {(geo['l']-geo['r']+geo['e'])*1e3:>7.2f} mm  "
          f"[stroke ≈ {2*geo['r']*1e3:.2f} mm]")
    print(f"    l – (r + e)          : {(geo['l'] - geo['r'] - geo['e'])*1e3:>+7.2f} mm  "
          f"[kin. margin, must be > 0]")

    # --- ROM and QRR (analytical, same formula as Stage 1 and physics engine) ---
    kin_m = _kinematics.calculate_metrics(geo['r'], geo['l'], geo['e'])
    operating  = config.get('operating', {})
    rom_target = float(operating.get('ROM', 0.25)) * 1e3          # mm
    rom_tol    = float(operating.get('ROM_tolerance', 0.0005)) * 1e3  # mm
    _qrr_cfg   = operating.get('QRR', {})
    qrr_min_v  = float(_qrr_cfg.get('min', 1.5)) if isinstance(_qrr_cfg, dict) else 1.5
    qrr_max_v  = float(_qrr_cfg.get('max', 2.5)) if isinstance(_qrr_cfg, dict) else 2.5
    if kin_m.get('valid', False):
        rom_mm  = kin_m['ROM'] * 1e3
        qrr_val = kin_m['QRR']
        rom_ok  = abs(rom_mm - rom_target) <= rom_tol
        qrr_ok  = qrr_min_v <= qrr_val <= qrr_max_v
        print(f"    ROM (actual)         : {rom_mm:>7.2f} mm  "
              f"[target {rom_target:.0f}±{rom_tol:.1f} mm]  "
              f"[{'OK  ' if rom_ok else 'FAIL'}]")
        print(f"    QRR (actual)         : {qrr_val:>7.3f}     "
              f"[target {qrr_min_v}–{qrr_max_v}]          "
              f"[{'OK  ' if qrr_ok else 'FAIL'}]")
    else:
        print(f"    ROM / QRR            : KINEMATIC INVALID — cannot compute")
    print()
    print("  Cross-section & pins")
    print(f"    width_r              : {geo['width_r']*1e3:>7.2f} mm  |  "
          f"thickness_r : {geo['thickness_r']*1e3:>6.2f} mm")
    print(f"    width_l              : {geo['width_l']*1e3:>7.2f} mm  |  "
          f"thickness_l : {geo['thickness_l']*1e3:>6.2f} mm")
    print(f"    pin_A (shaft)        : {geo['d_shaft_A']*1e3:>7.2f} mm")
    print(f"    pin_B (crank–rod)    : {geo['pin_diameter_B']*1e3:>7.2f} mm")
    print(f"    pin_C (rod–slider)   : {geo['pin_diameter_C']*1e3:>7.2f} mm")

    # --- Surrogate predictions ---
    print()
    print("  Surrogate predictions")
    pred_mass   = candidate.get("pred_total_mass",      float("nan"))
    pred_vol    = candidate.get("pred_volume_envelope", float("nan"))
    pred_tau    = candidate.get("pred_tau_A_max",       float("nan"))
    pred_erev   = candidate.get("pred_E_rev",           float("nan"))
    pred_nst    = candidate.get("pred_min_n_static",    float("nan"))
    pred_util   = candidate.get("pred_utilization",     float("nan"))
    pred_nbuck  = candidate.get("pred_n_buck",          float("nan"))
    print(f"    total_mass           : {pred_mass*1e3:>7.1f} g" if math.isfinite(pred_mass) else "    total_mass           :     N/A")
    print(f"    volume_envelope      : {pred_vol*1e6:>7.2f} cm³" if math.isfinite(pred_vol) else "    volume_envelope      :     N/A")
    print(f"    tau_A_max            : {pred_tau:>7.4f} N·m" if math.isfinite(pred_tau) else "    tau_A_max            :     N/A")
    print(f"    E_rev                : {pred_erev:>7.4f} J/rev" if math.isfinite(pred_erev) else "    E_rev                :     N/A")
    print(f"    min_n_static (pred)  : {pred_nst:>7.2f}" if math.isfinite(pred_nst) else "    min_n_static (pred)  :     N/A")
    print(f"    utilization (pred)   : {pred_util:>7.3f}" if math.isfinite(pred_util) else "    utilization (pred)   :     N/A")
    print(f"    n_buck (pred)        : {pred_nbuck:>7.2f}" if math.isfinite(pred_nbuck) else "    n_buck (pred)        :     N/A")

    # --- Physics engine results ---
    print()
    if not result["valid_physics"]:
        if result.get("kinematic_fail"):
            print(f"  KINEMATIC INFEASIBILITY — l < r+e by "
                  f"{abs(result['kin_margin'])*1e3:.2f} mm")
        else:
            print("  PHYSICS ENGINE ERROR — numerical singularity or NaN")
        print()
        print("=" * W)
        print("  OVERALL VERDICT: FAIL")
        print("=" * W)
        return

    # Mass / energy
    total_mass = design.get("total_mass", 0.0)
    vol_env    = design.get("volume_envelope", float("nan"))
    E_rev      = metrics.get("E_rev",    float("nan"))
    tau_A      = metrics.get("tau_A_max", float("nan"))

    print("  Physics engine — mass & energy")
    print(f"    total_mass (actual)  : {total_mass*1e3:>7.2f} g")
    print(f"    volume_envelope      : {vol_env*1e6:>7.2f} cm³" if math.isfinite(vol_env) else "    volume_envelope      :     N/A")
    print(f"    tau_A_max            : {tau_A:>7.4f} N·m" if math.isfinite(tau_A) else "    tau_A_max            :     N/A")
    print(f"    E_rev                : {E_rev*1e3:>7.3f} mJ/rev" if math.isfinite(E_rev) else "    E_rev                :     N/A")

    # Check table
    print()
    print(f"  {'Physics check':<38} {'Value':>10}   {'Limit':>10}   Status")
    print("  " + "─" * (W - 2))

    util = checks.get("utilization", float("nan"))
    print(_row("Static utilization (≤ limit)",
               util, c["utilization_max"], checks["check_static"], decimals=3))
    print(_row("n_static  rod   (≥ min)",
               checks["n_static_rod"],   c["n_static_min"], checks["check_static_rod"]))
    print(_row("n_static  crank (≥ min)",
               checks["n_static_crank"], c["n_static_min"], checks["check_static_crank"]))
    print(_row("n_static  pin   (≥ min)",
               checks["n_static_pin"],   c["n_static_min"], checks["check_static_pin"]))
    print(_row("n_buckling      (≥ min)",
               checks["n_buck"], c["n_buck_min"], checks["check_buck"]))
    print(_row("n_fatigue rod   (≥ min)",
               checks["n_fatigue_rod"],  c["n_fatigue_min"], checks["check_fatigue_rod"]))
    print(_row("n_fatigue crank (≥ min)",
               checks["n_fatigue_crank"],c["n_fatigue_min"], checks["check_fatigue_crank"]))
    print(_row("n_fatigue pin   (≥ min)",
               checks["n_fatigue_pin"],  c["n_fatigue_min"], checks["check_fatigue_pin"]))
    print(_row("D_miner   rod   (≤ max)",
               checks["D_rod"],   c["d_miner_max"], checks["check_miner_rod"]))
    print(_row("D_miner   crank (≤ max)",
               checks["D_crank"], c["d_miner_max"], checks["check_miner_crank"]))
    print(_row("D_miner   pin   (≤ max)",
               checks["D_pin"],   c["d_miner_max"], checks["check_miner_pin"]))
    print(_row("n_shaft  A      (≥ min)",
               checks["n_shaft"], c["n_shaft_min"], checks["check_shaft"]))

    # Net-section margins
    margins = _net_section_margins(geo, c["min_net"])
    print()
    print(f"  Net-section margins  (need > 0;  min_net = {c['min_net']*1e3:.3f} mm)")
    for label, margin in margins.items():
        flag = "OK  " if margin > 0 else "FAIL"
        print(f"    {label:<24}: {margin*1e3:>+8.3f} mm   [{flag}]")

    # Verdict
    verdict = "PASS" if result["passed"] else "FAIL"
    print()
    print("=" * W)
    print(f"  OVERALL VERDICT: {verdict}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(candidates_path: str, config_path: str, top_n: int) -> None:
    # Load candidates JSON
    p = Path(candidates_path)
    if not p.exists():
        logger.error("Candidates file not found: %s", p)
        print(f"\n  ERROR: {p} not found.  Run optimize_design.py first.", file=sys.stderr)
        sys.exit(1)

    with open(p) as fh:
        all_candidates = json.load(fh)

    if not all_candidates:
        print("\n  No candidates found in JSON.  Re-run the optimizer.", file=sys.stderr)
        sys.exit(1)

    candidates = all_candidates[:top_n]

    # Load config and constants once
    config    = load_config(config_path)
    constants = _load_constants(config)

    print()
    print("=" * W)
    print(f"  MANUFACTURING REPORT — Top {len(candidates)} Optimizer Candidates")
    print(f"  Specs: ROM≈250mm | QRR 1.5–2.5 | Al 2024-T3 | 500g | 30 RPM")
    print("=" * W)

    pass_count = 0
    for rank, candidate in enumerate(candidates, 1):
        result = _validate_candidate(candidate, config, constants)
        _print_candidate_report(rank, candidate, result, constants, config)
        if result["passed"]:
            pass_count += 1

    print()
    print("=" * W)
    print(f"  SUMMARY: {pass_count} / {len(candidates)} candidates passed full physics validation.")
    if pass_count == 0:
        print("  RECOMMENDATION: None of the top candidates are manufacture-ready.")
        print("  Consider re-training the surrogate or regenerating data.")
    elif pass_count >= 1:
        print("  RECOMMENDATION: Rank 1 is the primary manufacture candidate.")
        print("  Cross-check ROM and QRR with the kinematic preview before 3D printing.")
    print("=" * W)
    print()


def main() -> None:
    args = _build_parser().parse_args()
    try:
        run(
            candidates_path=args.candidates,
            config_path=args.config,
            top_n=args.top,
        )
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
