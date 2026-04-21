"""
summarize_results.py - physics-validate optimizer candidates and print a pre-manufacturing report.

Can be called standalone or imported by optimize_design.py.

Usage (standalone):
  python scripts/summarize_results.py
  python scripts/summarize_results.py --candidates data/results/candidates.json --top 3
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mech390.config import load_config
from mech390.physics import engine, kinematics as _kinematics, mass_properties as mp
from mech390.physics._utils import get_or_warn

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger("summarize_results")

W = 76   # report width

_CFG_PATH = PROJECT_ROOT / "configs" / "generate" / "baseline.yaml"


### CLI

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="summarize_results",
        description="Physics-validate top optimizer candidates and print a pre-manufacturing report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", "-c",
        default=str(PROJECT_ROOT / "data" / "results" / "candidates.json"),
        metavar="PATH", help="JSON written by optimize_design.py.")
    p.add_argument("--top", "-n", type=int, default=3, metavar="N",
        help="Number of candidates to validate.")
    return p


### Physics constants

def _load_constants(config: dict) -> dict:
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
    min_wall   = float(get_or_warn(sa_cfg, "min_wall_m",            0.5e-3, context=_ctx))

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


### Physics validation

def _validate_candidate(candidate: Dict[str, Any], config: dict, constants: dict) -> Dict[str, Any]:
    GEO_KEYS = ("r", "l", "e", "width_r", "thickness_r",
                "width_l", "thickness_l", "d_shaft_A", "pin_diameter_B", "pin_diameter_C")
    geo = {k: float(candidate[k]) for k in GEO_KEYS}

    kin_margin = geo["l"] - (geo["r"] + geo["e"])
    if kin_margin <= 0:
        return {"valid_physics": False, "kinematic_fail": True,
                "kin_margin": kin_margin, "passed": False, "checks": {}, "metrics": {}, "design_eval": geo}

    c = constants
    design_eval = dict(geo)
    design_eval.update({"omega": c["omega"], "m_block": c["m_block"], "mu": c["mu"], "g": c["g"]})

    mass_props = mp.compute_design_mass_properties(design_eval, config)
    design_eval.update(mass_props)
    design_eval["total_mass"] = (
        design_eval.get("mass_crank", 0.0)
        + design_eval.get("mass_rod",   0.0)
        + design_eval.get("mass_slider", 0.0)
    )

    _slider_cfg = config.get("geometry", {}).get("slider", {})
    _s_h = float(_slider_cfg.get("height", 0.02))
    _s_l = float(_slider_cfg.get("length", 0.02))
    design_eval["slider_height"] = _s_h
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

    metrics = engine.evaluate_design(design_eval)
    if not metrics.get("valid_physics", False):
        return {"valid_physics": False, "kinematic_fail": False, "passed": False,
                "checks": {}, "metrics": metrics, "design_eval": design_eval}

    sigma_limit = c["sigma_limit"]
    tau_limit   = c["tau_limit"]
    checks: dict = {}

    s_max = metrics.get("sigma_max", 0.0)
    t_max = metrics.get("tau_max",   0.0)
    utilization = max(
        s_max / sigma_limit if sigma_limit > 0 else 0.0,
        t_max / tau_limit   if tau_limit   > 0 else 0.0,
    )
    checks["utilization"]  = utilization
    checks["check_static"] = utilization <= c["utilization_max"]

    for comp, key in (("rod", "sigma_rod_peak"), ("crank", "sigma_crank_peak"), ("pin", "sigma_pin_peak")):
        peak = float(metrics.get(key, 0.0) or 0.0)
        n    = sigma_limit / peak if peak > 0.0 else float("inf")
        checks[f"n_static_{comp}"]     = n
        checks[f"check_static_{comp}"] = n >= c["n_static_min"]

    n_buck = metrics.get("n_buck", float("inf"))
    checks["n_buck"]     = n_buck
    checks["check_buck"] = (n_buck >= c["n_buck_min"]) if math.isfinite(n_buck) else True

    for comp in ("rod", "crank", "pin"):
        n = metrics.get(f"n_{comp}", float("inf"))
        checks[f"n_fatigue_{comp}"]     = n
        checks[f"check_fatigue_{comp}"] = (n >= c["n_fatigue_min"]) if math.isfinite(n) else True

    for comp in ("rod", "crank", "pin"):
        D = metrics.get(f"D_{comp}")
        checks[f"D_{comp}"]           = D
        checks[f"check_miner_{comp}"] = (D < c["d_miner_max"]) if D is not None else True

    n_shaft = metrics.get("n_shaft", float("inf"))
    checks["n_shaft"]     = n_shaft
    checks["check_shaft"] = bool((n_shaft >= c["n_shaft_min"]) if math.isfinite(n_shaft) else True)

    all_checks = [
        checks["check_static"],
        checks["check_static_rod"], checks["check_static_crank"], checks["check_static_pin"],
        checks["check_buck"],
        checks["check_fatigue_rod"], checks["check_fatigue_crank"], checks["check_fatigue_pin"],
        checks["check_miner_rod"], checks["check_miner_crank"], checks["check_miner_pin"],
        checks["check_shaft"],
    ]
    return {"valid_physics": True, "kinematic_fail": False, "passed": all(all_checks),
            "checks": checks, "metrics": metrics, "design_eval": design_eval}


### Formatting helpers

def _fv(v: Optional[float], fmt: str = ".3f", unit: str = "") -> str:
    """Format a float; return '-' if None or non-finite."""
    if v is None or not math.isfinite(v):
        return "—"
    return f"{v:{fmt}}{unit}"


def _ok(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def _bar(char: str = "─", width: int = W) -> str:
    return "  " + char * (width - 2)


### Report sections

def _section_geometry(geo: dict, config: dict) -> None:
    """Two-column geometry block with ROM/QRR confirmation."""
    operating  = config.get("operating", {})
    rom_target = float(operating.get("ROM",           0.25))  * 1e3
    rom_tol    = float(operating.get("ROM_tolerance", 0.0005)) * 1e3
    _qrr_cfg   = operating.get("QRR", {})
    qrr_min_v  = float(_qrr_cfg.get("min", 1.5)) if isinstance(_qrr_cfg, dict) else 1.5
    qrr_max_v  = float(_qrr_cfg.get("max", 2.5)) if isinstance(_qrr_cfg, dict) else 2.5

    kin_m  = _kinematics.calculate_metrics(geo["r"], geo["l"], geo["e"])
    valid  = kin_m.get("valid", False)
    rom_mm = kin_m["ROM"] * 1e3 if valid else float("nan")
    qrr    = kin_m["QRR"]       if valid else float("nan")
    rom_ok = abs(rom_mm - rom_target) <= rom_tol if valid else False
    qrr_ok = qrr_min_v <= qrr <= qrr_max_v      if valid else False

    L = [
        ("r  (crank)",   f"{geo['r']*1e3:>7.2f} mm"),
        ("l  (rod)",     f"{geo['l']*1e3:>7.2f} mm"),
        ("e  (offset)",  f"{geo['e']*1e3:>7.2f} mm"),
        ("",             ""),
        ("ROM (actual)",  f"{rom_mm:>7.2f} mm  [{'OK  ' if rom_ok else 'FAIL'}]" if valid else "INVALID"),
        ("QRR (actual)",  f"{qrr:>7.3f}     [{'OK  ' if qrr_ok else 'FAIL'}]" if valid else "INVALID"),
    ]
    R = [
        ("width_r",       f"{geo['width_r']*1e3:>7.2f} mm"),
        ("thickness_r",   f"{geo['thickness_r']*1e3:>7.2f} mm"),
        ("width_l",       f"{geo['width_l']*1e3:>7.2f} mm"),
        ("thickness_l",   f"{geo['thickness_l']*1e3:>7.2f} mm"),
        ("pin_A (shaft)", f"{geo['d_shaft_A']*1e3:>7.2f} mm"),
        ("pin_B",         f"{geo['pin_diameter_B']*1e3:>7.2f} mm"),
        ("pin_C",         f"{geo['pin_diameter_C']*1e3:>7.2f} mm"),
    ]
    nrows = max(len(L), len(R))
    L += [("", "")] * (nrows - len(L))
    R += [("", "")] * (nrows - len(R))

    print(_bar())
    print(f"  {'GEOMETRY'}")
    print(_bar())
    for (lk, lv), (rk, rv) in zip(L, R):
        left  = f"  {lk:<16} {lv:<22}" if lk else " " * 40
        right = f"  {rk:<16} {rv}" if rk else ""
        print(f"{left}{right}")


def _section_performance(
    candidate:  Dict[str, Any],
    result:     Dict[str, Any],
    constants:  dict,
    is_dataset: bool = False,
) -> None:
    """Metric | Predicted | Actual | Limit | Status table for one candidate."""
    checks  = result["checks"]
    metrics = result["metrics"]
    design  = result["design_eval"]
    c       = constants

    def _pred(key: str) -> Optional[float]:
        return candidate.get(f"pred_{key}")

    # table header
    H_METRIC = "Metric"
    H_PRED   = "CSV actual" if is_dataset else "Predicted"
    H_ACTUAL = "Re-validated" if is_dataset else "Actual"
    H_LIMIT  = "Limit"
    H_STATUS = "Status"
    COL = (28, 12, 12, 10, 6)  # column widths

    def _row(metric, pred_s, actual_s, limit_s, status_s):
        return (f"  {metric:<{COL[0]}} {pred_s:>{COL[1]}} {actual_s:>{COL[2]}} "
                f"{limit_s:>{COL[3]}} {status_s:>{COL[4]}}")

    sep_row = "  " + "─" * (sum(COL) + 4 * 1 + 2)
    perf_title = ("PERFORMANCE  (CSV training values  vs  Physics re-validation)"
                  if is_dataset else
                  "PERFORMANCE  (Surrogate predicted  vs  Physics actual)")
    print()
    print(_bar())
    print(f"  {perf_title}")
    print(_bar())
    print(_row(H_METRIC, H_PRED, H_ACTUAL, H_LIMIT, H_STATUS))
    print(sep_row)

    # surrogate regression targets (mass/size/torque, informational)
    total_mass_act = design.get("total_mass", float("nan"))
    vol_act        = design.get("volume_envelope", float("nan"))
    tau_act        = metrics.get("tau_A_max", float("nan"))
    E_act          = metrics.get("E_rev", float("nan"))

    print(_row("total_mass",
               _fv(_pred("total_mass") * 1e3 if _pred("total_mass") is not None else None, ".1f", " g"),
               _fv(total_mass_act * 1e3,  ".1f", " g"),
               "—", "—"))
    print(_row("volume_envelope",
               _fv(_pred("volume_envelope") * 1e6 if _pred("volume_envelope") else None, ".2f", " cm³"),
               _fv(vol_act * 1e6, ".2f", " cm³"),
               "—", "—"))
    print(_row("tau_A_max",
               _fv(_pred("tau_A_max"),  ".4f", " N·m"),
               _fv(tau_act,            ".4f", " N·m"),
               "—", "—"))
    print(_row("E_rev",
               _fv(_pred("E_rev"),  ".4f", " J"),
               _fv(E_act,          ".4f", " J"),
               "—", "—"))

    print(sep_row)

    # safety factor rows (predicted + actual + limit + status)
    def _fos_row(label, pred_val, actual_val, limit, ok, minimize=False, fmt=".2f",
                 nonneg: bool = True):
        # show '-' if the model predicted a negative value for a non-negative quantity
        pv = None if (nonneg and pred_val is not None and pred_val < 0.0) else pred_val
        pred_s   = _fv(pv,          fmt) if pv is not None else "—"
        actual_s = _fv(actual_val,  fmt)
        limit_s  = f"{'≤' if minimize else '≥'}{limit:{fmt}}"
        return _row(label, pred_s, actual_s, limit_s, _ok(ok))

    print(_fos_row("utilization",
                   _pred("utilization"), checks.get("utilization"),
                   c["utilization_max"], checks["check_static"],
                   minimize=True, fmt=".3f"))

    print(_fos_row("n_buck",
                   _pred("n_buck"), checks.get("n_buck"),
                   c["n_buck_min"], checks["check_buck"]))

    print(_fos_row("n_shaft",
                   _pred("n_shaft"), checks.get("n_shaft"),
                   c["n_shaft_min"], checks["check_shaft"]))

    min_n_static_act = min(
        checks.get("n_static_rod",   float("inf")),
        checks.get("n_static_crank", float("inf")),
        checks.get("n_static_pin",   float("inf")),
    )
    print(_fos_row("min_n_static  (min of 3 links)",
                   _pred("min_n_static"), min_n_static_act,
                   c["n_static_min"], all([checks["check_static_rod"],
                                          checks["check_static_crank"],
                                          checks["check_static_pin"]])))

    min_n_fat_act = min(
        checks.get("n_fatigue_rod",   float("inf")),
        checks.get("n_fatigue_crank", float("inf")),
        checks.get("n_fatigue_pin",   float("inf")),
    )
    print(_fos_row("min_n_fatigue  (min of 3 links)",
                   _pred("min_n_fatigue"), min_n_fat_act,
                   c["n_fatigue_min"], all([checks["check_fatigue_rod"],
                                           checks["check_fatigue_crank"],
                                           checks["check_fatigue_pin"]])))

    print(sep_row)

    # per-link breakdown (physics only, no surrogate prediction)
    print(_row("  - per-link detail (physics only) -", "", "", "", ""))

    for comp in ("rod", "crank", "pin"):
        print(_fos_row(f"  n_static  {comp}",
                       None, checks.get(f"n_static_{comp}"),
                       c["n_static_min"], checks[f"check_static_{comp}"]))

    for comp in ("rod", "crank", "pin"):
        print(_fos_row(f"  n_fatigue {comp}",
                       None, checks.get(f"n_fatigue_{comp}"),
                       c["n_fatigue_min"], checks[f"check_fatigue_{comp}"]))

    for comp in ("rod", "crank", "pin"):
        D   = checks.get(f"D_{comp}")
        ok_ = checks.get(f"check_miner_{comp}", True)
        print(_fos_row(f"  D_miner   {comp}",
                       None, D, c["d_miner_max"], ok_,
                       minimize=True, fmt=".4f"))

        t_f = metrics.get(f"t_f_{comp}")
        if t_f is None or not math.isfinite(t_f) or t_f > 3.1536e13:
            life_str = "Infinite"
        else:
            years = t_f / (3600.0 * 24.0 * 365.25)
            life_str = ">10^6 yrs" if years >= 1e6 else f"{years:.1f} yrs"

        print(_row(f"    Life    {comp}", "—", life_str, "—", "—"))

    print(sep_row)


def _section_net_sections(geo: dict, min_net: float) -> None:
    pairs = [
        ("width_r - pin_A", geo["width_r"], geo["d_shaft_A"]),
        ("width_r - pin_B", geo["width_r"], geo["pin_diameter_B"]),
        ("width_l - pin_B", geo["width_l"], geo["pin_diameter_B"]),
        ("width_l - pin_C", geo["width_l"], geo["pin_diameter_C"]),
    ]
    print()
    print(f"  NET-SECTION MARGINS  (need > 0  |  min_net = {min_net*1e3:.3f} mm)")
    parts = []
    for label, w, d in pairs:
        margin = w - d - min_net
        flag = "OK  " if margin > 0 else "FAIL"
        parts.append(f"  {label}: {margin*1e3:+.2f} mm [{flag}]")
    # print two per line
    for i in range(0, len(parts), 2):
        line = parts[i]
        if i + 1 < len(parts):
            line = f"{line:<42}{parts[i+1]}"
        print(line)


### Per-candidate report

def _print_candidate_report(
    rank:      int,
    candidate: Dict[str, Any],
    result:    Dict[str, Any],
    constants: dict,
    config:    dict,
    label:     Optional[str] = None,
) -> None:
    GEO_KEYS = ("r", "l", "e", "width_r", "thickness_r",
                "width_l", "thickness_l", "d_shaft_A", "pin_diameter_B", "pin_diameter_C")
    geo            = {k: float(candidate[k]) for k in GEO_KEYS}
    pass_prob      = candidate.get("pass_prob",      float("nan"))
    weighted_score = candidate.get("weighted_score", float("nan"))
    verdict        = "PASS" if result.get("passed") else "FAIL"
    is_dataset     = candidate.get("source") == "dataset"

    if label is None:
        label = "DATASET BEST  (actual train values)" if is_dataset else f"RANK {rank}"

    score_label = "Objective score" if is_dataset else "Surrogate score"

    print()
    print("═" * W)
    print(f"  {label}  │  {score_label}: {weighted_score:.4f}"
          f"  │  pass_prob: {pass_prob:.1%}  │  Physics: {verdict}")
    print("═" * W)

    _section_geometry(geo, config)

    if not result["valid_physics"]:
        print()
        if result.get("kinematic_fail"):
            print(f"  ✗ KINEMATIC INFEASIBILITY  l-(r+e) = {result['kin_margin']*1e3:+.2f} mm  (must be > 0)")
        else:
            print("  ✗ PHYSICS ENGINE ERROR - numerical singularity or NaN in dynamics solver")
        print()
        print("═" * W)
        print(f"  OVERALL VERDICT: FAIL")
        print("═" * W)
        return

    _section_performance(candidate, result, constants, is_dataset=is_dataset)
    _section_net_sections(geo, constants["min_net"])

    print()
    print("═" * W)
    print(f"  OVERALL VERDICT: {verdict}")
    print("═" * W)


### Entry point

def run(candidates_path: str, top_n: int, config_path: str | None = None) -> int:
    """Validate and report top_n candidates. Returns pass count."""
    p = Path(candidates_path)
    if not p.exists():
        logger.error("Candidates file not found: %s", p)
        print(f"\n  ERROR: {p} not found.  Run optimize_design.py first.", file=sys.stderr)
        sys.exit(1)

    with open(p) as fh:
        all_candidates = json.load(fh)

    if not all_candidates:
        print("\n  No candidates in JSON. Re-run the optimizer.", file=sys.stderr)
        sys.exit(1)

    # separate surrogate candidates from dataset best
    surrogate_candidates = [c for c in all_candidates if c.get("source") != "dataset"]
    dataset_best         = next((c for c in all_candidates if c.get("source") == "dataset"), None)

    candidates = surrogate_candidates[:top_n]
    config     = load_config(config_path if config_path else str(_CFG_PATH))
    constants  = _load_constants(config)

    print()
    print("═" * W)
    print(f"  PRE-MANUFACTURING REPORT - Top {len(candidates)} Optimizer Candidates")
    print(f"  Specs: ROM~250mm | QRR 1.5-2.5 | Al 2024-T3 | 500g load | 30 RPM")
    print(f"  Material: S_y={constants['S_y']/1e6:.0f} MPa  "
          f"| sigma_limit={constants['sigma_limit']/1e6:.0f} MPa  "
          f"| Cycles={int(constants['total_cycles']):,}")
    print("═" * W)

    pass_count = 0
    for rank, candidate in enumerate(candidates, 1):
        result = _validate_candidate(candidate, config, constants)
        _print_candidate_report(rank, candidate, result, constants, config)
        if result.get("passed"):
            pass_count += 1

    # summary
    print()
    print("═" * W)
    print(f"  SUMMARY  {pass_count}/{len(candidates)} candidates passed full physics validation")
    if pass_count == 0:
        print("  > No manufacture-ready candidate found.")
        print("  > Re-run optimizer or retrain surrogate with more data.")
    else:
        print("  > Rank 1 is the primary manufacture candidate.")
        if pass_count < len(candidates):
            print(f"  > Ranks {pass_count+1}-{len(candidates)} failed physics - do not manufacture.")
    print("═" * W)

    if dataset_best is not None:
        print()
        print("─" * W)
        print(f"  DATASET BEST - highest-scoring design in training data (actual physics values)")
        print(f"  Compare against Rank 1 to assess surrogate optimizer quality.")
        print("─" * W)
        db_result = _validate_candidate(dataset_best, config, constants)
        _print_candidate_report(0, dataset_best, db_result, constants, config)

    print()

    return pass_count


def main() -> None:
    args = _build_parser().parse_args()
    try:
        run(candidates_path=args.candidates, top_n=args.top)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
