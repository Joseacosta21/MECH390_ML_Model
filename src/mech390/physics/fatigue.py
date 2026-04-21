"""
Fatigue analysis for the offset crank-slider mechanism.

Computes rod, crank, and pin fatigue over one full crank revolution using
modified Goodman, ECY static yield check, Basquin S-N curve, and Miner's rule.

Inputs are per-component sigma and tau histories (Pa) collected by engine.py
from stresses.evaluate() at each sweep step.

Material properties read from the design dict (injected from baseline.yaml):
    'S_ut'          - ultimate tensile strength (Pa)
    'S_y'           - yield strength (Pa)
    'Sn'            - fatigue strength at design life cycles (Pa)
    'basquin_A'     - Basquin intercept A (Pa); AA2024-T3: 924 MPa
    'basquin_b'     - Basquin slope b (dimensionless); AA2024-T3: -0.086
    'n_rpm'         - crank rotational speed (RPM)
    'total_cycles'  - design life in cycles for Miner's rule check

All units are SI throughout (Pa, m, s).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from mech390.physics._utils import get_or_warn

# All Marin correction factors and Basquin constants are required arguments.
# They must be injected from the design dict (read from baseline.yaml by generate.py).
# Missing keys surface via get_or_warn() in evaluate() - no silent fallbacks.


### Fatigue correction factors

# size factor C_s for any cross-section
# d_mm is equivalent diameter in mm (rectangular: d_e = 0.808 * sqrt(w * t) * 1000;
# circular pins: d_e = pin_diameter * 1000)
def _C_s_size(d_mm: float) -> float:
    if d_mm <= 7.62:
        return 1.0
    elif d_mm <= 50.0:
        return (d_mm / 7.62) ** (-0.11)
    else:
        return 0.859 - 0.000837 * d_mm


# size factor for rectangular cross-section in bending
# equivalent diameter: d_e = 0.808 * sqrt(w * t)
def _C_s_size_rect(w: float, t: float) -> float:
    d_e_mm = 0.808 * math.sqrt(w * t) * 1e3
    return _C_s_size(d_e_mm)


# size factor for circular pin section
def _C_s_size_pin(d: float) -> float:
    return _C_s_size(d * 1e3)


# corrected fatigue strength for a rectangular link section
# S_n_prime = Sn * C_sur * C_s * C_st * C_R * C_m * C_f
def _sn_prime_rect(
    w: float, t: float, Sn: float,
    C_sur: float,
    C_st: float,
    C_R: float,
    C_m: float,
    C_f: float,
) -> float:
    C_s = _C_s_size_rect(w, t)
    return Sn * C_sur * C_s * C_st * C_R * C_m * C_f


# corrected fatigue strength for a circular pin section
# uses pin diameter as equivalent diameter; use smallest pin for most conservative result
def _sn_prime_pin(
    d: float, Sn: float,
    C_sur: float,
    C_st: float,
    C_R: float,
    C_m: float,
    C_f: float,
) -> float:
    C_s = _C_s_size_pin(d)
    return Sn * C_sur * C_s * C_st * C_R * C_m * C_f


### Stress cycling helpers

# extracts cycling parameters from a stress history array
# returns (s_max, s_min, s_m, s_a, R)
def _stress_cycle(
    history: np.ndarray,
) -> tuple[float, float, float, float, float]:
    s_max = float(np.max(history))
    s_min = float(np.min(history))
    s_m   = (s_max + s_min) / 2.0
    s_a   = (s_max - s_min) / 2.0
    R     = s_min / s_max if abs(s_max) > 1e-12 else 0.0
    return s_max, s_min, s_m, s_a, R


### Basquin / S-N finite life helpers

# cycles to failure from Basquin S-N curve
# sigma_a = A * N_f^b  ->  N_f = (sigma_a / A)^(1/b)
# AA2024-T3: b = -0.086, A = 924 MPa; valid range ~10^5 to 10^9 cycles
# returns inf if sigma_a_eq <= 0 (no alternating stress)
def _cycles_to_failure(
    sigma_a_eq: float,
    basquin_A: float,
    basquin_b: float,
) -> float:
    if sigma_a_eq <= 0.0:
        return float('inf')
    return (sigma_a_eq / basquin_A) ** (1.0 / basquin_b)


# life in seconds: t_f = N_f / (n_rpm / 60)
def _life_seconds(N_f: float, n_rpm: float) -> float:
    if not math.isfinite(N_f) or n_rpm <= 0.0:
        return float('inf')
    return N_f / (n_rpm / 60.0)


### Per-component fatigue evaluation

# full fatigue analysis for one structural component over one revolution
def _component_fatigue(
    sigma_history: np.ndarray,
    tau_history: np.ndarray,
    S_n_prime: float,
    S_ut: float,
    S_y: float,
    n_rpm: float,
    total_cycles: Optional[float],
    basquin_A: float,
    basquin_b: float,
) -> Dict[str, Any]:
    # stress cycling
    s_max, s_min, sigma_m, sigma_a, R = _stress_cycle(sigma_history)
    t_max, t_min, tau_m, tau_a, _     = _stress_cycle(tau_history)

    # Von Mises equivalent alternating and mean stresses
    sigma_a_eq = math.sqrt(sigma_a**2 + 3.0 * tau_a**2)
    sigma_m_eq = math.sqrt(sigma_m**2 + 3.0 * tau_m**2)

    # modified Goodman fatigue safety factor
    # n_f = 1 / (sigma_a_eq/S_n_prime + sigma_m_eq/S_ut)
    denom_goodman = sigma_a_eq / S_n_prime + sigma_m_eq / S_ut
    n_f = 1.0 / denom_goodman if denom_goodman > 0.0 else float('inf')

    # ECY (static yield) safety factor
    # n_y = S_y / (sigma_a_eq + sigma_m_eq)
    denom_ecy = sigma_a_eq + sigma_m_eq
    n_y = S_y / denom_ecy if denom_ecy > 0.0 else float('inf')

    # governing safety factor
    n = min(n_f, n_y)

    # Basquin S-N finite life
    # b_B is the experimental Basquin slope - same for all components
    b_B = basquin_b
    N_f = _cycles_to_failure(sigma_a_eq, basquin_A, basquin_b)
    t_f = _life_seconds(N_f, n_rpm)

    # Miner's rule
    D: Optional[float] = None
    failed_miner: Optional[bool] = None
    if total_cycles is not None:
        D = total_cycles / N_f if math.isfinite(N_f) else 0.0
        failed_miner = D >= 1.0

    return {
        'sigma_max':    s_max,
        'sigma_min':    s_min,
        'sigma_m':      sigma_m,
        'sigma_a':      sigma_a,
        'tau_m':        tau_m,
        'tau_a':        tau_a,
        'R':            R,
        'sigma_a_eq':   sigma_a_eq,
        'sigma_m_eq':   sigma_m_eq,
        'S_n_prime':    S_n_prime,
        'n_f':          n_f,
        'n_y':          n_y,
        'n':            n,
        'b_B':          b_B,
        'N_f':          N_f,
        't_f':          t_f,
        'D':            D,
        'failed_miner': failed_miner,
    }


### Public interface

def evaluate(
    sigma_rod_history:   Sequence[float],
    tau_rod_history:     Sequence[float],
    sigma_crank_history: Sequence[float],
    tau_crank_history:   Sequence[float],
    sigma_pin_history:   Sequence[float],
    tau_pin_history:     Sequence[float],
    design: Dict[str, Any],
) -> Dict[str, Any]:
    """Full fatigue analysis for rod, crank, and pins over one crank revolution."""
    # material properties (from design dict; fallback to 2024-T3 Al defaults)
    _ctx = 'fatigue.evaluate'
    S_ut = float(get_or_warn(design, 'S_ut', 483e6, context=_ctx))
    S_y  = float(get_or_warn(design, 'S_y',  345e6, context=_ctx))
    Sn   = float(get_or_warn(design, 'Sn',   133e6, context=_ctx))
    n_rpm = float(get_or_warn(design, 'n_rpm', 30.0, context=_ctx))
    total_cycles_raw = get_or_warn(design, 'total_cycles', None, context=_ctx)
    total_cycles: Optional[float] = (
        float(total_cycles_raw) if total_cycles_raw is not None else None
    )

    # configurable fatigue constants - must be injected from baseline.yaml
    try:
        basquin_A = float(design['basquin_A'])
        basquin_b = float(design['basquin_b'])
        C_sur     = float(design['C_sur'])
        C_st      = float(design['C_st'])
        C_R       = float(design['C_R'])
        C_m       = float(design['C_m'])
        C_f       = float(design['C_f'])
    except KeyError as exc:
        raise KeyError(
            f"fatigue.evaluate: required key {exc} missing from design dict. "
            f"Ensure generate.py / validate_candidate.py injects all stress_analysis "
            f"constants from baseline.yaml before calling engine.evaluate_design()."
        ) from exc

    # link dimensions
    w_rod   = float(design['width_l'])
    t_rod   = float(design['thickness_l'])
    w_crank = float(design['width_r'])
    t_crank = float(design['thickness_r'])
    D_pB    = float(design['pin_diameter_B'])
    D_pC    = float(design['pin_diameter_C'])

    # corrected fatigue strengths
    S_n_prime_rod   = _sn_prime_rect(w_rod,   t_rod,   Sn, C_sur, C_st, C_R, C_m, C_f)
    S_n_prime_crank = _sn_prime_rect(w_crank, t_crank, Sn, C_sur, C_st, C_R, C_m, C_f)
    # for lug pins B and C: use smallest diameter; d_shaft_A uses a separate check in engine.py
    d_pin_min = min(D_pB, D_pC)
    S_n_prime_pin   = _sn_prime_pin(d_pin_min, Sn, C_sur, C_st, C_R, C_m, C_f)

    # convert histories to numpy arrays
    arr_sig_rod   = np.asarray(sigma_rod_history,   dtype=float)
    arr_tau_rod   = np.asarray(tau_rod_history,     dtype=float)
    arr_sig_crank = np.asarray(sigma_crank_history, dtype=float)
    arr_tau_crank = np.asarray(tau_crank_history,   dtype=float)
    arr_sig_pin   = np.asarray(sigma_pin_history,   dtype=float)
    arr_tau_pin   = np.asarray(tau_pin_history,     dtype=float)

    # per-component fatigue
    rod   = _component_fatigue(
        arr_sig_rod,   arr_tau_rod,
        S_n_prime_rod,   S_ut, S_y, n_rpm, total_cycles, basquin_A, basquin_b,
    )
    crank = _component_fatigue(
        arr_sig_crank, arr_tau_crank,
        S_n_prime_crank, S_ut, S_y, n_rpm, total_cycles, basquin_A, basquin_b,
    )
    pin   = _component_fatigue(
        arr_sig_pin,   arr_tau_pin,
        S_n_prime_pin,   S_ut, S_y, n_rpm, total_cycles, basquin_A, basquin_b,
    )

    # build output dict with component suffixes
    result: Dict[str, Any] = {}
    for suffix, comp in (('rod', rod), ('crank', crank), ('pin', pin)):
        for key, val in comp.items():
            result[f'{key}_{suffix}'] = val

    return result
