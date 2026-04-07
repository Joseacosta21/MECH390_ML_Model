"""
Fatigue analysis module for Offset Crank-Slider Mechanism.

Implements Sections 9-13 of the Mother Doc for each structural component
(connecting rod, crank arm, pins) independently.

  - Section 9:  Stress cycling — mean and alternating components
  - Section 10: Endurance limit and Marin correction factors
  - Section 12: Fatigue failure criteria (Modified Goodman + ECY)
  - Section 13: Finite life analysis (Basquin / S-N curve + Miner's rule)

Inputs:
  Per-component sigma and tau histories (Pa) over one full crank revolution,
  collected by engine.py from stresses.evaluate() at each sweep step.

Material properties are read from the design dict (injected by generate.py
from baseline.yaml). Required keys:
    'S_ut'          — ultimate tensile strength (Pa)
    'S_y'           — yield strength (Pa)
    'Sn'            — fatigue strength at design life cycles (Pa)
    'basquin_A'     — Basquin intercept A (Pa); AA2024-T3: 924 MPa
    'basquin_b'     — Basquin slope b (dimensionless); AA2024-T3: -0.086
    'n_rpm'         — crank rotational speed (RPM)
    'total_cycles'  — design life in cycles for Miner's rule check

All units are SI throughout (Pa, m, s).
Ref: Mother Doc v7 Sections 9-13, instructions.md
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from mech390.physics._utils import get_or_warn

# Fallback defaults (overridden by config via design dict)
# Basquin S-N constants — AA2024-T3 experimental (fully reversed, R=-1)
# Anchors: (10^7 cycles, 230 MPa) and (10^9 cycles, 155 MPa)
# b = log10(155/230) / log10(1e9/1e7) = -0.086
# A = 230 / (1e7)^(-0.086) = 924 MPa
_BASQUIN_A_DEFAULT: float = 924e6   # Pa — baseline.yaml stress_analysis.basquin_A
_BASQUIN_B_DEFAULT: float = -0.086  # dimensionless — baseline.yaml stress_analysis.basquin_b
_C_SUR_DEFAULT: float = 0.88  # machined surface, Al 2024-T3 — baseline.yaml stress_analysis.C_sur
_C_ST_DEFAULT: float  = 1.0
_C_R_DEFAULT: float   = 0.81
_C_M_DEFAULT: float   = 1.0
_C_F_DEFAULT: float   = 1.0


# ---------------------------------------------------------------------------
# Fatigue correction factors (Mott Ch. 5)
# ---------------------------------------------------------------------------

def _C_s_size(d_mm: float) -> float:
    """
    Size factor C_s for any cross-section (Mott Table 5-3).

    Args:
        d_mm: equivalent diameter in mm
              (for rectangular sections: d_e = 0.808 * sqrt(w * t) * 1000;
               for circular pins: d_e = pin diameter * 1000)

    Returns:
        C_s (dimensionless)
    """
    if d_mm <= 7.62:
        return 1.0
    elif d_mm <= 50.0:
        return (d_mm / 7.62) ** (-0.11)
    else:
        return 0.859 - 0.000837 * d_mm


def _C_s_size_rect(w: float, t: float) -> float:
    """
    Size factor for rectangular cross-section in bending (Mott Table 5-3).

    Equivalent diameter per Mott: d_e = 0.808 * sqrt(w * t)

    Args:
        w: link width (m)
        t: link thickness (m)
    """
    d_e_mm = 0.808 * math.sqrt(w * t) * 1e3
    return _C_s_size(d_e_mm)


def _C_s_size_pin(d: float) -> float:
    """
    Size factor for circular pin section (Mott Table 5-3).

    Args:
        d: pin diameter (m)
    """
    return _C_s_size(d * 1e3)


def _sn_prime_rect(
    w: float, t: float, Sn: float,
    C_sur: float = _C_SUR_DEFAULT,
    C_st: float  = _C_ST_DEFAULT,
    C_R: float   = _C_R_DEFAULT,
    C_m: float   = _C_M_DEFAULT,
    C_f: float   = _C_F_DEFAULT,
) -> float:
    """
    Corrected fatigue strength for a rectangular link section (Mott Ch. 5).

        S'n = Sn * C_sur * C_s * C_st * C_R * C_m * C_f

    Args:
        w:    link width (m)
        t:    link thickness (m)
        Sn:   uncorrected fatigue strength at design life cycles (Pa); AA2024-T3: 133 MPa @ 18.72M cycles
        C_sur: surface finish factor (from config; Mott Table 5-4)
        C_st:  load / stress-type factor
        C_R:   reliability factor
        C_m:   material factor
        C_f:   miscellaneous factor

    Returns:
        S_n_prime (Pa)
    """
    C_s = _C_s_size_rect(w, t)
    return Sn * C_sur * C_s * C_st * C_R * C_m * C_f


def _sn_prime_pin(
    d: float, Sn: float,
    C_sur: float = _C_SUR_DEFAULT,
    C_st: float  = _C_ST_DEFAULT,
    C_R: float   = _C_R_DEFAULT,
    C_m: float   = _C_M_DEFAULT,
    C_f: float   = _C_F_DEFAULT,
) -> float:
    """
    Corrected fatigue strength for a circular pin section (Mott Ch. 5).

    Uses pin diameter as the equivalent diameter (circular section).
    Most conservative pin diameter (smallest) is recommended for S_n_prime_pin.

    Args:
        d:    pin diameter (m)
        Sn:   uncorrected fatigue strength at design life cycles (Pa); AA2024-T3: 133 MPa @ 18.72M cycles
        C_sur: surface finish factor (from config; Mott Table 5-4)
        C_st:  load / stress-type factor
        C_R:   reliability factor
        C_m:   material factor
        C_f:   miscellaneous factor

    Returns:
        S_n_prime (Pa)
    """
    C_s = _C_s_size_pin(d)
    return Sn * C_sur * C_s * C_st * C_R * C_m * C_f


# ---------------------------------------------------------------------------
# Section 9: Stress cycling helpers (Mother Doc Eqs 9.1-9.5)
# ---------------------------------------------------------------------------

def _stress_cycle(
    history: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Extract cycling parameters from a stress history array.

    Mother Doc Eqs 9.1-9.5:
        sigma_max = max(sigma(theta))
        sigma_min = min(sigma(theta))
        sigma_m   = (sigma_max + sigma_min) / 2
        sigma_a   = (sigma_max - sigma_min) / 2
        R         = sigma_min / sigma_max

    Args:
        history: 1-D array of stress values over one revolution (Pa)

    Returns:
        (s_max, s_min, s_m, s_a, R)
    """
    s_max = float(np.max(history))
    s_min = float(np.min(history))
    s_m   = (s_max + s_min) / 2.0
    s_a   = (s_max - s_min) / 2.0
    R     = s_min / s_max if abs(s_max) > 1e-12 else 0.0
    return s_max, s_min, s_m, s_a, R


# ---------------------------------------------------------------------------
# Section 13: Basquin / S-N finite life helpers (Mother Doc Eqs 13.1-13.7)
# ---------------------------------------------------------------------------

def _cycles_to_failure(
    sigma_a_eq: float,
    basquin_A: float = _BASQUIN_A_DEFAULT,
    basquin_b: float = _BASQUIN_B_DEFAULT,
) -> float:
    """
    Cycles to failure N_f from experimental Basquin S-N curve.

        σa = A · N_f^b  →  N_f = (σa / A)^(1/b)

    Source: AA2024-T3 experimental data, fully reversed (R=-1)
        Anchors: (10^7 cycles, 230 MPa) and (10^9 cycles, 155 MPa)
        b = -0.086,  A = 924 MPa
    Valid range: 10^5 to 10^9 cycles.

    Returns inf if sigma_a_eq <= 0 (no alternating stress).

    Args:
        sigma_a_eq: Von Mises equivalent alternating stress (Pa)
        basquin_A:  Basquin intercept (Pa)
        basquin_b:  Basquin slope (dimensionless, negative)
    """
    if sigma_a_eq <= 0.0:
        return float('inf')
    return (sigma_a_eq / basquin_A) ** (1.0 / basquin_b)


def _life_seconds(N_f: float, n_rpm: float) -> float:
    """
    Life in seconds (Mother Doc Eq 13.5): t_f = N_f / (n_rpm / 60).

    Args:
        N_f:   cycles to failure
        n_rpm: rotational speed (RPM)
    """
    if not math.isfinite(N_f) or n_rpm <= 0.0:
        return float('inf')
    return N_f / (n_rpm / 60.0)


# ---------------------------------------------------------------------------
# Per-component fatigue evaluation (Sections 9, 12, 13)
# ---------------------------------------------------------------------------

def _component_fatigue(
    sigma_history: np.ndarray,
    tau_history: np.ndarray,
    S_n_prime: float,
    S_ut: float,
    S_y: float,
    n_rpm: float,
    total_cycles: Optional[float],
    basquin_A: float = _BASQUIN_A_DEFAULT,
    basquin_b: float = _BASQUIN_B_DEFAULT,
) -> Dict[str, Any]:
    """
    Compute full fatigue analysis for one structural component.

    Args:
        sigma_history:  normal stress over one revolution (Pa)
        tau_history:    shear stress over one revolution (Pa)
        S_n_prime:      corrected fatigue strength for this component (Pa)
        S_ut:           ultimate tensile strength (Pa)
        S_y:            yield strength (Pa)
        n_rpm:          crank speed (RPM)
        total_cycles:   design life in cycles (None = skip Miner)
        basquin_A:      Basquin intercept A (Pa); AA2024-T3: 924 MPa
        basquin_b:      Basquin slope b (dimensionless); AA2024-T3: -0.086

    Returns:
        dict with generic keys (prefixed by caller): sigma_max, sigma_min,
        sigma_m, sigma_a, tau_m, tau_a, R, sigma_a_eq, sigma_m_eq, S_n_prime,
        n_f, n_y, n, b_B, N_f, t_f, D, failed_miner.
    """
    # --- Section 9: Stress cycling ---
    s_max, s_min, sigma_m, sigma_a, R = _stress_cycle(sigma_history)
    t_max, t_min, tau_m, tau_a, _     = _stress_cycle(tau_history)

    # Von Mises equivalent alternating and mean stresses (Eqs 9.6-9.7)
    sigma_a_eq = math.sqrt(sigma_a**2 + 3.0 * tau_a**2)
    sigma_m_eq = math.sqrt(sigma_m**2 + 3.0 * tau_m**2)

    # --- Section 12.1: Modified Goodman fatigue safety factor (Eq 12.2) ---
    # n_f = 1 / (sigma_a_eq/S'n + sigma_m_eq/S_ut)
    denom_goodman = sigma_a_eq / S_n_prime + sigma_m_eq / S_ut
    n_f = 1.0 / denom_goodman if denom_goodman > 0.0 else float('inf')

    # --- Section 12.2: ECY safety factor (Eq 12.4) ---
    # n_y = S_y / (sigma_a_eq + sigma_m_eq)
    denom_ecy = sigma_a_eq + sigma_m_eq
    n_y = S_y / denom_ecy if denom_ecy > 0.0 else float('inf')

    # --- Section 12.3: Governing safety factor (Eq 12.5) ---
    n = min(n_f, n_y)

    # --- Section 13: Basquin / S-N finite life ---
    # b_B is the experimental Basquin slope — same for all components
    b_B = basquin_b
    N_f = _cycles_to_failure(sigma_a_eq, basquin_A, basquin_b)
    t_f = _life_seconds(N_f, n_rpm)

    # --- Section 13.6-13.7: Miner's rule ---
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


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    sigma_rod_history:   Sequence[float],
    tau_rod_history:     Sequence[float],
    sigma_crank_history: Sequence[float],
    tau_crank_history:   Sequence[float],
    sigma_pin_history:   Sequence[float],
    tau_pin_history:     Sequence[float],
    design: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform full fatigue analysis for rod, crank, and pins over one crank
    revolution.

    Called by engine.py after the full 15-degree sweep completes, passing
    the per-step stress arrays collected from stresses.evaluate().

    Material properties required in design dict (injected by generate.py):
        'S_ut'       — ultimate tensile strength (Pa)
        'S_y'        — yield strength (Pa)
        'Sn'         — fatigue strength at design life (Pa); AA2024-T3: 133 MPa
        'basquin_A'  — Basquin intercept A (Pa); AA2024-T3: 924 MPa
        'basquin_b'  — Basquin slope b (dimensionless); AA2024-T3: -0.086
        'n_rpm'      — crank speed (RPM)
        'total_cycles' — design life in cycles

    Geometry required in design dict:
        'width_l', 'thickness_l'              — rod cross-section (m)
        'width_r', 'thickness_r'              — crank cross-section (m)
        'pin_diameter_A/B/C'                  — pin diameters (m)

    Args:
        sigma_rod_history   : normal stress in rod at each sweep step (Pa)
        tau_rod_history     : shear stress in rod at each sweep step (Pa)
        sigma_crank_history : normal stress in crank at each sweep step (Pa)
        tau_crank_history   : shear stress in crank at each sweep step (Pa)
        sigma_pin_history   : normal stress in pins at each sweep step (Pa)
        tau_pin_history     : shear stress in pins at each sweep step (Pa)
        design              : design parameter dict (see above)

    Returns:
        dict with keys prefixed by component name:
            Rod   : 'sigma_max_rod', 'n_f_rod', 'n_y_rod', 'n_rod', 'N_f_rod',
                    't_f_rod', 'D_rod', 'failed_miner_rod', 'S_n_prime_rod',
                    'sigma_a_eq_rod', 'sigma_m_eq_rod', 'b_B_rod', ...
            Crank : same pattern with '_crank' suffix
            Pin   : same pattern with '_pin' suffix
    """
    # --- Material properties (from design dict; fallback to 2024-T3 Al defaults) ---
    _ctx = 'fatigue.evaluate'
    S_ut = float(get_or_warn(design, 'S_ut', 483e6, context=_ctx))
    S_y  = float(get_or_warn(design, 'S_y',  345e6, context=_ctx))
    Sn   = float(get_or_warn(design, 'Sn',   133e6, context=_ctx))
    n_rpm = float(get_or_warn(design, 'n_rpm', 30.0, context=_ctx))
    total_cycles_raw = get_or_warn(design, 'total_cycles', None, context=_ctx)
    total_cycles: Optional[float] = (
        float(total_cycles_raw) if total_cycles_raw is not None else None
    )

    # Configurable fatigue constants (from baseline.yaml via design dict)
    basquin_A = float(get_or_warn(design, 'basquin_A', _BASQUIN_A_DEFAULT, context=_ctx))
    basquin_b = float(get_or_warn(design, 'basquin_b', _BASQUIN_B_DEFAULT, context=_ctx))
    C_sur     = float(get_or_warn(design, 'C_sur', _C_SUR_DEFAULT, context=_ctx))
    C_st     = float(get_or_warn(design, 'C_st',  _C_ST_DEFAULT,  context=_ctx))
    C_R      = float(get_or_warn(design, 'C_R',   _C_R_DEFAULT,   context=_ctx))
    C_m      = float(get_or_warn(design, 'C_m',   _C_M_DEFAULT,   context=_ctx))
    C_f      = float(get_or_warn(design, 'C_f',   _C_F_DEFAULT,   context=_ctx))

    # --- Link dimensions ---
    w_rod   = float(design['width_l'])
    t_rod   = float(design['thickness_l'])
    w_crank = float(design['width_r'])
    t_crank = float(design['thickness_r'])
    D_pA    = float(design['pin_diameter_A'])
    D_pB    = float(design['pin_diameter_B'])
    D_pC    = float(design['pin_diameter_C'])

    # --- Corrected fatigue strengths (Mott Ch. 5) ---
    S_n_prime_rod   = _sn_prime_rect(w_rod,   t_rod,   Sn, C_sur, C_st, C_R, C_m, C_f)
    S_n_prime_crank = _sn_prime_rect(w_crank, t_crank, Sn, C_sur, C_st, C_R, C_m, C_f)
    # For pins: use smallest pin diameter — most conservative size factor
    d_pin_min = min(D_pA, D_pB, D_pC)
    S_n_prime_pin   = _sn_prime_pin(d_pin_min, Sn, C_sur, C_st, C_R, C_m, C_f)

    # --- Convert histories to numpy arrays ---
    arr_sig_rod   = np.asarray(sigma_rod_history,   dtype=float)
    arr_tau_rod   = np.asarray(tau_rod_history,     dtype=float)
    arr_sig_crank = np.asarray(sigma_crank_history, dtype=float)
    arr_tau_crank = np.asarray(tau_crank_history,   dtype=float)
    arr_sig_pin   = np.asarray(sigma_pin_history,   dtype=float)
    arr_tau_pin   = np.asarray(tau_pin_history,     dtype=float)

    # --- Per-component fatigue (Sections 9, 12, 13) ---
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

    # --- Build output dict with component suffixes ---
    result: Dict[str, Any] = {}
    for suffix, comp in (('rod', rod), ('crank', crank), ('pin', pin)):
        for key, val in comp.items():
            result[f'{key}_{suffix}'] = val

    return result
