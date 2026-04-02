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
    'S_prime_e'     — reference fatigue strength at 10^8 cycles (Pa)
    'sigma_f_prime' — fatigue strength coefficient (Pa)
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
_N_BASQUIN_ANCHOR_DEFAULT: float = 2.0e6   # cycles — baseline.yaml stress_analysis.N_basquin_anchor
_Z_A_RELIABILITY_DEFAULT: float = 3.091    # 99.9% — baseline.yaml stress_analysis.z_a_reliability


# ---------------------------------------------------------------------------
# Marin correction factors (Mother Doc Section 10.2)
# ---------------------------------------------------------------------------

def _k_a(S_ut: float) -> float:
    """
    Surface factor for machined finish (Mother Doc Section 10.2).

    k_a = 4.51 * S_ut^(-0.265)  with S_ut in MPa.

    Args:
        S_ut: ultimate tensile strength (Pa)
    """
    S_ut_MPa = S_ut / 1e6
    return 4.51 * S_ut_MPa ** (-0.265)


def _k_b_rect(w: float, t: float) -> float:
    """
    Size factor for rectangular cross-section in bending (Mother Doc Eq 10.2).

    Equivalent diameter: d_e = 0.808 * sqrt(w * t)
    Piecewise per Shigley / Mother Doc (d_e in mm):
      d_e <= 51 mm : k_b = 0.879 * d_e^(-0.107)
      d_e >  51 mm : k_b = 1.24  * d_e^(-0.107)

    Args:
        w: link width (m)
        t: link thickness (m)
    """
    d_e_mm = 0.808 * math.sqrt(w * t) * 1e3
    if d_e_mm <= 51.0:
        return 0.879 * d_e_mm ** (-0.107)
    return 1.24 * d_e_mm ** (-0.107)


def _k_b_pin(d: float) -> float:
    """
    Size factor for circular pin section (d_e = pin diameter).

    Args:
        d: pin diameter (m)
    """
    d_mm = d * 1e3
    if d_mm <= 51.0:
        return 0.879 * d_mm ** (-0.107)
    return 1.24 * d_mm ** (-0.107)


def _k_c() -> float:
    """Load factor = 1.0 (Von Mises applied separately; Mother Doc Section 10.2)."""
    return 1.0


def _k_d() -> float:
    """Temperature factor = 1.0 for T <= 70°C (Mother Doc Section 10.2)."""
    return 1.0


def _k_e(z_a: float = _Z_A_RELIABILITY_DEFAULT) -> float:
    """
    Reliability factor (Mother Doc Section 10.2).

    k_e = 1 - 0.08 * z_a
    Default z_a = 3.091 (99.9% reliability).

    Args:
        z_a: reliability z-value (dimensionless)
    """
    return 1.0 - 0.08 * z_a


def _k_f() -> float:
    """Miscellaneous factor = 1.0 (no fretting; Mother Doc Section 10.2)."""
    return 1.0


def _endurance_limit_rect(
    w: float, t: float, S_ut: float, S_prime_e: float,
    z_a: float = _Z_A_RELIABILITY_DEFAULT,
) -> float:
    """
    Corrected endurance limit for a rectangular link section (Mother Doc Eq 10.1).

        S_e = k_a * k_b * k_c * k_d * k_e * k_f * S'_e

    Args:
        w: link width (m)
        t: link thickness (m)
        S_ut: ultimate tensile strength (Pa)
        S_prime_e: reference fatigue strength at 10^8 cycles (Pa)
        z_a: reliability z-value (dimensionless)

    Returns:
        S_e (Pa)
    """
    return _k_a(S_ut) * _k_b_rect(w, t) * _k_c() * _k_d() * _k_e(z_a) * _k_f() * S_prime_e


def _endurance_limit_pin(
    d: float, S_ut: float, S_prime_e: float,
    z_a: float = _Z_A_RELIABILITY_DEFAULT,
) -> float:
    """
    Corrected endurance limit for a circular pin section (Mother Doc Eq 10.1).

    Uses pin diameter as the equivalent diameter (circular section).
    Most conservative pin diameter (smallest) is recommended for S_e_pin.

    Args:
        d: pin diameter (m)
        S_ut: ultimate tensile strength (Pa)
        S_prime_e: reference fatigue strength at 10^8 cycles (Pa)
        z_a: reliability z-value (dimensionless)

    Returns:
        S_e (Pa)
    """
    return _k_a(S_ut) * _k_b_pin(d) * _k_c() * _k_d() * _k_e(z_a) * _k_f() * S_prime_e


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

def _basquin_exponent(
    S_e: float, S_ut: float,
    N_anchor: float = _N_BASQUIN_ANCHOR_DEFAULT,
) -> float:
    """
    Basquin exponent b_B (Mother Doc Eq 13.3).

    Two-point S-N method:
        b_B = -log10(0.9 * S_ut / S_e) / log10(2 * N_anchor)

    Args:
        S_e:      corrected endurance limit (Pa)
        S_ut:     ultimate tensile strength (Pa)
        N_anchor: S-N curve second anchor point (cycles)
    """
    return -math.log10(0.9 * S_ut / S_e) / math.log10(2.0 * N_anchor)


def _cycles_to_failure(
    sigma_a_eq: float,
    S_e: float,
    S_ut: float,
    sigma_f_prime: float,
    N_anchor: float = _N_BASQUIN_ANCHOR_DEFAULT,
) -> float:
    """
    Cycles to failure N_f from Basquin relation (Mother Doc Eq 13.4).

        N_f = (1/2) * (sigma_a_eq / sigma'_f)^(1/b_B)

    Returns inf if sigma_a_eq <= 0 (no alternating stress).

    Args:
        sigma_a_eq:    Von Mises equivalent alternating stress (Pa)
        S_e:           corrected endurance limit (Pa)
        S_ut:          ultimate tensile strength (Pa)
        sigma_f_prime: fatigue strength coefficient (Pa)
        N_anchor:      S-N curve second anchor point (cycles)
    """
    if sigma_a_eq <= 0.0:
        return float('inf')
    b_B = _basquin_exponent(S_e, S_ut, N_anchor)
    return 0.5 * (sigma_a_eq / sigma_f_prime) ** (1.0 / b_B)


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
    S_e: float,
    S_ut: float,
    S_y: float,
    sigma_f_prime: float,
    n_rpm: float,
    total_cycles: Optional[float],
    N_anchor: float = _N_BASQUIN_ANCHOR_DEFAULT,
) -> Dict[str, Any]:
    """
    Compute full fatigue analysis for one structural component.

    Args:
        sigma_history:  normal stress over one revolution (Pa)
        tau_history:    shear stress over one revolution (Pa)
        S_e:            corrected endurance limit for this component (Pa)
        S_ut:           ultimate tensile strength (Pa)
        S_y:            yield strength (Pa)
        sigma_f_prime:  fatigue strength coefficient (Pa)
        n_rpm:          crank speed (RPM)
        total_cycles:   design life in cycles (None = skip Miner)
        N_anchor:       Basquin S-N anchor point (cycles)

    Returns:
        dict with generic keys (prefixed by caller): sigma_max, sigma_min,
        sigma_m, sigma_a, tau_m, tau_a, R, sigma_a_eq, sigma_m_eq, S_e,
        n_f, n_y, n, b_B, N_f, t_f, D, failed_miner.
    """
    # --- Section 9: Stress cycling ---
    s_max, s_min, sigma_m, sigma_a, R = _stress_cycle(sigma_history)
    t_max, t_min, tau_m, tau_a, _     = _stress_cycle(tau_history)

    # Von Mises equivalent alternating and mean stresses (Eqs 9.6-9.7)
    sigma_a_eq = math.sqrt(sigma_a**2 + 3.0 * tau_a**2)
    sigma_m_eq = math.sqrt(sigma_m**2 + 3.0 * tau_m**2)

    # --- Section 12.1: Modified Goodman fatigue safety factor (Eq 12.2) ---
    # n_f = 1 / (sigma_a_eq/S_e + sigma_m_eq/S_ut)
    denom_goodman = sigma_a_eq / S_e + sigma_m_eq / S_ut
    n_f = 1.0 / denom_goodman if denom_goodman > 0.0 else float('inf')

    # --- Section 12.2: ECY safety factor (Eq 12.4) ---
    # n_y = S_y / (sigma_a_eq + sigma_m_eq)
    denom_ecy = sigma_a_eq + sigma_m_eq
    n_y = S_y / denom_ecy if denom_ecy > 0.0 else float('inf')

    # --- Section 12.3: Governing safety factor (Eq 12.5) ---
    n = min(n_f, n_y)

    # --- Section 13: Basquin / S-N finite life ---
    b_B = _basquin_exponent(S_e, S_ut, N_anchor)
    N_f = _cycles_to_failure(sigma_a_eq, S_e, S_ut, sigma_f_prime, N_anchor)
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
        'S_e':          S_e,
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
        'S_ut'          — ultimate tensile strength (Pa)
        'S_y'           — yield strength (Pa)
        'S_prime_e'     — reference fatigue strength at 10^8 cycles (Pa)
        'sigma_f_prime' — fatigue strength coefficient (Pa)
        'n_rpm'         — crank speed (RPM)
        'total_cycles'  — design life in cycles

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
                    't_f_rod', 'D_rod', 'failed_miner_rod', 'S_e_rod',
                    'sigma_a_eq_rod', 'sigma_m_eq_rod', 'b_B_rod', ...
            Crank : same pattern with '_crank' suffix
            Pin   : same pattern with '_pin' suffix
    """
    # --- Material properties (from design dict; fallback to 2024-T3 Al defaults) ---
    _ctx = 'fatigue.evaluate'
    S_ut          = float(get_or_warn(design, 'S_ut',          483e6,  context=_ctx))
    S_y           = float(get_or_warn(design, 'S_y',           345e6,  context=_ctx))
    S_prime_e     = float(get_or_warn(design, 'S_prime_e',     130e6,  context=_ctx))
    sigma_f_prime = float(get_or_warn(design, 'sigma_f_prime', 807e6,  context=_ctx))
    n_rpm         = float(get_or_warn(design, 'n_rpm',         30.0,   context=_ctx))
    total_cycles_raw = get_or_warn(design, 'total_cycles', None, context=_ctx)
    total_cycles: Optional[float] = (
        float(total_cycles_raw) if total_cycles_raw is not None else None
    )

    # Configurable fatigue constants (from baseline.yaml via design dict)
    N_anchor = float(get_or_warn(design, 'N_basquin_anchor', _N_BASQUIN_ANCHOR_DEFAULT, context=_ctx))
    z_a      = float(get_or_warn(design, 'z_a_reliability',  _Z_A_RELIABILITY_DEFAULT,  context=_ctx))

    # --- Link dimensions ---
    w_rod   = float(design['width_l'])
    t_rod   = float(design['thickness_l'])
    w_crank = float(design['width_r'])
    t_crank = float(design['thickness_r'])
    D_pA    = float(design['pin_diameter_A'])
    D_pB    = float(design['pin_diameter_B'])
    D_pC    = float(design['pin_diameter_C'])

    # --- Corrected endurance limits (Section 10) ---
    S_e_rod   = _endurance_limit_rect(w_rod,   t_rod,   S_ut, S_prime_e, z_a)
    S_e_crank = _endurance_limit_rect(w_crank, t_crank, S_ut, S_prime_e, z_a)
    # For pins: use smallest pin diameter — most conservative k_b (size factor)
    d_pin_min = min(D_pA, D_pB, D_pC)
    S_e_pin   = _endurance_limit_pin(d_pin_min, S_ut, S_prime_e, z_a)

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
        S_e_rod,   S_ut, S_y, sigma_f_prime, n_rpm, total_cycles, N_anchor,
    )
    crank = _component_fatigue(
        arr_sig_crank, arr_tau_crank,
        S_e_crank, S_ut, S_y, sigma_f_prime, n_rpm, total_cycles, N_anchor,
    )
    pin   = _component_fatigue(
        arr_sig_pin,   arr_tau_pin,
        S_e_pin,   S_ut, S_y, sigma_f_prime, n_rpm, total_cycles, N_anchor,
    )

    # --- Build output dict with component suffixes ---
    result: Dict[str, Any] = {}
    for suffix, comp in (('rod', rod), ('crank', crank), ('pin', pin)):
        for key, val in comp.items():
            result[f'{key}_{suffix}'] = val

    return result
