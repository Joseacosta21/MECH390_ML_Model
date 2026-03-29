"""
Buckling analysis module for Offset Crank-Slider Mechanism.

Implements Section 14 of the Mother Doc:
  - Euler buckling check for the connecting rod under compression
  - Only applicable at crank angles where F_r,rod,B < 0 (rod in compression)

The connecting rod uses pin-pin boundary conditions (K_c = 1.0) and
buckles about its weak axis (y-axis, since w_rod > t_rod).

All units are SI throughout (Pa, m, N).

Elastic modulus E is read from the design dict (key 'E'), injected by
generate.py from baseline.yaml material properties. Falls back to the
2024-T3 aluminium value (73.1 GPa) if not present.

Ref: Mother Doc v7 Section 14, instructions.md
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

# ---------------------------------------------------------------------------
# Buckling constants — Mother Doc Section 14
# ---------------------------------------------------------------------------
_K_C: float = 1.0          # Effective length factor — pin-pin ends (Eq 14.3)
_N_BUCK_TARGET: float = 3.0  # Target buckling safety factor for machinery (Eq 14.5)
_E_DEFAULT: float = 73.1e9  # 2024-T3 Al elastic modulus fallback (Pa)


def evaluate(
    F_r_rod_history: Sequence[float],
    design: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Perform Euler buckling check for the connecting rod over one crank
    revolution.

    Called by engine.py after the full angle sweep, passing the signed
    F_r,rod,B axial force array computed independently at each sweep step
    (Option A: engine recomputes the signed value without abs()).

    The signed convention follows the codebase dynamics (F_B = force on crank;
    rod sees -F_B). Negative F_r,rod,B means the rod is in compression.

    Args:
        F_r_rod_history : sequence of signed F_r,rod,B axial force values at
                          each crank angle step (N). Negative = compression
                          (Mother Doc Eq 2.9 convention as implemented in
                          stresses._rod_frame_forces).
        design          : design parameter dict. Must include:
                            'width_l'     — rod width w_rod (m)
                            'thickness_l' — rod thickness t_rod (m)
                            'l'           — rod length l_rod (m)
                          Optional:
                            'E'           — elastic modulus (Pa); defaults to
                                            73.1e9 (2024-T3 Al) if absent

    Returns:
        dict with keys:
          'I_min_r'        : float — second moment of area about weak axis (m^4)
          'P_cr'           : float — Euler critical buckling load (N)
          'N_max_comp'     : float — maximum compressive axial force (N, >= 0)
          'n_buck'         : float — buckling safety factor P_cr / N_max_comp;
                                     inf if no compressive loading found
          'passed'         : bool  — True if n_buck >= 3.0
          'has_compression': bool  — True if any angle has F_r,rod,B < 0
    """
    w = float(design['width_l'])       # rod width (in-plane, longer dimension)
    t = float(design['thickness_l'])   # rod thickness (out-of-plane, shorter)
    L = float(design['l'])             # rod length (centre distance)
    E = float(design.get('E', _E_DEFAULT))

    # --- Eq 14.2: Second moment of area about weak axis ---
    # I_min,r = I_yr = w_rod * t_rod^3 / 12
    # Buckling occurs about y-axis since w_rod > t_rod (t is the smaller dim)
    I_min_r = w * t**3 / 12.0

    # --- Eq 14.1: Euler critical buckling load ---
    # P_cr = pi^2 * E * I_min / (K_c * l_rod)^2
    P_cr = (math.pi**2 * E * I_min_r) / (_K_C * L)**2

    # --- Eq 14.4: Maximum compressive axial force ---
    # F_r,rod,B < 0 indicates compression (see module and _rod_frame_forces docs)
    compressive_forces = [abs(f) for f in F_r_rod_history if f < 0.0]
    has_compression = len(compressive_forces) > 0
    N_max_comp = max(compressive_forces) if has_compression else 0.0

    # --- Eq 14.5: Buckling safety factor ---
    # n_buck = P_cr / N_max_comp  (target >= 3.0 for machinery)
    # Returns inf if no compressive loading exists in the sweep.
    if N_max_comp <= 0.0:
        n_buck = float('inf')
    else:
        n_buck = P_cr / N_max_comp

    return {
        'I_min_r':         I_min_r,
        'P_cr':            P_cr,
        'N_max_comp':      N_max_comp,
        'n_buck':          n_buck,
        'passed':          n_buck >= _N_BUCK_TARGET,
        'has_compression': has_compression,
    }
