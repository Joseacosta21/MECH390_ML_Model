"""
Euler buckling check for the connecting rod in the offset crank-slider mechanism.

Checks pin-pin column buckling at crank angles where the rod is in compression.
All units are SI (Pa, m, N).

Elastic modulus E is read from the design dict (key 'E'), injected from
baseline.yaml. Falls back to 73.1 GPa (2024-T3 aluminium) if absent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from mech390.physics._utils import get_or_warn

### Buckling constants
_K_C: float = 1.0                   # pin-pin boundary conditions
_N_BUCK_TARGET_DEFAULT: float = 3.0  # default from baseline.yaml stress_analysis.n_buck_target
_E_DEFAULT: float = 73.1e9          # 2024-T3 Al elastic modulus fallback (Pa)


# second moment of area about the weak (minimum) axis for a rectangular section
# returns min(w*t^3, t*w^3) / 12
def I_weak_axis(w: float, t: float) -> float:
    return min(w * t**3, t * w**3) / 12.0


# Euler critical buckling load for a pin-pin rectangular rod
# P_cr = pi^2 * E * I_min / (K_c * l)^2, with K_c = 1.0
def critical_load(w: float, t: float, l: float, E: float = _E_DEFAULT) -> float:
    return (math.pi**2 * E * I_weak_axis(w, t)) / (_K_C * l)**2


def evaluate(
    F_r_rod_history: Sequence[float],
    design: Dict[str, Any],
) -> Dict[str, Any]:
    """Euler buckling check over one crank revolution. Returns n_buck and related outputs."""
    _ctx = 'buckling.evaluate'
    try:
        w = float(design['width_l'])       # rod width (in-plane, longer dimension)
        t = float(design['thickness_l'])   # rod thickness (out-of-plane, shorter)
        L = float(design['l'])             # rod length (centre distance)
    except KeyError as exc:
        raise KeyError(
            f"buckling.evaluate: required key {exc} missing from design dict"
        ) from exc
    E = float(get_or_warn(design, 'E', _E_DEFAULT, context=_ctx))
    n_buck_target = float(get_or_warn(design, 'n_buck_target', _N_BUCK_TARGET_DEFAULT, context=_ctx))

    # weak-axis I and critical load
    I_min_r = I_weak_axis(w, t)
    P_cr    = critical_load(w, t, L, E)

    # maximum compressive axial force (F_r,rod,B < 0 means compression)
    compressive_forces = [abs(f) for f in F_r_rod_history if f < 0.0]
    has_compression = len(compressive_forces) > 0
    N_max_comp = max(compressive_forces) if has_compression else 0.0

    # n_buck = P_cr / N_max_comp (target >= 3.0); inf if no compression in the sweep
    if N_max_comp <= 0.0:
        n_buck = float('inf')
    else:
        n_buck = P_cr / N_max_comp

    return {
        'I_min_r':         I_min_r,
        'P_cr':            P_cr,
        'N_max_comp':      N_max_comp,
        'n_buck':          n_buck,
        'passed':          n_buck >= n_buck_target,
        'has_compression': has_compression,
    }
