"""
Pass/fail validation logic for evaluated crank-slider designs.

Separated from generate.py so that validate_candidate.py, tests, and any
future tooling can import the business rules without pulling in the full
dataset generation pipeline.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compute_checks(
    metrics:       Dict[str, Any],
    sigma_limit:   float,
    tau_limit:     float,
    check_limits:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute all pass/fail checks and the overall pass_fail label.

    Args:
        metrics:      Output dict from engine.evaluate_design().
        sigma_limit:  Allowable normal stress (S_y / safety_factor) [Pa].
        tau_limit:    Allowable shear stress (0.577·S_y / safety_factor) [Pa].
        check_limits: Dict of limit values read from baseline.yaml limits section.
                      Required keys: utilization_max, n_buck_min, n_shaft_min,
                      n_static_{rod,crank,pin}_min, n_fatigue_{rod,crank,pin}_min,
                      D_miner_{rod,crank,pin}_max.

    Returns:
        Flat dict of check columns to be merged into the per-design row.
        Always contains 'pass_fail' (1 = pass, 0 = fail).
    """
    checks: Dict[str, Any] = {}

    # --- Static stress ---
    s_max = metrics.get('sigma_max', 0.0)
    t_max = metrics.get('tau_max',   0.0)
    u_sigma = s_max / sigma_limit if sigma_limit > 0 else 0.0
    u_tau   = t_max / tau_limit   if tau_limit   > 0 else 0.0
    utilization = max(u_sigma, u_tau)
    utilization_max = float(check_limits['utilization_max'])
    checks['utilization']         = utilization
    checks['utilization_max']     = utilization_max
    checks['check_static_passed'] = bool(utilization <= utilization_max)

    # Static FoS per component from peak normal stresses
    for comp, peak_key in (
        ('rod',   'sigma_rod_peak'),
        ('crank', 'sigma_crank_peak'),
        ('pin',   'sigma_pin_peak'),
    ):
        peak = float(metrics.get(peak_key, 0.0) or 0.0)
        n_static = sigma_limit / peak if peak > 0.0 else float('inf')
        n_static_min = float(check_limits[f'n_static_{comp}_min'])
        checks[f'n_static_{comp}']               = n_static
        checks[f'n_static_{comp}_min']           = n_static_min
        checks[f'check_static_sf_{comp}_passed'] = bool(
            n_static >= n_static_min if np.isfinite(n_static) else True
        )

    # --- Shaft A (Mott 12-24 ASME-Elliptic) ---
    n_shaft     = metrics.get('n_shaft', float('inf'))
    n_shaft_min = float(check_limits['n_shaft_min'])
    checks['n_shaft']            = n_shaft
    checks['n_shaft_min']        = n_shaft_min
    checks['check_shaft_passed'] = bool(
        n_shaft >= n_shaft_min if np.isfinite(n_shaft) else True
    )

    # --- Buckling ---
    n_buck     = metrics.get('n_buck', float('inf'))
    n_buck_min = float(check_limits['n_buck_min'])
    checks['n_buck']               = n_buck
    checks['n_buck_min']           = n_buck_min
    checks['P_cr']                 = metrics.get('P_cr',       float('nan'))
    checks['N_max_comp']           = metrics.get('N_max_comp', float('nan'))
    checks['check_buckling_passed'] = bool(
        n_buck >= n_buck_min if np.isfinite(n_buck) else True
    )

    # --- Fatigue Goodman+ECY governing safety factor ---
    for comp in ('rod', 'crank', 'pin'):
        n = metrics.get(f'n_{comp}', float('inf'))
        checks[f'n_{comp}']              = n
        checks[f'S_n_prime_{comp}']      = metrics.get(f'S_n_prime_{comp}',  float('nan'))
        checks[f'sigma_a_eq_{comp}']     = metrics.get(f'sigma_a_eq_{comp}', float('nan'))
        checks[f'sigma_m_eq_{comp}']     = metrics.get(f'sigma_m_eq_{comp}', float('nan'))
        checks[f'n_f_{comp}']            = metrics.get(f'n_f_{comp}',        float('nan'))
        checks[f'n_y_{comp}']            = metrics.get(f'n_y_{comp}',        float('nan'))
        checks[f'N_f_{comp}']            = metrics.get(f'N_f_{comp}',        float('nan'))
        checks[f't_f_{comp}']            = metrics.get(f't_f_{comp}',        float('nan'))
        n_fatigue_min = float(check_limits[f'n_fatigue_{comp}_min'])
        checks[f'n_fatigue_{comp}_min']       = n_fatigue_min
        checks[f'check_fatigue_{comp}_passed'] = bool(
            n >= n_fatigue_min if np.isfinite(n) else True
        )

    # --- Miner's rule damage ---
    for comp in ('rod', 'crank', 'pin'):
        D           = metrics.get(f'D_{comp}', None)
        d_miner_max = float(check_limits[f'D_miner_{comp}_max'])
        checks[f'D_{comp}']               = D
        checks[f'D_miner_{comp}_max']     = d_miner_max
        if D is not None:
            checks[f'check_miner_{comp}_passed'] = bool(D < d_miner_max)
        else:
            checks[f'check_miner_{comp}_passed'] = True  # not computed → not failed

    # --- sigma/tau sweep summary ---
    checks['sigma_max']       = metrics.get('sigma_max',       float('nan'))
    checks['tau_max']         = metrics.get('tau_max',         float('nan'))
    checks['theta_sigma_max'] = metrics.get('theta_sigma_max', float('nan'))
    checks['theta_tau_max']   = metrics.get('theta_tau_max',   float('nan'))

    # --- Overall pass/fail ---
    bool_checks = [
        checks['check_static_passed'],
        checks['check_static_sf_rod_passed'],
        checks['check_static_sf_crank_passed'],
        checks['check_static_sf_pin_passed'],
        checks['check_shaft_passed'],
        checks['check_buckling_passed'],
        checks['check_fatigue_rod_passed'],
        checks['check_fatigue_crank_passed'],
        checks['check_fatigue_pin_passed'],
        checks['check_miner_rod_passed'],
        checks['check_miner_crank_passed'],
        checks['check_miner_pin_passed'],
    ]
    checks['pass_fail'] = 1 if all(bool_checks) else 0

    return checks
