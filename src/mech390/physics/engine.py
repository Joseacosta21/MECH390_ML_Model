"""
Physics engine for Offset Crank-Slider Mechanism.
Orchestrates the 15-degree sweep, evaluates kinematics/dynamics/stresses
at each crank angle, collects per-component stress histories, then runs
buckling and fatigue analyses over the full cycle.

Ref: instructions.md (Authoritative), Mother Doc v7
"""

import math
from typing import Any, Dict, List

import numpy as np

from mech390.physics import (
    buckling,
    dynamics,
    fatigue,
    kinematics,
    mass_properties,
    stresses,
)


def evaluate_design(design: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a 3D mechanism design over one full crank revolution (15° steps).

    At each crank angle the engine:
      1. Evaluates kinematics (position, velocity, acceleration)
      2. Evaluates mass-property COG positions
      3. Solves dynamics (joint reaction forces via Newton-Euler)
      4. Injects per-step values into a design copy:
           design['theta'] = theta          (current angle, rad)
           design['tau_A'] = forces['tau_A'] (motor torque at this step, N*m)
      5. Calls stresses.evaluate() → per-component stress dict
      6. Appends component stresses to per-component history lists
      7. Computes signed F_r,rod,B independently (for buckling history)

    After the sweep:
      - Calls buckling.evaluate(F_r_rod_history, design)
      - Calls fatigue.evaluate(per-component histories, design)

    Material properties and operating parameters must be present in the
    design dict before calling this function. generate.py injects them
    from baseline.yaml (keys: 'E', 'S_ut', 'S_y', 'S_prime_e',
    'sigma_f_prime', 'n_rpm', 'total_cycles').

    Args:
        design (dict): Must contain at least:
            'r', 'l', 'e'                          — 2D kinematics (m)
            'omega'                                 — angular velocity (rad/s)
            'mass_crank', 'mass_rod', 'mass_slider' — link masses (kg)
            'width_r', 'thickness_r'               — crank cross-section (m)
            'width_l', 'thickness_l'               — rod cross-section (m)
            'pin_diameter_A/B/C'                   — pin diameters (m)
            'I_area_crank_yy/zz', 'I_area_rod_yy/zz' — area moments (m^4)
            'I_mass_crank_cg_z', 'I_mass_rod_cg_z'   — mass MOI (kg*m^2)
            'E', 'S_ut', 'S_y', 'S_prime_e',
            'sigma_f_prime', 'n_rpm', 'total_cycles'  — material / fatigue props

    Returns:
        dict with keys:
            'sigma_max'        : float — peak overall normal stress [Pa]
            'tau_max'          : float — peak overall shear stress [Pa]
            'theta_sigma_max'  : float — crank angle at sigma_max [rad]
            'theta_tau_max'    : float — crank angle at tau_max [rad]
            'valid_physics'    : bool
            'n_buck'           : float — buckling safety factor
            'P_cr'             : float — Euler critical load (N)
            'N_max_comp'       : float — max compressive axial force (N)
            'buckling_passed'  : bool
            + all keys from fatigue.evaluate() (suffixed _rod, _crank, _pin)
    """
    r      = design['r']
    l      = design['l']
    e      = design['e']
    omega  = design.get('omega', 1.0)

    mass_crank  = design.get('mass_crank',  1.0)
    mass_rod    = design.get('mass_rod',    1.0)
    mass_slider = design.get('mass_slider', 1.0)
    i_crank     = design.get('I_mass_crank_cg_z', 1.0)
    i_rod       = design.get('I_mass_rod_cg_z',   1.0)
    mu          = design.get('mu',    0.0)
    g           = design.get('g',    9.81)
    alpha_r     = design.get('alpha_r', 0.0)

    # 15-degree sweep over one full revolution
    thetas = np.deg2rad(np.arange(0, 360, 15))

    sigma_max      = 0.0
    tau_max        = 0.0
    theta_sigma_max = 0.0
    theta_tau_max   = 0.0

    # Per-component stress histories (one entry per sweep step)
    sigma_rod_hist:   List[float] = []
    tau_rod_hist:     List[float] = []
    sigma_crank_hist: List[float] = []
    tau_crank_hist:   List[float] = []
    sigma_pin_hist:   List[float] = []
    tau_pin_hist:     List[float] = []

    # Signed F_r,rod,B history for buckling check (negative = compression)
    F_r_rod_hist: List[float] = []

    try:
        for theta in thetas:
            # --- Kinematics ---
            kinematics.slider_position(theta, r, l, e)
            kinematics.slider_velocity(theta, omega, r, l, e)
            kinematics.slider_acceleration(theta, omega, r, l, e)
            kinematics.crank_pin_position(theta, r)
            kinematics.crank_pin_velocity(theta, omega, r)
            kinematics.crank_pin_acceleration(theta, omega, r)

            # --- Mass properties (COGs) ---
            mass_properties.crank_cog(theta, r)
            mass_properties.rod_cog(theta, r, l, e)
            mass_properties.slider_cog(theta, r, l, e)

            # --- Dynamics ---
            forces = dynamics.joint_reaction_forces(
                theta, omega, r, l, e,
                mass_crank, mass_rod, mass_slider,
                I_crank=i_crank,
                I_rod=i_rod,
                mu=mu,
                g=g,
                alpha_r=alpha_r,
            )
            F_A = forces['F_A']
            F_B = forces['F_B']
            F_C = forces['F_C']

            # --- Inject per-step values into a shallow design copy ---
            step_design = design.copy()
            step_design['theta'] = float(theta)
            step_design['tau_A'] = float(forces['tau_A'])

            # --- Stresses (returns dict with per-component + overall values) ---
            stress_result = stresses.evaluate(step_design, F_A, F_B, F_C)

            sigma = stress_result['sigma']
            tau   = stress_result['tau']

            # --- Collect per-component histories ---
            sigma_rod_hist.append(stress_result['sigma_rod'])
            tau_rod_hist.append(stress_result['tau_rod'])
            sigma_crank_hist.append(stress_result['sigma_crank'])
            tau_crank_hist.append(stress_result['tau_crank'])
            sigma_pin_hist.append(stress_result['sigma_pin'])
            tau_pin_hist.append(stress_result['tau_pin'])

            # --- Signed F_r,rod,B for buckling (Option A: independent recompute) ---
            # Sign convention: F_B is force on crank; rod sees -F_B at pin B.
            # Negative result means rod is in compression.
            phi = kinematics.rod_angle(theta, r, l, e)
            F_r_rod_B = -F_B[0] * math.cos(phi) - F_B[1] * math.sin(phi)
            F_r_rod_hist.append(F_r_rod_B)

            # --- Track overall sweep maxima ---
            if sigma > sigma_max:
                sigma_max       = sigma
                theta_sigma_max = float(theta)
            if tau > tau_max:
                tau_max       = tau
                theta_tau_max = float(theta)

    except Exception:
        return {'valid_physics': False}

    # --- Post-sweep: Buckling check (Section 14) ---
    buckling_result = buckling.evaluate(F_r_rod_hist, design)

    # --- Post-sweep: Fatigue analysis (Sections 9-13) ---
    fatigue_result = fatigue.evaluate(
        sigma_rod_hist,   tau_rod_hist,
        sigma_crank_hist, tau_crank_hist,
        sigma_pin_hist,   tau_pin_hist,
        design,
    )

    result: Dict[str, Any] = {
        'sigma_max':       sigma_max,
        'tau_max':         tau_max,
        'theta_sigma_max': theta_sigma_max,
        'theta_tau_max':   theta_tau_max,
        'valid_physics':   True,
        # Buckling
        'n_buck':          buckling_result['n_buck'],
        'P_cr':            buckling_result['P_cr'],
        'N_max_comp':      buckling_result['N_max_comp'],
        'buckling_passed': buckling_result['passed'],
    }
    result.update(fatigue_result)

    return result
