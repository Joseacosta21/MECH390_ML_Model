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
from mech390.physics._utils import get_or_warn


def evaluate_design(design: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a 3D mechanism design over one full crank revolution (15° steps).

    At each crank angle the engine:
      1. Evaluates kinematics (position, velocity, acceleration for all bodies)
      2. Solves dynamics (joint reaction forces via Newton-Euler)
      3. Injects per-step values into a design copy:
           design['theta'] = theta           (current angle, rad)
           design['tau_A'] = forces['tau_A'] (motor torque at this step, N*m)
      4. Calls stresses.evaluate() → per-component stress dict
      5. Appends all per-step outputs to their respective history lists
      6. Computes signed F_r,rod,B independently (for buckling history)

    After the sweep:
      - Calls buckling.evaluate(F_r_rod_history, design)
      - Calls fatigue.evaluate(per-component histories, design)

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
            'E', 'S_ut', 'S_y', 'Sn',
            'sigma_f_prime', 'n_rpm', 'total_cycles'  — material / fatigue props

    Returns:
        dict with keys:
            'sigma_max'          : float — peak overall normal stress [Pa]
            'tau_max'            : float — peak overall shear stress [Pa]
            'theta_sigma_max'    : float — crank angle at sigma_max [rad]
            'theta_tau_max'      : float — crank angle at tau_max [rad]
            'valid_physics'      : bool
            'n_buck'             : float — buckling safety factor
            'P_cr'               : float — Euler critical load (N)
            'N_max_comp'         : float — max compressive axial force (N)
            'buckling_passed'    : bool
            'kinematics_history' : list of dicts, one per angle step
            'dynamics_history'   : list of dicts, one per angle step
            'stresses_history'   : list of dicts, one per angle step
            + all keys from fatigue.evaluate() (suffixed _rod, _crank, _pin)
    """
    _ctx = 'engine.evaluate_design'
    r      = design['r']
    l      = design['l']
    e      = design['e']
    omega  = get_or_warn(design, 'omega',  1.0,  context=_ctx)

    mass_crank  = get_or_warn(design, 'mass_crank',  1.0, context=_ctx)
    mass_rod    = get_or_warn(design, 'mass_rod',    1.0, context=_ctx)
    mass_slider = get_or_warn(design, 'mass_slider', 1.0, context=_ctx)
    i_crank     = get_or_warn(design, 'I_mass_crank_cg_z', 1.0, context=_ctx)
    i_rod       = get_or_warn(design, 'I_mass_rod_cg_z',   1.0, context=_ctx)
    mu          = get_or_warn(design, 'mu',      0.0,  context=_ctx)
    g           = get_or_warn(design, 'g',       9.81, context=_ctx)
    alpha_r     = get_or_warn(design, 'alpha_r', 0.0,  context=_ctx)
    m_block     = get_or_warn(design, 'm_block', 0.0,  context=_ctx)

    # 15-degree sweep over one full revolution
    thetas = np.deg2rad(np.arange(0, 360, 15))

    sigma_max       = 0.0
    tau_max         = 0.0
    theta_sigma_max = 0.0
    theta_tau_max   = 0.0

    # Per-step output histories returned to the caller for CSV export
    kinematics_history: List[Dict[str, Any]] = []
    dynamics_history:   List[Dict[str, Any]] = []
    stresses_history:   List[Dict[str, Any]] = []

    # Per-component stress histories consumed by fatigue.evaluate()
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
            angle_deg = float(np.rad2deg(theta))

            # --- Kinematics (all outputs captured) ---
            pos_C    = kinematics.slider_position(theta, r, l, e)
            vel_C    = kinematics.slider_velocity(theta, omega, r, l, e)
            acc_C    = kinematics.slider_acceleration(theta, omega, r, l, e)
            pos_B    = kinematics.crank_pin_position(theta, r)
            vel_B    = kinematics.crank_pin_velocity(theta, omega, r)
            acc_B    = kinematics.crank_pin_acceleration(theta, omega, r)
            phi      = kinematics.rod_angle(theta, r, l, e)
            omega_cb = kinematics.rod_angular_velocity(theta, omega, r, l, e)
            alpha_cb = kinematics.rod_angular_acceleration(theta, omega, r, l, e)

            kinematics_history.append({
                'angle_deg': angle_deg,
                'x_C':       float(pos_C[0]),
                'v_Cx':      float(vel_C[0]),
                'a_Cx':      float(acc_C[0]),
                'pos_Bx':    float(pos_B[0]),
                'pos_By':    float(pos_B[1]),
                'vel_Bx':    float(vel_B[0]),
                'vel_By':    float(vel_B[1]),
                'acc_Bx':    float(acc_B[0]),
                'acc_By':    float(acc_B[1]),
                'phi_rad':   float(phi),
                'omega_cb':  float(omega_cb),
                'alpha_cb':  float(alpha_cb),
            })

            # --- Dynamics ---
            forces = dynamics.joint_reaction_forces(
                theta, omega, r, l, e,
                mass_crank, mass_rod, mass_slider,
                I_crank=i_crank,
                I_rod=i_rod,
                mu=mu,
                g=g,
                alpha_r=alpha_r,
                m_block=m_block,
            )
            F_A = forces['F_A']
            F_B = forces['F_B']
            F_C = forces['F_C']

            dynamics_history.append({
                'angle_deg': angle_deg,
                'F_Ax':      float(F_A[0]),
                'F_Ay':      float(F_A[1]),
                'F_Bx':      float(F_B[0]),
                'F_By':      float(F_B[1]),
                'F_Cx':      float(F_C[0]),
                'F_Cy':      float(F_C[1]),
                'N':         float(forces['N']),
                'F_f':       float(forces['F_f']),
                'tau_A':     float(forces['tau_A']),
            })

            # --- Inject per-step values into a shallow design copy ---
            step_design = design.copy()
            step_design['theta'] = float(theta)
            step_design['tau_A'] = float(forces['tau_A'])

            # --- Stresses ---
            stress_result = stresses.evaluate(step_design, F_A, F_B, F_C)

            sigma = stress_result['sigma']
            tau   = stress_result['tau']

            stresses_history.append({
                'angle_deg':   angle_deg,
                'sigma_rod':   stress_result['sigma_rod'],
                'tau_rod':     stress_result['tau_rod'],
                'sigma_crank': stress_result['sigma_crank'],
                'tau_crank':   stress_result['tau_crank'],
                'sigma_pin':   stress_result['sigma_pin'],
                'tau_pin':     stress_result['tau_pin'],
                'sigma':       sigma,
                'tau':         tau,
            })

            # --- Collect per-component histories for fatigue ---
            sigma_rod_hist.append(stress_result['sigma_rod'])
            tau_rod_hist.append(stress_result['tau_rod'])
            sigma_crank_hist.append(stress_result['sigma_crank'])
            tau_crank_hist.append(stress_result['tau_crank'])
            sigma_pin_hist.append(stress_result['sigma_pin'])
            tau_pin_hist.append(stress_result['tau_pin'])

            # --- Signed F_r,rod,B for buckling (negative = compression) ---
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

    # --- Post-sweep: Motor torque, energy per revolution, peak joint forces,
    #     and per-link peak normal stresses (used by generate.py for n_static) ---
    tau_A_arr   = np.array([s['tau_A'] for s in dynamics_history], dtype=float)
    delta_theta = 2.0 * np.pi / len(tau_A_arr)

    tau_A_max = float(np.max(np.abs(tau_A_arr)))
    E_rev     = float(np.sum(tau_A_arr) * delta_theta)   # [J] per revolution

    F_A_max = float(np.max(np.hypot(
        [s['F_Ax'] for s in dynamics_history],
        [s['F_Ay'] for s in dynamics_history],
    )))
    F_B_max = float(np.max(np.hypot(
        [s['F_Bx'] for s in dynamics_history],
        [s['F_By'] for s in dynamics_history],
    )))
    F_C_max = float(np.max(np.abs([s['F_Cx'] for s in dynamics_history])))

    sigma_rod_peak   = float(np.max(np.abs([s['sigma_rod']   for s in stresses_history])))
    sigma_crank_peak = float(np.max(np.abs([s['sigma_crank'] for s in stresses_history])))
    sigma_pin_peak   = float(np.max(np.abs([s['sigma_pin']   for s in stresses_history])))

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
        'sigma_max':         sigma_max,
        'tau_max':           tau_max,
        'theta_sigma_max':   theta_sigma_max,
        'theta_tau_max':     theta_tau_max,
        'valid_physics':     True,
        # Buckling
        'n_buck':            buckling_result['n_buck'],
        'P_cr':              buckling_result['P_cr'],
        'N_max_comp':        buckling_result['N_max_comp'],
        'buckling_passed':   buckling_result['passed'],
        # Motor torque & energy per revolution
        'tau_A_max':         tau_A_max,
        'E_rev':             E_rev,
        # Peak joint forces (for bearing selection)
        'F_A_max':           F_A_max,
        'F_B_max':           F_B_max,
        'F_C_max':           F_C_max,
        # Per-link peak normal stresses (generate.py divides by sigma_limit → n_static)
        'sigma_rod_peak':    sigma_rod_peak,
        'sigma_crank_peak':  sigma_crank_peak,
        'sigma_pin_peak':    sigma_pin_peak,
        # Per-angle histories for CSV export
        'kinematics_history': kinematics_history,
        'dynamics_history':   dynamics_history,
        'stresses_history':   stresses_history,
    }
    result.update(fatigue_result)

    return result
