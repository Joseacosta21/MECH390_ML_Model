"""
Physics engine for the offset crank-slider mechanism.
Orchestrates the 15-degree sweep, evaluates kinematics/dynamics/stresses
at each crank angle, collects per-component stress histories, then runs
buckling and fatigue analyses over the full cycle.
"""

import logging
import math
import traceback
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

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
    Runs the full physics sweep for one design over 360 degrees at 15-degree steps.

    At each crank angle: evaluates kinematics, solves dynamics, runs stress analysis.
    After the sweep: runs buckling and fatigue analyses over the full cycle.

    design must contain 'r', 'l', 'e', 'omega', link masses and inertias,
    cross-section dimensions, material properties (E, S_ut, S_y, Sn, Basquin
    coefficients), and fatigue config (n_rpm, total_cycles).
    Returns {'valid_physics': False} on any failure during the sweep.
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
    alpha_r        = get_or_warn(design, 'alpha_r',        0.0,  context=_ctx)
    m_block        = get_or_warn(design, 'm_block',        0.0,  context=_ctx)
    sweep_step_deg = get_or_warn(design, 'sweep_step_deg', 15.0, context=_ctx)

    # sweep over one full revolution at the configured step size
    thetas = np.deg2rad(np.arange(0, 360, float(sweep_step_deg)))

    sigma_max       = 0.0
    tau_max         = 0.0
    theta_sigma_max = 0.0
    theta_tau_max   = 0.0

    # per-step output histories returned to the caller for CSV export
    kinematics_history: List[Dict[str, Any]] = []
    dynamics_history:   List[Dict[str, Any]] = []
    stresses_history:   List[Dict[str, Any]] = []

    # per-component stress histories consumed by fatigue.evaluate()
    sigma_rod_hist:   List[float] = []
    tau_rod_hist:     List[float] = []
    sigma_crank_hist: List[float] = []
    tau_crank_hist:   List[float] = []
    sigma_pin_hist:   List[float] = []
    tau_pin_hist:     List[float] = []

    # signed F_r,rod,B history for buckling check (negative = compression)
    F_r_rod_hist: List[float] = []

    # shaft A |F_A| and |tau_A| histories for ASME-elliptic check
    F_A_mag_hist:  List[float] = []
    tau_A_abs_hist: List[float] = []

    try:
        for theta in thetas:
            angle_deg = float(np.rad2deg(theta))

            # kinematics
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

            # dynamics
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

            # inject per-step values into a shallow design copy for stress evaluation
            step_design = design.copy()
            step_design['theta'] = float(theta)
            step_design['tau_A'] = float(forces['tau_A'])

            # stresses
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

            # collect per-component histories for fatigue
            sigma_rod_hist.append(stress_result['sigma_rod'])
            tau_rod_hist.append(stress_result['tau_rod'])
            sigma_crank_hist.append(stress_result['sigma_crank'])
            tau_crank_hist.append(stress_result['tau_crank'])
            sigma_pin_hist.append(stress_result['sigma_pin'])
            tau_pin_hist.append(stress_result['tau_pin'])

            # signed F_r,rod,B for buckling (negative = compression)
            F_r_rod_B = -F_B[0] * math.cos(phi) - F_B[1] * math.sin(phi)
            F_r_rod_hist.append(F_r_rod_B)

            # shaft A histories for ASME-elliptic check
            F_A_mag_hist.append(float(np.linalg.norm(F_A)))
            tau_A_abs_hist.append(abs(float(forces['tau_A'])))

            # track overall sweep maxima
            if sigma > sigma_max:
                sigma_max       = sigma
                theta_sigma_max = float(theta)
            if tau > tau_max:
                tau_max       = tau
                theta_tau_max = float(theta)

    except Exception as exc:
        logger.warning(
            "evaluate_design failed during sweep: %s\n%s",
            exc, traceback.format_exc(),
        )
        return {'valid_physics': False}

    # post-sweep: motor torque, energy per revolution, peak joint forces,
    # and per-link peak normal stresses (used by generate.py for n_static)
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

    # shaft A ASME-elliptic safety factor
    # Press-fit connection; Kf = 1.0 (no stress concentration from keyway or spline).
    # Bending: fully reversed (M = |F_A| * L_bearing at each step).
    # Torsion: steady (T = |tau_A| at each step).
    # N_shaft = pi*d^3 / [16 * sqrt((2M/Sn_prime)^2 + 3*(T/S_y)^2)]
    # where Sn_prime = Marin-corrected finite-life fatigue strength for circular shaft.
    d_shaft_A  = float(get_or_warn(design, 'd_shaft_A',  0.003, context=_ctx))
    L_bearing  = float(get_or_warn(design, 'L_bearing',  0.010, context=_ctx))
    n_shaft_min_val = float(get_or_warn(design, 'n_shaft_min', 2.0,  context=_ctx))
    S_y_shaft  = float(get_or_warn(design, 'S_y',   345e6, context=_ctx))
    Sn_shaft   = float(get_or_warn(design, 'Sn',    133e6, context=_ctx))
    C_sur_s    = float(get_or_warn(design, 'C_sur',  0.88, context=_ctx))
    C_st_s     = float(get_or_warn(design, 'C_st',   1.0,  context=_ctx))
    C_R_s      = float(get_or_warn(design, 'C_R',    0.81, context=_ctx))
    C_m_s      = float(get_or_warn(design, 'C_m',    1.0,  context=_ctx))
    C_f_s      = float(get_or_warn(design, 'C_f',    1.0,  context=_ctx))
    # corrected endurance limit - circular section (same Marin factors as links)
    C_s_shaft  = fatigue._C_s_size_pin(d_shaft_A)
    Sn_prime_shaft = Sn_shaft * C_sur_s * C_s_shaft * C_st_s * C_R_s * C_m_s * C_f_s
    # evaluate at every sweep step; governing = minimum N_shaft
    n_shaft = float('inf')
    for F_A_mag, tau_A_abs in zip(F_A_mag_hist, tau_A_abs_hist):
        M_s = F_A_mag * L_bearing
        T_s = tau_A_abs
        denom_sq = (2.0 * M_s / Sn_prime_shaft)**2 + 3.0 * (T_s / S_y_shaft)**2
        if denom_sq > 0.0:
            N_s = math.pi * d_shaft_A**3 / (16.0 * math.sqrt(denom_sq))
            if N_s < n_shaft:
                n_shaft = N_s
    n_shaft_passed = (n_shaft >= n_shaft_min_val) if math.isfinite(n_shaft) else True

    # buckling check
    buckling_result = buckling.evaluate(F_r_rod_hist, design)

    # fatigue analysis
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
        # shaft A
        'n_shaft':           n_shaft,
        'n_shaft_passed':    n_shaft_passed,
        # buckling
        'n_buck':            buckling_result['n_buck'],
        'P_cr':              buckling_result['P_cr'],
        'N_max_comp':        buckling_result['N_max_comp'],
        'buckling_passed':   buckling_result['passed'],
        # motor torque & energy per revolution
        'tau_A_max':         tau_A_max,
        'E_rev':             E_rev,
        # peak joint forces (for bearing selection)
        'F_A_max':           F_A_max,
        'F_B_max':           F_B_max,
        'F_C_max':           F_C_max,
        # per-link peak normal stresses
        'sigma_rod_peak':    sigma_rod_peak,
        'sigma_crank_peak':  sigma_crank_peak,
        'sigma_pin_peak':    sigma_pin_peak,
        # per-angle histories for CSV export
        'kinematics_history': kinematics_history,
        'dynamics_history':   dynamics_history,
        'stresses_history':   stresses_history,
    }
    result.update(fatigue_result)

    return result
