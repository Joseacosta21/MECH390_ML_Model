"""
Physics engine for Offset Crank-Slider Mechanism.
Orchestrates the 15-degree sweep, evaluates kinematics/dynamics/stresses
at each crank angle, and tracks peak stress metrics.

Ref: instructions.md (Authoritative)
"""

import numpy as np
from typing import Dict, Any

from mech390.physics import kinematics, dynamics, stresses, mass_properties


def evaluate_design(design: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a 3D mechanism design over one full crank revolution (15° steps).

    At each crank angle the engine computes:
      - pos_slider  = kinematics.slider_position(...)    -> np.ndarray([x, 0])
      - vel_slider  = kinematics.slider_velocity(...)    -> np.ndarray([vx, 0])
      - acc_slider  = kinematics.slider_acceleration(...)-> np.ndarray([ax, 0])
      - pos_B       = kinematics.crank_pin_position(...) -> np.ndarray([x, y])
      - vel_B       = kinematics.crank_pin_velocity(...) -> np.ndarray([vx, vy])
      - acc_B       = kinematics.crank_pin_acceleration(...)-> np.ndarray([ax, ay])
      - cog_crank   = mass_properties.crank_cog(...)     -> np.ndarray([x, y])
      - cog_rod     = mass_properties.rod_cog(...)       -> np.ndarray([x, y])
      - cog_slider  = mass_properties.slider_cog(...)    -> np.ndarray([x, 0])
      - forces      = dynamics.joint_reaction_forces(...) -> dict of np.ndarray([Fx, Fy])
      - sigma, tau  = stresses.evaluate(...)             -> floats

    Args:
        design (dict): Must contain at least:
            'r', 'l', 'e'    — 2D kinematic parameters
            'omega'          — angular velocity [rad/s]
            'mass_crank'     — crank mass [kg]
            'mass_rod'       — rod mass [kg]
            'mass_slider'    — slider mass [kg]
            (plus 3D geometry params for stress evaluation)

    Returns:
        dict with keys:
            'sigma_max'        : float — peak normal stress [Pa]
            'tau_max'          : float — peak shear stress [Pa]
            'theta_sigma_max'  : float — crank angle at sigma_max [rad]
            'theta_tau_max'    : float — crank angle at tau_max [rad]
            'valid_physics'    : bool
    """
    r = design['r']
    l = design['l']
    e = design['e']
    omega = design.get('omega', 1.0)  # default 1 rad/s if not specified

    mass_crank  = design.get('mass_crank', 1.0)
    mass_rod    = design.get('mass_rod', 1.0)
    mass_slider = design.get('mass_slider', 1.0)
    i_crank = design.get('I_mass_crank_cg_z', 1.0)
    i_rod = design.get('I_mass_rod_cg_z', 1.0)
    mu = design.get('mu', 0.0)
    g = design.get('g', 9.81)
    alpha_r = design.get('alpha_r', 0.0)

    # 15-degree sweep over one full revolution
    thetas = np.deg2rad(np.arange(0, 360, 15))

    sigma_max = 0.0
    tau_max = 0.0
    theta_sigma_max = 0.0
    theta_tau_max = 0.0

    try:
        for theta in thetas:
            # --- Kinematics (all return np.ndarray([x, y])) ---
            pos_slider = kinematics.slider_position(theta, r, l, e)
            vel_slider = kinematics.slider_velocity(theta, omega, r, l, e)
            acc_slider = kinematics.slider_acceleration(theta, omega, r, l, e)

            pos_B = kinematics.crank_pin_position(theta, r)
            vel_B = kinematics.crank_pin_velocity(theta, omega, r)
            acc_B = kinematics.crank_pin_acceleration(theta, omega, r)

            # --- Mass properties (COGs as np.ndarray([x, y])) ---
            cog_crank  = mass_properties.crank_cog(theta, r)
            cog_rod    = mass_properties.rod_cog(theta, r, l, e)
            cog_slider = mass_properties.slider_cog(theta, r, l, e)

            # --- Dynamics (forces as np.ndarray([Fx, Fy])) ---
            forces = dynamics.joint_reaction_forces(
                theta, omega, r, l, e,
                mass_crank, mass_rod, mass_slider,
                I_crank=i_crank,
                I_rod=i_rod,
                mu=mu,
                g=g,
                alpha_r=alpha_r,
            )
            F_B = forces['F_B']
            F_C = forces['F_C']
            F_O = forces['F_O']

            # --- Stresses ---
            # stresses.evaluate returns (sigma, tau) scalars
            # TODO: implement when stresses.py is populated
            # sigma, tau = stresses.evaluate(design, F_B, F_C)
            sigma, tau = 0.0, 0.0  # placeholder until stresses.py is implemented

            # --- Track maxima ---
            if sigma > sigma_max:
                sigma_max = sigma
                theta_sigma_max = theta

            if tau > tau_max:
                tau_max = tau
                theta_tau_max = theta

    except Exception:
        return {'valid_physics': False}

    return {
        'sigma_max': sigma_max,
        'tau_max': tau_max,
        'theta_sigma_max': theta_sigma_max,
        'theta_tau_max': theta_tau_max,
        'valid_physics': True,
    }
