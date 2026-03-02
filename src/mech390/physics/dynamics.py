"""
Dynamics module for Offset Crank-Slider Mechanism.
Implements Newton-Euler equations for joint reaction forces.

All force/position vectors are np.ndarray of shape (2,) representing [x, y].

Ref: instructions.md (Authoritative)
"""

import numpy as np
from mech390.physics import kinematics


def joint_reaction_forces(
    theta: float,
    omega: float,
    r: float,
    l: float,
    e: float,
    mass_crank: float,
    mass_rod: float,
    mass_slider: float,
) -> dict:
    """
    Computes joint reaction force vectors at each pin for a given crank angle.
    Uses Newton-Euler equations applied to each body.

    Bodies:
        Crank (link 2): rotates about fixed pivot O.
        Connecting rod (link 3): couples crank pin B to slider C.
        Slider (link 4): constrained to x-axis.

    Args:
        theta (float): Crank angle in radians.
        omega (float): Angular velocity in rad/s (constant).
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset (vertical distance from crank pivot to slider line).
        mass_crank (float): Mass of crank [kg].
        mass_rod (float): Connecting rod mass [kg].
        mass_slider (float): Slider mass [kg].

    Returns:
        dict with keys:
            'F_B' : np.ndarray([Fx, Fy]) — reaction force at crank pin B [N]
            'F_C' : np.ndarray([Fx, Fy]) — reaction force at slider pin C [N]
            'F_O' : np.ndarray([Fx, Fy]) — ground reaction at crank pivot O [N]
    """
    # --- Kinematics ---
    acc_slider = kinematics.slider_acceleration(theta, omega, r, l, e)  # [ax, 0]
    acc_B = kinematics.crank_pin_acceleration(theta, omega, r)          # [ax, ay]

    # --- Slider (link 4): F_net = m * a ---
    # Only horizontal force acts (N from ground provides vertical equilibrium)
    F_C = mass_slider * acc_slider          # np.ndarray([Fx, 0])

    # --- Connecting rod (link 3): F_B + F_C_reaction = m_rod * a_G3 ---
    # Center of gravity of connecting rod (approximated as midpoint of B and C)
    pos_B = kinematics.crank_pin_position(theta, r)
    pos_C = kinematics.slider_position(theta, r, l, e)
    pos_G3 = (pos_B + pos_C) / 2.0

    # Velocity/acceleration of G3 (finite-difference would be needed for exact value;
    # for now we use the mean of B and C accelerations as a first-order approximation)
    acc_G3 = (acc_B + acc_slider) / 2.0

    # F_B = m_rod * a_G3 - F_C_on_rod
    # F_C_on_rod is reaction to F_C (Newton 3rd law): -F_C
    F_B = mass_rod * acc_G3 - (-F_C)

    # --- Crank (link 2): F_O = m_crank * a_G2 - F_B_reaction ---
    pos_G2 = kinematics.crank_pin_position(theta, r) / 2.0  # midpoint of O and B
    acc_G2 = kinematics.crank_pin_acceleration(theta, omega, r) / 2.0  # same scaling

    F_O = mass_crank * acc_G2 - (-F_B)

    return {
        'F_B': F_B,
        'F_C': F_C,
        'F_O': F_O,
    }