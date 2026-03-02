"""
Mass properties module for Offset Crank-Slider Mechanism.
Computes center of gravity (COG) positions and mass moments of inertia for each link.

All COG positions are np.ndarray of shape (2,) representing [x, y].

Ref: instructions.md (Authoritative)
"""

import numpy as np
from mech390.physics import kinematics


def crank_cog(theta: float, r: float) -> np.ndarray:
    """
    Computes the center of gravity of the crank (link 2).

    The crank rotates about fixed pivot O (origin). Its COG is approximated
    at the midpoint between O and the crank pin B.

    Args:
        theta (float): Crank angle in radians.
        r (float): Crank radius.

    Returns:
        np.ndarray: COG position vector [x, y] of the crank.
    """
    pos_B = kinematics.crank_pin_position(theta, r)
    pos_O = np.array([0.0, 0.0])  # fixed pivot at origin
    return (pos_O + pos_B) / 2.0


def rod_cog(theta: float, r: float, l: float, e: float) -> np.ndarray:
    """
    Computes the center of gravity of the connecting rod (link 3).

    Approximated at the midpoint between the crank pin B and slider pin C.

    Args:
        theta (float): Crank angle in radians.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset.

    Returns:
        np.ndarray: COG position vector [x, y] of the connecting rod.
    """
    pos_B = kinematics.crank_pin_position(theta, r)
    pos_C = kinematics.slider_position(theta, r, l, e)  # [x_C, 0.0]
    return (pos_B + pos_C) / 2.0


def slider_cog(theta: float, r: float, l: float, e: float) -> np.ndarray:
    """
    Computes the center of gravity of the slider (link 4).

    The slider is constrained to the x-axis, so its COG coincides with
    the slider pin C.

    Args:
        theta (float): Crank angle in radians.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset.

    Returns:
        np.ndarray: COG position vector [x_C, 0.0].
    """
    return kinematics.slider_position(theta, r, l, e)


def moment_of_inertia_rod(mass: float, length: float) -> float:
    """
    Mass moment of inertia of the connecting rod about its center of gravity.
    Modelled as a uniform slender rod: I = (1/12) * m * L^2.

    Args:
        mass (float): Mass of the rod [kg].
        length (float): Length of the rod [m].

    Returns:
        float: Izz about the rod's COG [kg·m²].
    """
    return (1.0 / 12.0) * mass * length**2


def moment_of_inertia_crank(mass: float, r: float) -> float:
    """
    Mass moment of inertia of the crank about the fixed pivot O.
    Modelled as a uniform slender rod: I = (1/3) * m * r^2.

    Args:
        mass (float): Mass of the crank [kg].
        r (float): Crank radius [m].

    Returns:
        float: Izz about pivot O [kg·m²].
    """
    return (1.0 / 3.0) * mass * r**2