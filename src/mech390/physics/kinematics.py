"""
Kinematics module for Offset Crank-Slider Mechanism.
Implements deterministic, physics-based calculations for position, velocity, acceleration,
and derived metrics (ROM, QRR).

All position/velocity/acceleration functions return np.ndarray of shape (2,)
representing [x, y] components.

Ref: instructions.md (Authoritative)
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Slider C — constrained to x-axis (y = 0 always)
# ---------------------------------------------------------------------------

def slider_position(theta: float, r: float, l: float, e: float) -> np.ndarray:
    """
    Calculates the slider position vector pos_C(theta).

    Equation:
        x_C(theta) = r * cos(theta) + sqrt(l^2 - (r * sin(theta) + e)^2)
        y_C        = 0  (slider constrained to x-axis)

    Args:
        theta (float): Crank angle in radians.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset (vertical distance from crank pivot to slider line).

    Returns:
        np.ndarray: Slider position vector [x_C, 0.0].

    Raises:
        ValueError: If geometry constraints are violated (term under sqrt < 0).
    """
    term_under_sqrt = l**2 - (r * np.sin(theta) + e)**2

    # Check for physical validity
    if np.any(term_under_sqrt < 0):
        if np.isscalar(term_under_sqrt):
            raise ValueError(
                f"Geometry violation: l={l}, r={r}, e={e}, theta={theta}. "
                f"Term under sqrt {term_under_sqrt} is negative."
            )
        else:
            pass

    x_C = r * np.cos(theta) + np.sqrt(term_under_sqrt)
    return np.array([x_C, 0.0])


def slider_velocity(theta: float, omega: float, r: float, l: float, e: float) -> np.ndarray:
    """
    Calculates the slider linear velocity vector vel_C.

    Derivation: v_C = dx/dt = (dx/dtheta) * omega
                y-component is always 0 (slider constrained to x-axis).

    Args:
        theta (float): Crank angle in radians.
        omega (float): Angular velocity in rad/s.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset.

    Returns:
        np.ndarray: Slider velocity vector [v_Cx, 0.0].
    """
    sq_term = np.sqrt(l**2 - (r * np.sin(theta) + e)**2)
    # dx/dtheta = -r*sin(theta) - (r*cos(theta)*(r*sin(theta) + e)) / sqrt(...)
    dx_dtheta = -r * np.sin(theta) - (r * np.cos(theta) * (r * np.sin(theta) + e)) / sq_term
    v_Cx = dx_dtheta * omega
    return np.array([v_Cx, 0.0])


def slider_acceleration(theta: float, omega: float, r: float, l: float, e: float) -> np.ndarray:
    """
    Calculates the slider linear acceleration vector acc_C.
    Assumes constant omega (alpha = 0).

    a_C = d(v_C)/dt = d(v_C)/dtheta * omega
    y-component is always 0 (slider constrained to x-axis).

    Args:
        theta (float): Crank angle in radians.
        omega (float): Angular velocity in rad/s.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset.

    Returns:
        np.ndarray: Slider acceleration vector [a_Cx, 0.0].
    """
    # Let u = r*sin(theta) + e
    # Let S = sqrt(l^2 - u^2)
    # x = r*cos(theta) + S
    # dx/dtheta = -r*sin(theta) - (r*cos(theta)*u)/S

    # d(dx/dtheta)/dtheta:
    # Term 1: d(-r*sin(theta)) = -r*cos(theta)
    # Term 2: d( (r*cos(theta)*u) / S ) via quotient rule

    # u = r*sin(theta) + e  => du/dtheta = r*cos(theta)
    # S = (l^2 - u^2)^0.5   => dS/dtheta = -u/S * r*cos(theta)

    # Numerator N = r*cos(theta)*u
    # dN/dtheta = -r*u*sin(theta) + r^2*cos^2(theta)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    u = r * sin_t + e
    S = np.sqrt(l**2 - u**2)

    term1 = -r * cos_t

    numerator_deriv = -r * u * sin_t + r**2 * cos_t**2
    S_deriv = -(u * r * cos_t) / S

    term2_deriv = (S * numerator_deriv - (r * cos_t * u) * S_deriv) / (S**2)

    d2x_dtheta2 = term1 - term2_deriv
    a_Cx = d2x_dtheta2 * (omega**2)
    return np.array([a_Cx, 0.0])


# ---------------------------------------------------------------------------
# Crank pin B — moves in a full circle (genuine 2D motion)
# ---------------------------------------------------------------------------

def crank_pin_position(theta: float, r: float) -> np.ndarray:
    """
    Calculates the crank pin B position vector.

    The crank pin moves in a full circle of radius r about the origin.

    Args:
        theta (float): Crank angle in radians.
        r (float): Crank radius.

    Returns:
        np.ndarray: Crank pin position vector [r*cos(theta), r*sin(theta)].
    """
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def crank_pin_velocity(theta: float, omega: float, r: float) -> np.ndarray:
    """
    Calculates the crank pin B velocity vector.

    Derivation: d/dt [r*cos(theta), r*sin(theta)] with dtheta/dt = omega.

    Args:
        theta (float): Crank angle in radians.
        omega (float): Angular velocity in rad/s.
        r (float): Crank radius.

    Returns:
        np.ndarray: Crank pin velocity vector [-r*omega*sin(theta), r*omega*cos(theta)].
    """
    return np.array([-r * omega * np.sin(theta), r * omega * np.cos(theta)])


def crank_pin_acceleration(theta: float, omega: float, r: float) -> np.ndarray:
    """
    Calculates the crank pin B acceleration vector.
    Assumes constant omega (alpha = 0), so only centripetal acceleration.

    Derivation: d/dt [-r*omega*sin(theta), r*omega*cos(theta)]

    Args:
        theta (float): Crank angle in radians.
        omega (float): Angular velocity in rad/s.
        r (float): Crank radius.

    Returns:
        np.ndarray: Crank pin acceleration vector [-r*omega^2*cos(theta), -r*omega^2*sin(theta)].
    """
    return np.array([-r * omega**2 * np.cos(theta), -r * omega**2 * np.sin(theta)])


# ---------------------------------------------------------------------------
# Dead-center detection and metrics
# ---------------------------------------------------------------------------

def get_dead_center_angles(r: float, l: float, e: float):
    """
    Finds the two crank angles where slider velocity is zero (dead centers).
    Uses robust root-finding as required by specs.

    Velocity is zero when dx/dtheta is zero.
    Equation: -r*sin(theta) - (r*cos(theta)*(r*sin(theta)+e)) / sqrt(...) = 0

    Arguments:
        r, l, e: geometric parameters

    Returns:
        tuple: (theta_retracted, theta_extended) in range [0, 2pi), sorted.
        Returns None if roots cannot be found (invalid geometry).
    """

    def velocity_proxy(theta):
        # Returns x-component of slider velocity (proportional to velocity scalar)
        if abs(r * np.sin(theta) + e) >= l:
            return np.nan
        return slider_velocity(theta, 1.0, r, l, e)[0]  # index [0] for scalar

    # Sweep [0, 2pi) at 1-degree resolution to bracket roots
    thetas = np.linspace(0, 2*np.pi, 360)
    vals = []

    for th in thetas:
        try:
            v = velocity_proxy(th)
            vals.append(v)
        except (ValueError, ArithmeticError):
            vals.append(np.nan)

    vals = np.array(vals)

    # Identify sign changes (bracket roots)
    roots = []
    for i in range(len(vals) - 1):
        if np.isnan(vals[i]) or np.isnan(vals[i+1]):
            continue
        if np.sign(vals[i]) != np.sign(vals[i+1]):
            try:
                root = brentq(velocity_proxy, thetas[i], thetas[i+1])
                roots.append(root)
            except Exception:
                pass

    # Normalize to [0, 2pi)
    roots = np.unique(np.array(roots) % (2*np.pi))

    if len(roots) != 2:
        pass

    return np.sort(roots)


def calculate_metrics(r: float, l: float, e: float) -> dict:
    """
    Computes ROM and QRR for the given geometry.

    Args:
        r, l, e: Geometry.

    Returns:
        dict: {'ROM': float, 'QRR': float, 'theta_retracted': float,
               'theta_extended': float, 'x_min': float, 'x_max': float}
              or {'valid': False, 'reason': str} if geometry invalid.
    """

    # 1. Validate basic existence
    if l <= r + abs(e):
        # l > r + |e| ensures full rotation without locking
        # Max of (r*sin(theta)+e)^2 is (r+|e|)^2, so l^2 > (r+|e|)^2 => l > r+|e|
        return {'valid': False, 'reason': f"Rod too short for full rotation: l={l}, r={r}, e={e} (needs l > {r + abs(e)})"}

    # 2. Find dead centers
    roots = get_dead_center_angles(r, l, e)

    if len(roots) < 2:
        return {'valid': False, 'reason': 'Cannot find 2 dead centers'}

    theta1, theta2 = roots[0], roots[1]

    # Calculate positions — index [0] to extract x-component for ROM calculation
    x1 = slider_position(theta1, r, l, e)[0]
    x2 = slider_position(theta2, r, l, e)[0]

    # Determine Extended (max x) and Retracted (min x)
    if x1 > x2:
        x_max, theta_max = x1, theta1
        x_min, theta_min = x2, theta2
    else:
        x_max, theta_max = x2, theta2
        x_min, theta_min = x1, theta1

    ROM = x_max - x_min

    # 3. Calculate QRR
    # Delta_theta_forward = (theta_max - theta_min) mod 2pi
    # Delta_theta_return  = 2pi - Delta_theta_forward
    # QRR = Delta_theta_forward / Delta_theta_return
    delta_forward = (theta_max - theta_min) % (2 * np.pi)
    delta_return = (2 * np.pi) - delta_forward

    if delta_return == 0:
        QRR = float('inf')
    else:
        QRR = delta_forward / delta_return

    return {
        'valid': True,
        'ROM': ROM,
        'QRR': QRR,
        'theta_retracted': theta_min,
        'theta_extended': theta_max,
        'x_min': x_min,
        'x_max': x_max
    }
