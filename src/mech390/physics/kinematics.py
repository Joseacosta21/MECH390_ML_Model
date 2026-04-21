"""
Kinematics for the offset crank-slider mechanism.

Computes position, velocity, and acceleration for all bodies,
plus derived metrics (ROM, QRR). All position/velocity/acceleration
functions return np.ndarray of shape (2,) for [x, y] components.
"""

import numpy as np
from scipy.optimize import brentq


def _slider_radicand(theta: float, r: float, l: float, e: float) -> float:
    """Return radicand of the slider-position square-root expression."""
    return float(l**2 - (r * np.sin(theta) + e) ** 2)


def _slider_sqrt_term(theta: float, r: float, l: float, e: float) -> tuple[float, float]:
    """
    Returns (u, sqrt_term) where u = r*sin(theta) + e, sqrt_term = sqrt(l^2 - u^2).
    Raises ValueError for invalid geometry.
    """
    u = float(r * np.sin(theta) + e)
    radicand = _slider_radicand(theta, r, l, e)
    if radicand < 0.0:
        raise ValueError(
            f"Geometry violation: l={l}, r={r}, e={e}, theta={theta}. "
            f"Term under sqrt {radicand} is negative."
        )
    return u, float(np.sqrt(radicand))


### Slider C - constrained to x-axis (y = 0 always)

# x_C(theta) = r*cos(theta) + sqrt(l^2 - (r*sin(theta) + e)^2), y_C = 0
def slider_position(theta: float, r: float, l: float, e: float) -> np.ndarray:
    """Returns slider position vector [x_C, 0.0]."""
    _, sq_term = _slider_sqrt_term(theta, r, l, e)
    x_C = r * np.cos(theta) + sq_term
    return np.array([x_C, 0.0])


# v_C = dx/dt = (dx/dtheta) * omega, y-component always 0
def slider_velocity(theta: float, omega: float, r: float, l: float, e: float) -> np.ndarray:
    """Returns slider velocity vector [v_Cx, 0.0]."""
    u, sq_term = _slider_sqrt_term(theta, r, l, e)
    # dx/dtheta = -r*sin(theta) - (r*cos(theta)*(r*sin(theta) + e)) / sqrt(...)
    dx_dtheta = -r * np.sin(theta) - (r * np.cos(theta) * u) / sq_term
    v_Cx = dx_dtheta * omega
    return np.array([v_Cx, 0.0])


# a_C = d(v_C)/dtheta * omega, assumes constant omega (alpha = 0)
def slider_acceleration(theta: float, omega: float, r: float, l: float, e: float) -> np.ndarray:
    """Returns slider acceleration vector [a_Cx, 0.0]."""
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
    u, S = _slider_sqrt_term(theta, r, l, e)

    term1 = -r * cos_t

    numerator_deriv = -r * u * sin_t + r**2 * cos_t**2
    S_deriv = -(u * r * cos_t) / S

    term2_deriv = (S * numerator_deriv - (r * cos_t * u) * S_deriv) / (S**2)

    d2x_dtheta2 = term1 - term2_deriv
    a_Cx = d2x_dtheta2 * (omega**2)
    return np.array([a_Cx, 0.0])


### Crank pin B - moves in a full circle (genuine 2D motion)

def crank_pin_position(theta: float, r: float) -> np.ndarray:
    """Returns crank pin B position [r*cos(theta), r*sin(theta)]."""
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def crank_pin_velocity(theta: float, omega: float, r: float) -> np.ndarray:
    """Returns crank pin B velocity [-r*omega*sin(theta), r*omega*cos(theta)]."""
    return np.array([-r * omega * np.sin(theta), r * omega * np.cos(theta)])


# centripetal only - constant omega assumed (alpha = 0)
def crank_pin_acceleration(theta: float, omega: float, r: float) -> np.ndarray:
    """Returns crank pin B acceleration [-r*omega^2*cos(theta), -r*omega^2*sin(theta)]."""
    return np.array([-r * omega**2 * np.cos(theta), -r * omega**2 * np.sin(theta)])


### Dead-center detection and metrics

def rod_angle(theta: float, r: float, l: float, e: float) -> float:
    """
    Connecting-rod orientation phi from the positive x-axis (angle between link BC and slider axis).

    sin(phi) = -(e + r*sin(theta)) / l enforces the slider constraint y_C = 0.
    Returns the branch with positive cosine (open configuration).
    """
    sin_phi = -(e + r * np.sin(theta)) / l
    if abs(sin_phi) > 1.0:
        raise ValueError(f"rod_angle undefined (|sin_phi|>1): {sin_phi}")

    # choose cos_phi positive to match the open mechanism
    cos_phi = np.sqrt(max(0.0, 1.0 - sin_phi ** 2))
    phi = np.arctan2(sin_phi, cos_phi)
    return phi


def rod_angular_velocity(theta: float, omega: float, r: float, l: float, e: float) -> float:
    """
    Angular speed of the connecting rod about pin B.

    From V_{Cy}=0 and V_B = [-r*omega*sin(theta), r*omega*cos(theta)]:
        omega_cb = -V_By / (l * cos(phi))
    """
    phi = rod_angle(theta, r, l, e)
    V_By = omega * r * np.cos(theta)
    return -V_By / (l * np.cos(phi))


def rod_angular_acceleration(
    theta: float,
    omega: float,
    r: float,
    l: float,
    e: float,
    alpha2: float = 0.0,
) -> float:
    """
    Angular acceleration of the connecting rod about pin B.

    From a_{Cy}=0:
        alpha_cb = (omega_cb^2 * l * sin(phi) - a_By) / (l * cos(phi))
    where a_By = alpha2 * r * cos(theta) - omega^2 * r * sin(theta).
    alpha2 is the crank angular acceleration (0 at constant speed).
    """
    phi = rod_angle(theta, r, l, e)
    omega_cb = rod_angular_velocity(theta, omega, r, l, e)
    # acceleration of B (y component)
    a_By = alpha2 * r * np.cos(theta) - omega ** 2 * r * np.sin(theta)
    return (omega_cb ** 2 * l * np.sin(phi) - a_By) / (l * np.cos(phi))


def get_dead_center_angles(r: float, l: float, e: float):
    """
    Finds the two crank angles where slider velocity is zero (dead centers).

    Returns a sorted np.ndarray of two roots in [0, 2pi), or an empty array
    when dead centers cannot be found.
    """

    def velocity_proxy(theta):
        # returns scalar dx/dtheta for root finding
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        u = r * sin_t + e
        if abs(u) >= l:
            return np.nan
        sq_term = np.sqrt(l**2 - u**2)
        return -r * sin_t - (r * cos_t * u) / sq_term

    # sweep [0, 2pi) at 1-degree resolution to bracket roots
    thetas = np.linspace(0, 2*np.pi, 360)
    vals = np.array([velocity_proxy(th) for th in thetas], dtype=float)

    # identify sign changes (bracket roots)
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

    # normalize to [0, 2pi)
    roots = np.unique(np.array(roots) % (2*np.pi))

    if len(roots) != 2:
        return np.array([])

    return np.sort(roots)


def calculate_metrics(r: float, l: float, e: float) -> dict:
    """
    Computes ROM and QRR for the given geometry.

    Returns {'ROM', 'QRR', 'theta_retracted', 'theta_extended', 'x_min', 'x_max'}
    or {'valid': False, 'reason': str} if the geometry is invalid.
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

    # calculate positions - index [0] to extract x-component for ROM calculation
    x1 = slider_position(theta1, r, l, e)[0]
    x2 = slider_position(theta2, r, l, e)[0]

    # determine extended (max x) and retracted (min x)
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
