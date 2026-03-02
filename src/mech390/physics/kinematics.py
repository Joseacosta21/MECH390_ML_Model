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
        raise ValueError(
            f"Geometry violation: l={l}, r={r}, e={e}, theta={theta}. "
            f"Term under sqrt {term_under_sqrt} is negative."
        )

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


# ---------------------------------------------------------------------------
# Helper helpers derived from vector formulation provided by user notes
# ---------------------------------------------------------------------------

def rod_angle(theta: float, r: float, l: float, e: float) -> float:
    """
    Compute the connecting-rod orientation \phi measured from the positive
    x-axis (i.e. the angle between link BC and the slider axis).

    The user derivation gives

        sin(phi) = -(e + r*sin(theta)) / l

    which enforces the slider constraint y_C = 0.  We choose the branch with
    *positive* cosine (open configuration) by constructing the angle with
    ``atan2(sin, cos_positive)``.

    Args:
        theta: crank angle [rad]
        r: crank radius
        l: rod length
        e: offset (positive upward)

    Returns:
        phi: rod angle [rad]

    Raises:
        ValueError: if the argument of arcsin exceeds unity (invalid geometry).
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
    Relative angular speed of the connecting rod about pin B (\omega_{C/B}).

    Derived from the condition that the slider moves purely horizontally
    (V_{Cy}=0) and using

        V_B = [\omega r sin\theta, \omega r cos\theta]

    so

        \omega_{C/B} = -V_{By} / (l cos\phi)

    Args:
        theta: crank angle [rad]
        omega: crank angular velocity [rad/s] (CW positive)
        r, l, e: geometry
    Returns:
        omega_{C/B} [rad/s]
    """
    phi = rod_angle(theta, r, l, e)
    V_By = omega * r * np.cos(theta)  # from V_B j-component
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
    Angular acceleration of the connecting rod about pin B (\alpha_{C/B}).

    Using the slider constraint (a_{Cy}=0) and the previously derived
    expressions, we obtain

        \alpha_{C/B} = (\omega_{C/B}^2 l sin\phi - a_{By})/(l cos\phi)

    where
        a_{By} = -\alpha_2 r cos\theta - \omega^2 r sin\theta

    ``alpha2`` is the crank angular acceleration (default zero for constant
    speed).  The sign conventions follow the CW positive orientation used
    elsewhere in the code.
    """
    phi = rod_angle(theta, r, l, e)
    omega_cb = rod_angular_velocity(theta, omega, r, l, e)
    # acceleration of B (y component)
    a_By = -alpha2 * r * np.cos(theta) - omega ** 2 * r * np.sin(theta)
    return (omega_cb ** 2 * l * np.sin(phi) - a_By) / (l * np.cos(phi))


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
        # Returns scalar dx/dtheta for robust root finding.
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        u = r * sin_t + e
        if abs(u) >= l:
            return np.nan
        sq_term = np.sqrt(l**2 - u**2)
        return -r * sin_t - (r * cos_t * u) / sq_term

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
        return np.array([])

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

