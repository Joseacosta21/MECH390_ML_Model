"""
Kinematics module for Offset Crank-Slider Mechanism.
Implements deterministic, physics-based calculations for position, velocity, acceleration,
and derived metrics (ROM, QRR).

Ref: instructions.md (Authoritative)
"""

import numpy as np
from scipy.optimize import brentq


def slider_position(theta: float, r: float, l: float, e: float) -> float:
    """
    Calculates the slider position x_C(theta).

    Equation:
        x_C(theta) = r * cos(theta) + sqrt(l^2 - (r * sin(theta) + e)^2)

    Args:
        theta (float): Crank angle in radians.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset vertical distance.

    Returns:
        float: Slider position x_C.

    Raises:
        ValueError: If geometrical constraints are violated (term under sqrt < 0).
    """
    term_under_sqrt = l**2 - (r * np.sin(theta) + e)**2
    
    # Check for physical validity; floating point issues might cause slightly negative numbers for exact limits
    if np.any(term_under_sqrt < 0):
        # If it's a scalar check
        if np.isscalar(term_under_sqrt):
             raise ValueError(f"Geometry violation: l={l}, r={r}, e={e}, theta={theta}. Term under sqrt {term_under_sqrt} is negative.")
        else:
             # For arrays, we let it produce nans or handle it, but per spec this function might be called with scalars mostly.
             # If arrays are passed, we shouldn't raise immediately unless we want to fail fast.
             # Given the "deterministic" and "physics-first" requirement, returning NaN or raising is appropriate.
             # Let's rely on numpy warning/raising for invalid sqrt if not handled.
             pass

    return r * np.cos(theta) + np.sqrt(term_under_sqrt)


def slider_velocity(theta: float, omega: float, r: float, l: float, e: float) -> float:
    """
    Calculates the slider linear velocity v_C.

    Derivation: v_C = dx/dt = (dx/dtheta) * (dtheta/dt) = (dx/dtheta) * omega

    Args:
        theta (float): Crank angle in radians.
        omega (float): Angular velocity in rad/s.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset.

    Returns:
        float: Slider velocity.
    """
    sq_term = np.sqrt(l**2 - (r * np.sin(theta) + e)**2)
    # Derivative of x_C w.r.t theta:
    # dx/dtheta = -r*sin(theta) + (1/(2*sqrt(...))) * (-2 * (r*sin(theta)+e) * r*cos(theta))
    #           = -r*sin(theta) - (r*cos(theta)*(r*sin(theta) + e)) / sqrt(...)
    
    dx_dtheta = -r * np.sin(theta) - (r * np.cos(theta) * (r * np.sin(theta) + e)) / sq_term
    return dx_dtheta * omega


def slider_acceleration(theta: float, omega: float, r: float, l: float, e: float) -> float:
    """
    Calculates the slider linear acceleration a_C.
    Assumes constant omega (alpha = 0).

    a_C = d(v_C)/dt = d(v_C)/dtheta * omega

    Args:
        theta (float): Crank angle in radians.
        omega (float): Angular velocity in rad/s.
        r (float): Crank radius.
        l (float): Connecting rod length.
        e (float): Offset.

    Returns:
        float: Slider acceleration.
    """
    # Let u = r*sin(theta) + e
    # Let S = sqrt(l^2 - u^2)
    # x = r*cos(theta) + S
    # dx/dtheta = -r*sin(theta) - (r*cos(theta)*u)/S
    
    # We need d(dx/dtheta)/dtheta
    # Term 1: d(-r*sin(theta)) = -r*cos(theta)
    # Term 2: d( (r*cos(theta)*u) / S )
    # quotient rule: (S * d(numerator) - numerator * d(S)) / S^2
    
    # u = r*sin(theta) + e  => du/dtheta = r*cos(theta)
    # S = (l^2 - u^2)^0.5   => dS/dtheta = 0.5*(l^2-u^2)^(-0.5) * (-2u * du/dtheta) = -u/S * r*cos(theta)
    
    # Numerator N = r*cos(theta)*u
    # dN/dtheta = r*(-sin(theta))*u + r*cos(theta)*(du/dtheta)
    #           = -r*u*sin(theta) + r^2*cos^2(theta)
    
    # d(N/S) = (S * (-r*u*sin(theta) + r^2*cos^2(theta)) - (r*cos(theta)*u) * (-u*r*cos(theta)/S)) / S^2
    #        = ( (-r*u*sin(theta) + r^2*cos^2(theta)) / S ) + ( (r^2 * cos^2(theta) * u^2) / S^3 )
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    u = r * sin_t + e
    S = np.sqrt(l**2 - u**2)
    
    term1 = -r * cos_t
    
    # d(N/S) expanded for clarity/stability
    numerator_deriv = -r * u * sin_t + r**2 * cos_t**2
    S_deriv = -(u * r * cos_t) / S
    
    term2_deriv = (S * numerator_deriv - (r * cos_t * u) * S_deriv) / (S**2)
    
    d2x_dtheta2 = term1 - term2_deriv
    
    return d2x_dtheta2 * (omega**2)


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
        # Computes dx/dtheta (proportional to velocity)
        # We handle domain errors inside by returning NaN or checking bounds
        if abs(r * np.sin(theta) + e) >= l:
             # Invalid position physically
             return np.nan
        val = slider_velocity(theta, 1.0, r, l, e)
        return val

    # We expect two roots in [0, 2pi).
    # Analytic insight suggests they are near where the connecting rod and crank are collinear.
    # Collinear extended: theta s.t. rod and crank align outward.
    # Collinear retracted: rod overlaps crank.
    # However, strictly using numerical search as requested.
    
    # We will sweep the domain to bracket roots.
    # 0 to 2pi
    thetas = np.linspace(0, 2*np.pi, 360) # 1 degree coarse search
    vals = []
    
    valid_geometry = True
    for th in thetas:
        try:
             v = velocity_proxy(th)
             vals.append(v)
        except (ValueError, ArithmeticError):
             vals.append(np.nan)
             valid_geometry = False
    
    vals = np.array(vals)
    
    # Identify sign changes
    roots = []
    for i in range(len(vals) - 1):
        if np.isnan(vals[i]) or np.isnan(vals[i+1]):
             continue
        if np.sign(vals[i]) != np.sign(vals[i+1]):
             # Bracket found
             try:
                 root = brentq(velocity_proxy, thetas[i], thetas[i+1])
                 roots.append(root)
             except Exception:
                 pass
                 
    # Normalize to [0, 2pi)
    roots = np.unique(np.array(roots) % (2*np.pi))
    
    # We expect exactly 2 roots for a valid crank-slider
    if len(roots) != 2:
        # Fallback or error. 
        # Sometimes roots might be near 0/2pi boundary
        # Let's try to return what we have or raise error if critical
        pass
        
    return np.sort(roots)


def calculate_metrics(r: float, l: float, e: float) -> dict:
    """
    Computes ROM and QRR for the given geometry.

    Args:
        r, l, e: Geometry.

    Returns:
        dict: {'ROM': float, 'QRR': float, 'theta_min': float, 'theta_max': float}
              or fails if geometry invalid.
    """
    
    # 1. Validate basic existence
    if l <= r + abs(e):
         # Valid crank-slider constraint: l must be long enough to reach.
         # Actually more strict: l > r + e for full rotation?
         # For full rotation (Grashof condition for slider-crank):
         # The crank must be the shortest link relative to ground offset?
         # Specifically: l must be > r + |e| ensures full rotation without locking?
         # Let's check the sqrt term: l^2 - (r*sin(theta)+e)^2 >= 0 for all theta
         # Max of (r*sin(theta)+e)^2 is (r + |e|)^2.
         # So l^2 >= (r+|e|)^2  =>  l >= r + |e|.
         # Strictly greater for non-locking behavior usually.
        # Valid crank-slider constraint: l must be long enough to reach.
        # Strict inequality l > r + |e| ensures full rotation without locking
        # as the term under sqrt in x(theta) must be non-negative for all theta.
        # Max value of (r*sin(theta) + e)^2 is (r + |e|)^2.
        # So we need l^2 > (r + |e|)^2  =>  l > r + |e|.
        return {'valid': False, 'reason': f"Rod too short for full rotation: l={l}, r={r}, e={e} (needs l > {r + abs(e)})"}

    # 2. Find dead centers
    roots = get_dead_center_angles(r, l, e)
    
    if len(roots) < 2:
         return {'valid': False, 'reason': 'Cannot find 2 dead centers'}
         
    theta1, theta2 = roots[0], roots[1]
    
    # Calculate positions
    x1 = slider_position(theta1, r, l, e)
    x2 = slider_position(theta2, r, l, e)
    
    # Determine which is Extended (max x) and Retracted (min x)
    if x1 > x2:
        x_max, theta_max = x1, theta1
        x_min, theta_min = x2, theta2
    else:
        x_max, theta_max = x2, theta2
        x_min, theta_min = x1, theta1
        
    ROM = x_max - x_min
    
    # Calculate QRR
    # "Forward" stroke is usually the working stroke (slower?), Return is quick.
    # QRR = Time_forward / Time_return
    # Depending on convention, usually we want to know ratio > 1.
    # Let's compute delta thetas.
    
    # We move from Retracted (min) to Extended (max) -> Forward?
    # Or Extended to Retracted?
    # Spec says:
    # Delta_theta_forward = (theta_max - theta_min) mod 2pi ??
    # Spec lines 121-123:
    # "Delta_theta_forward = (theta_max - theta_min) mod 2pi"
    # "Delta_theta_return = 2pi - Delta_theta_forward"
    # "QRR = Delta_theta_forward / Delta_theta_return"
    # This implies Forward is defined as moving from min to max? 
    # Let's strictly follow the forumla.
    
    delta_forward = (theta_max - theta_min) % (2 * np.pi)
    delta_return = (2 * np.pi) - delta_forward
    
    # Avoid division by zero
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
