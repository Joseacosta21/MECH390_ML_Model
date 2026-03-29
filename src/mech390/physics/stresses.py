"""
Stress evaluation module for Offset Crank-Slider Mechanism.

Implements all stress components from the Mother Doc (Sections 4-8) for the
connecting rod and crank arm at a single crank angle. Returns per-component
stress dicts so that engine.py can build separate histories for fatigue.

All quantities are in SI units throughout:
  - lengths : metres (m)
  - forces  : Newtons (N)
  - stresses: Pascals (Pa)

Mother Doc notation -> code variable mapping:
  w_rod     = design['width_l']         (rod width, in-plane)
  t_rod     = design['thickness_l']     (rod thickness, out-of-plane)
  w_crank   = design['width_r']         (crank width, in-plane)
  t_crank   = design['thickness_r']     (crank thickness, out-of-plane)
  l_rod     = design['l']               (rod length / centre distance)
  L_crank   = design['r']               (crank arm length / centre distance)
  D_pA      = design['pin_diameter_A']
  D_pB      = design['pin_diameter_B']
  D_pC      = design['pin_diameter_C']
  I_yr (rod Iyy)   = design['I_area_rod_yy']    = w_l * t_l^3 / 12
  I_zr (rod Izz)   = design['I_area_rod_zz']    = t_l * w_l^3 / 12
  I_yc (crank Iyy) = design['I_area_crank_yy']  = w_r * t_r^3 / 12
  I_zc (crank Izz) = design['I_area_crank_zz']  = t_r * w_r^3 / 12

Force frame decomposition sign convention (per codebase dynamics.py):
  In dynamics.py, F_B is the force ON the crank at pin B; the rod sees -F_B.
  In dynamics.py, F_A is the bearing reaction acting ON the crank.
  These conventions differ from the Mother Doc notation but are consistent
  with the Newton-Euler solver implemented in dynamics.py.

  phi = rod_angle(theta, r, l, e)       [rad]

  Rod frame (F_B is crank force, rod sees -F_B at pin B end):
    F_r,rod,B = -F_Bx*cos(phi) - F_By*sin(phi)
    F_t,rod,B =  F_Bx*sin(phi) - F_By*cos(phi)
    F_r,rod,C =  F_Cx*cos(phi) + F_Cy*sin(phi)
    F_t,rod,C = -F_Cx*sin(phi) + F_Cy*cos(phi)

  Crank frame (F_A acts on crank; F_B acts on crank):
    F_r,crank,B =  F_Bx*cos(theta) + F_By*sin(theta)
    F_t,crank,B = -F_Bx*sin(theta) + F_By*cos(theta)
    F_r,crank,A =  F_Ax*cos(theta) + F_Ay*sin(theta)
    F_t,crank,A = -F_Ax*sin(theta) + F_Ay*cos(theta)

Ref: The Mother Doc v7 (authoritative derivations), instructions.md
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from mech390.physics import kinematics

# ---------------------------------------------------------------------------
# Default diametral clearance
# ---------------------------------------------------------------------------
# D_hole = D_pin + _DELTA_DEFAULT
# 0.1 mm (1e-4 m) is a typical close-running fit clearance for small pins.
# TODO: Replace with design['delta_A/B/C'] once clearance is added to Stage 2.
_DELTA_DEFAULT: float = 1e-4  # 0.1 mm diametral clearance (m)

# Kt_fixed = 2.34 applied to net-section area Z_r or Z_c at each pin hole.
# Conservative estimate for single-shear round-ended lugs.
# Ref: Mother Doc Section 8.1 and Eqs 4.7, 6.8.
_KT_FIXED: float = 2.34

# ---------------------------------------------------------------------------
# Saint-Venant torsion coefficient (Roark approximation)
# ---------------------------------------------------------------------------

def _beta_torsion(w: float, t: float) -> float:
    """
    Roark approximation for Saint-Venant torsion stress coefficient beta.

    Mother Doc Eq 5.7 / 6.11:
        beta = 1/3 - 0.21 * (t/w) * [1 - t^4 / (12 * w^4)]

    Valid for w >= t (w is the longer cross-section dimension).

    Args:
        w: longer cross-section dimension (m)
        t: shorter cross-section dimension (m)

    Returns:
        beta (dimensionless)
    """
    return 1.0 / 3.0 - 0.21 * (t / w) * (1.0 - t**4 / (12.0 * w**4))


# ---------------------------------------------------------------------------
# Force frame decomposition helpers
# ---------------------------------------------------------------------------

def _rod_frame_forces(
    F_B: np.ndarray,
    F_C: np.ndarray,
    phi: float,
) -> Tuple[float, float, float, float]:
    """
    Decompose F_B and F_C into rod-local axial (r) and transverse (t) components.

    Sign convention matches codebase dynamics.py (F_B = force on crank; rod
    sees -F_B at pin B). See module docstring for full derivation.

    Returns:
        (F_r_rod_B, F_t_rod_B, F_r_rod_C, F_t_rod_C) in Newtons
    """
    cos_p = math.cos(phi)
    sin_p = math.sin(phi)

    F_r_rod_B = -F_B[0] * cos_p - F_B[1] * sin_p
    F_t_rod_B =  F_B[0] * sin_p - F_B[1] * cos_p
    F_r_rod_C =  F_C[0] * cos_p + F_C[1] * sin_p
    F_t_rod_C = -F_C[0] * sin_p + F_C[1] * cos_p

    return F_r_rod_B, F_t_rod_B, F_r_rod_C, F_t_rod_C


def _crank_frame_forces(
    F_A: np.ndarray,
    F_B: np.ndarray,
    theta: float,
) -> Tuple[float, float, float, float]:
    """
    Decompose F_A and F_B into crank-local axial (r) and transverse (t) components.

    Sign convention matches codebase dynamics.py (F_A = bearing reaction on
    crank; F_B = force on crank). See module docstring for full derivation.

    Returns:
        (F_r_crank_B, F_t_crank_B, F_r_crank_A, F_t_crank_A) in Newtons
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    F_r_crank_B =  F_B[0] * cos_t + F_B[1] * sin_t
    F_t_crank_B = -F_B[0] * sin_t + F_B[1] * cos_t
    F_r_crank_A =  F_A[0] * cos_t + F_A[1] * sin_t
    F_t_crank_A = -F_A[0] * sin_t + F_A[1] * cos_t

    return F_r_crank_B, F_t_crank_B, F_r_crank_A, F_t_crank_A


# ---------------------------------------------------------------------------
# Rod stresses (Mother Doc Sections 4 and 5)
# ---------------------------------------------------------------------------

def _rod_stresses(
    design: Dict[str, Any],
    F_B: np.ndarray,
    F_C: np.ndarray,
    theta: float,
) -> Tuple[float, float]:
    """
    Compute worst-case normal and shear stress in the connecting rod at one
    crank angle theta.

    Covers (Mother Doc Sections 4 and 5):
      - Axial body stress at Pin B and C ends           (sigma_ax,rod,body)
      - Axial hole stress at Pin B and C                (sigma_ax,rod,hole)
      - Gravity-induced peak bending stress             (sigma from M_rod,max)
      - Out-of-plane bending stress at Pin B            (sigma_oop,rod,B)
      - Torsional shear at rod body and holes           (tau_T,rod)
      - Transverse shear stress                         (tau from F_t)
      - Smallest-area lug shear at Pin B and C          (tau_sma)

    Returns:
        (sigma_rod, tau_rod) worst-case pair (Pa)
    """
    # --- Geometry ---
    w    = design['width_l']        # rod width (in-plane)
    t    = design['thickness_l']    # rod thickness (out-of-plane)
    L    = design['l']              # rod length (centre distance)
    D_pB = design['pin_diameter_B']
    D_pC = design['pin_diameter_C']
    I_yr = design['I_area_rod_yy']  # w*t^3/12 — out-of-plane bending (weak axis)
    I_zr = design['I_area_rod_zz']  # t*w^3/12 — in-plane bending (strong axis)
    g    = design.get('g', 9.81)
    m_rod = design.get('mass_rod', 0.0)

    # i_offset: out-of-plane offset at Pin B joint (Mother Doc)
    i_offset = (design['thickness_l'] + design['thickness_r']) / 2.0

    # Gross cross-section area
    A_r = w * t

    # Extreme fibre distances
    c_yr = t / 2.0   # out-of-plane (weak axis)
    c_zr = w / 2.0   # in-plane (strong axis)

    # Rod angle
    phi = kinematics.rod_angle(theta, design['r'], L, design['e'])

    # Force decomposition into rod frame
    F_r_B, F_t_B, F_r_C, F_t_C = _rod_frame_forces(F_B, F_C, phi)

    # --- Gravity distributed bending (Mother Doc Section 4.1) ---
    # w_rod,g = (m_rod * g / l_rod) * cos(phi)
    # M_rod,max = w_rod,g * l_rod^2 / 8  (midspan, parabolic, pin-pin BC)
    w_rod_g = (m_rod * g / L) * math.cos(phi)
    M_rod_max = w_rod_g * L**2 / 8.0

    # --- Axial body stress (Mother Doc Eq 4.6) — worst case (sum) ---
    sigma_ax_body_B = abs(F_r_B) / A_r + abs(M_rod_max) * c_zr / I_zr
    sigma_ax_body_C = abs(F_r_C) / A_r + abs(M_rod_max) * c_zr / I_zr

    # --- Axial hole stress (Mother Doc Eq 4.7) ---
    # Z_r,B = (w_rod - D_B_hole) * t_rod
    D_B_hole = D_pB + _DELTA_DEFAULT
    D_C_hole = D_pC + _DELTA_DEFAULT
    Z_r_B = max(w - D_B_hole, 1e-9) * t
    Z_r_C = max(w - D_C_hole, 1e-9) * t
    sigma_ax_hole_B = _KT_FIXED * abs(F_r_B) / Z_r_B
    sigma_ax_hole_C = _KT_FIXED * abs(F_r_C) / Z_r_C

    # --- Out-of-plane bending stress at Pin B (Mother Doc Eq 5.5) ---
    # M_eta,rod,B = F_r,rod,B * i_offset
    # sigma_oop = M_eta * c_yr / I_yr
    M_eta_rod_B = abs(F_r_B) * i_offset
    sigma_oop_rod_B = M_eta_rod_B * c_yr / I_yr

    # --- Torsion (Mother Doc Sections 5.5 / 5.6) ---
    # T_rod = F_t,rod,C * i_offset  (Eq 5.6)
    T_rod = abs(F_t_C) * i_offset
    beta_r = _beta_torsion(w, t)

    # tau_T,rod body (Eq 5.8): T_rod / (beta_r * w_rod^2 * t_rod)
    tau_T_rod_body = T_rod / (beta_r * w**2 * t)

    # tau_nom,hole at Pin B and C (Eq 5.9)
    hole_factor_B = max(1.0 - math.pi * D_B_hole**2 / (4.0 * w * t), 0.01)
    hole_factor_C = max(1.0 - math.pi * D_C_hole**2 / (4.0 * w * t), 0.01)
    tau_nom_hole_B = T_rod / (beta_r * w**2 * t * hole_factor_B)
    tau_nom_hole_C = T_rod / (beta_r * w**2 * t * hole_factor_C)

    # Peak hole torsional shear — Kt_u1 = 4 (Peterson 4.9.1, conservative)
    tau_max_hole_B = 4.0 * tau_nom_hole_B
    tau_max_hole_C = 4.0 * tau_nom_hole_C

    # --- Transverse shear (Mother Doc Eq 4.8) ---
    tau_xy_B = abs(F_t_B) / A_r
    tau_xy_C = abs(F_t_C) / A_r

    # --- Smallest-area lug shear (Mother Doc Eq 4.9) ---
    denom_B = max(w - D_B_hole, 1e-9) * t
    denom_C = max(w - D_C_hole, 1e-9) * t
    tau_sma_B = abs(F_t_B) / denom_B
    tau_sma_C = abs(F_t_C) / denom_C

    # --- Collect worst-case ---
    # sigma_ax_body already includes M_rod_max bending; sigma_grav not listed separately
    sigma_rod = max(
        sigma_ax_body_B,
        sigma_ax_body_C,
        sigma_ax_hole_B,
        sigma_ax_hole_C,
        sigma_oop_rod_B,
    )

    tau_rod = max(
        tau_T_rod_body,
        tau_max_hole_B,
        tau_max_hole_C,
        tau_xy_B,
        tau_xy_C,
        tau_sma_B,
        tau_sma_C,
    )

    return sigma_rod, tau_rod


# ---------------------------------------------------------------------------
# Crank stresses (Mother Doc Section 6)
# ---------------------------------------------------------------------------

def _crank_stresses(
    design: Dict[str, Any],
    F_A: np.ndarray,
    F_B: np.ndarray,
    theta: float,
) -> Tuple[float, float]:
    """
    Compute worst-case normal and shear stress in the crank arm at one
    crank angle theta.

    Covers (Mother Doc Section 6):
      - Axial body stress at Pin A and B ends           (sigma_ax,crank,body)
      - Axial hole stress at Pin A and B                (sigma_ax,crank,hole)
      - Gravity-induced peak bending stress             (sigma from M_crank,max)
      - Out-of-plane bending stress at Pin B            (sigma_oop,crank,B)
      - Torsional shear from T_in and T_offset          (tau_T,in, tau_T,offset)
      - Transverse shear stress                         (tau from F_t)
      - Smallest-area lug shear at Pin A and B          (tau_sma)

    Returns:
        (sigma_crank, tau_crank) worst-case pair (Pa)
    """
    # --- Geometry ---
    w    = design['width_r']         # crank width (in-plane)
    t    = design['thickness_r']     # crank thickness (out-of-plane)
    L    = design['r']               # crank arm length (centre distance)
    D_pA = design['pin_diameter_A']
    D_pB = design['pin_diameter_B']
    I_yc = design['I_area_crank_yy'] # w*t^3/12 — out-of-plane bending (weak axis)
    I_zc = design['I_area_crank_zz'] # t*w^3/12 — in-plane bending (strong axis)
    # T_in: motor input torque from Newton-Euler solver at this theta (N*m).
    # Injected into design dict by engine.py as design['tau_A'] = forces['tau_A'].
    T_in = design.get('tau_A', 0.0)
    g    = design.get('g', 9.81)
    m_crank = design.get('mass_crank', 0.0)

    # i_offset: out-of-plane offset at Pin B joint (Mother Doc)
    i_offset = (design['thickness_l'] + design['thickness_r']) / 2.0

    # Gross cross-section area
    A_c = w * t

    # Extreme fibre distances
    c_yc = t / 2.0   # out-of-plane (weak axis)
    c_zc = w / 2.0   # in-plane (strong axis)

    # Force decomposition into crank frame
    F_r_crank_B, F_t_crank_B, F_r_crank_A, F_t_crank_A = _crank_frame_forces(
        F_A, F_B, theta
    )

    # --- Gravity distributed bending (Mother Doc Section 6.1) ---
    # w_crank,g = (m_crank * g / L_crank) * cos(theta)
    # M_crank,max = w_crank,g * L_crank^2 / 8
    w_crank_g = (m_crank * g / L) * math.cos(theta)
    M_crank_max = w_crank_g * L**2 / 8.0

    # --- Axial body stress (Mother Doc Eq 6.7) — worst case (sum) ---
    sigma_ax_body_B = abs(F_r_crank_B) / A_c + abs(M_crank_max) * c_zc / I_zc
    sigma_ax_body_A = abs(F_r_crank_A) / A_c + abs(M_crank_max) * c_zc / I_zc

    # --- Axial hole stress (Mother Doc Eq 6.8) ---
    D_B_hole = D_pB + _DELTA_DEFAULT
    D_A_hole = D_pA + _DELTA_DEFAULT
    Z_c_B = max(w - D_B_hole, 1e-9) * t
    Z_c_A = max(w - D_A_hole, 1e-9) * t
    sigma_ax_hole_B = _KT_FIXED * abs(F_r_crank_B) / Z_c_B
    sigma_ax_hole_A = _KT_FIXED * abs(F_r_crank_A) / Z_c_A

    # --- Out-of-plane bending stress at Pin B (Mother Doc Eq 6.10) ---
    M_eta_crank_B = abs(F_r_crank_B) * i_offset
    sigma_oop_crank_B = M_eta_crank_B * c_yc / I_yc

    # --- Torsion (Mother Doc Sections 6.6 and 6.7) ---
    # T_offset = F_t,crank,B * i_offset  (Eq 6.16)
    T_offset = abs(F_t_crank_B) * i_offset
    beta_c = _beta_torsion(w, t)

    # tau_T,in = T_in / (beta_c * w_crank * t_crank^2)  (Eq 6.15)
    tau_T_in = abs(T_in) / (beta_c * w * t**2)

    # tau_T,offset = T_offset / (beta_c * w_crank * t_crank^2)  (Eq 6.17)
    tau_T_offset = T_offset / (beta_c * w * t**2)

    # Combined torsional shear at holes (Eq 6.18 / 6.19)
    # T_total = T_in + T_offset (same crank axis, same denominator)
    T_total = abs(T_in) + T_offset
    hole_factor_B = max(1.0 - math.pi * D_B_hole**2 / (4.0 * w * t), 0.01)
    hole_factor_A = max(1.0 - math.pi * D_A_hole**2 / (4.0 * w * t), 0.01)
    tau_nom_hole_B = T_total / (beta_c * w * t**2 * hole_factor_B)
    tau_nom_hole_A = T_total / (beta_c * w * t**2 * hole_factor_A)

    # Peak hole torsional shear — Kt_u1 = 4 (conservative, Peterson 4.9.1)
    tau_max_hole_B = 4.0 * tau_nom_hole_B
    tau_max_hole_A = 4.0 * tau_nom_hole_A

    # --- Transverse shear ---
    tau_xy_B = abs(F_t_crank_B) / A_c
    tau_xy_A = abs(F_t_crank_A) / A_c

    # --- Smallest-area lug shear (Mother Doc Eq 6.13) ---
    denom_B = max(w - D_B_hole, 1e-9) * t
    denom_A = max(w - D_A_hole, 1e-9) * t
    tau_sma_B = abs(F_t_crank_B) / denom_B
    tau_sma_A = abs(F_t_crank_A) / denom_A

    # --- Collect worst-case ---
    sigma_crank = max(
        sigma_ax_body_B,
        sigma_ax_body_A,
        sigma_ax_hole_B,
        sigma_ax_hole_A,
        sigma_oop_crank_B,
    )

    tau_crank = max(
        tau_T_in,
        tau_T_offset,
        tau_max_hole_B,
        tau_max_hole_A,
        tau_xy_B,
        tau_xy_A,
        tau_sma_B,
        tau_sma_A,
    )

    return sigma_crank, tau_crank


# ---------------------------------------------------------------------------
# Pin stresses (Mother Doc Section 7)
# ---------------------------------------------------------------------------

def _pin_stresses(
    design: Dict[str, Any],
    F_A: np.ndarray,
    F_B: np.ndarray,
    F_C: np.ndarray,
    theta: float,
) -> Tuple[float, float]:
    """
    Compute worst-case normal and shear stress in the pins.

    Covers (Mother Doc Section 7):
      - Pin shear: single shear at A and B, double shear at C
      - Pin bending stress
      - Bearing stress at all three pins

    Returns:
        (sigma_pin, tau_pin) worst-case pair (Pa)
    """
    D_pA = design['pin_diameter_A']
    D_pB = design['pin_diameter_B']
    D_pC = design['pin_diameter_C']
    t_rod   = design['thickness_l']
    t_crank = design['thickness_r']

    # Axial force components for bearing stress
    phi = kinematics.rod_angle(theta, design['r'], design['l'], design['e'])
    F_r_rod_B, _, F_r_rod_C, _ = _rod_frame_forces(F_B, F_C, phi)
    F_r_crank_B, _, F_r_crank_A, _ = _crank_frame_forces(F_A, F_B, theta)

    # Resultant force magnitudes for pin shear and pin bending
    mag_A = float(np.linalg.norm(F_A))
    mag_B = float(np.linalg.norm(F_B))
    mag_C = float(np.linalg.norm(F_C))

    # Pin cross-sectional areas
    A_pA = math.pi * D_pA**2 / 4.0
    A_pB = math.pi * D_pB**2 / 4.0
    A_pC = math.pi * D_pC**2 / 4.0

    # --- Pin shear (Mother Doc Section 7.1) ---
    tau_pin_A = mag_A / A_pA                   # single shear
    tau_pin_B = mag_B / A_pB                   # single shear
    tau_pin_C = mag_C / (2.0 * A_pC)           # double shear

    # --- Pin bending (Mother Doc Section 7.2) ---
    M_pin_A = mag_A * t_crank / 2.0
    M_pin_B = mag_B * t_crank / 2.0
    M_pin_C = mag_C * t_rod   / 4.0

    sigma_b_A = 32.0 * M_pin_A / (math.pi * D_pA**3)
    sigma_b_B = 32.0 * M_pin_B / (math.pi * D_pB**3)
    sigma_b_C = 32.0 * M_pin_C / (math.pi * D_pC**3)

    # --- Bearing stress (Mother Doc Section 7.3) ---
    sigma_br_A       = abs(F_r_crank_A) / (D_pA * t_crank)
    sigma_br_B_rod   = abs(F_r_rod_B)   / (D_pB * t_rod)
    sigma_br_B_crank = abs(F_r_crank_B) / (D_pB * t_crank)
    sigma_br_C       = abs(F_r_rod_C)   / (2.0 * D_pC * t_rod)

    # --- Collect worst-case ---
    sigma_pin = max(
        sigma_b_A,
        sigma_b_B,
        sigma_b_C,
        sigma_br_A,
        sigma_br_B_rod,
        sigma_br_B_crank,
        sigma_br_C,
    )

    tau_pin = max(
        tau_pin_A,
        tau_pin_B,
        tau_pin_C,
    )

    return sigma_pin, tau_pin


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def evaluate(
    design: Dict[str, Any],
    F_A: np.ndarray,
    F_B: np.ndarray,
    F_C: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate all stress components at a single crank angle and return
    per-component stresses plus overall worst-case scalars.

    Called by engine.py once per angle step in the 15-degree sweep.
    engine.py must inject the following into the design dict before calling:
        design['theta'] = theta          (current crank angle, rad)
        design['tau_A'] = forces['tau_A'] (motor torque at this step, N*m)

    Args:
        design : dict containing all geometry, cross-section, mass property,
                 and dynamics parameters. Required keys:
                   'r', 'l', 'e'                          — 2D kinematics (m)
                   'width_r', 'thickness_r'               — crank cross-section (m)
                   'width_l', 'thickness_l'               — rod cross-section (m)
                   'pin_diameter_A/B/C'                   — pin diameters (m)
                   'I_area_crank_yy', 'I_area_crank_zz'  — crank area MOI (m^4)
                   'I_area_rod_yy',   'I_area_rod_zz'    — rod area MOI (m^4)
                   'mass_rod', 'mass_crank'               — link masses (kg)
                   'tau_A'                                — motor torque (N*m)
                   'theta'                                — current crank angle (rad)
                   'g'           (optional, default 9.81) — gravity (m/s^2)
        F_A    : joint force at Pin A, np.ndarray([Fx, Fy]) (N)
        F_B    : joint force at Pin B, np.ndarray([Fx, Fy]) (N)
        F_C    : joint force at Pin C, np.ndarray([Fx, Fy]) (N)

    Returns:
        dict with keys:
            'sigma_rod'   : float — worst-case normal stress in rod (Pa)
            'tau_rod'     : float — worst-case shear stress in rod (Pa)
            'sigma_crank' : float — worst-case normal stress in crank (Pa)
            'tau_crank'   : float — worst-case shear stress in crank (Pa)
            'sigma_pin'   : float — worst-case normal stress in pins (Pa)
            'tau_pin'     : float — worst-case shear stress in pins (Pa)
            'sigma'       : float — overall worst-case normal stress (Pa)
            'tau'         : float — overall worst-case shear stress (Pa)
    """
    theta = design['theta']

    sigma_rod,   tau_rod   = _rod_stresses(design, F_B, F_C, theta)
    sigma_crank, tau_crank = _crank_stresses(design, F_A, F_B, theta)
    sigma_pin,   tau_pin   = _pin_stresses(design, F_A, F_B, F_C, theta)

    return {
        'sigma_rod':   sigma_rod,
        'tau_rod':     tau_rod,
        'sigma_crank': sigma_crank,
        'tau_crank':   tau_crank,
        'sigma_pin':   sigma_pin,
        'tau_pin':     tau_pin,
        'sigma':       max(sigma_rod, sigma_crank, sigma_pin),
        'tau':         max(tau_rod,   tau_crank,   tau_pin),
    }
