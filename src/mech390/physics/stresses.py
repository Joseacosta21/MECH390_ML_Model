"""
Stress evaluation for the offset crank-slider mechanism.

Computes rod, crank, and pin stresses at a single crank angle.
Returns per-component stress dicts so engine.py can build histories for fatigue.

All units are SI (lengths: m, forces: N, stresses: Pa).

Variable mapping:
  w_rod     = design['width_l']         (rod width, in-plane)
  t_rod     = design['thickness_l']     (rod thickness, out-of-plane)
  w_crank   = design['width_r']         (crank width, in-plane)
  t_crank   = design['thickness_r']     (crank thickness, out-of-plane)
  l_rod     = design['l']               (rod length / centre distance)
  L_crank   = design['r']               (crank arm length / centre distance)
  d_shaft_A = design['d_shaft_A']
  D_pB      = design['pin_diameter_B']
  D_pC      = design['pin_diameter_C']
  I_yr (rod Iyy)   = design['I_area_rod_yy']    = w_l * t_l^3 / 12
  I_zr (rod Izz)   = design['I_area_rod_zz']    = t_l * w_l^3 / 12
  I_yc (crank Iyy) = design['I_area_crank_yy']  = w_r * t_r^3 / 12
  I_zc (crank Izz) = design['I_area_crank_zz']  = t_r * w_r^3 / 12

Force frame decomposition sign convention (per dynamics.py):
  F_B is the force ON the crank at pin B; the rod sees -F_B.
  F_A is the bearing reaction acting ON the crank.

  phi = rod_angle(theta, r, l, e)

  Rod frame:
    F_r,rod,B = -F_Bx*cos(phi) - F_By*sin(phi)
    F_t,rod,B =  F_Bx*sin(phi) - F_By*cos(phi)
    F_r,rod,C =  F_Cx*cos(phi) + F_Cy*sin(phi)
    F_t,rod,C = -F_Cx*sin(phi) + F_Cy*cos(phi)

  Crank frame:
    F_r,crank,B =  F_Bx*cos(theta) + F_By*sin(theta)
    F_t,crank,B = -F_Bx*sin(theta) + F_By*cos(theta)
    F_r,crank,A =  F_Ax*cos(theta) + F_Ay*sin(theta)
    F_t,crank,A = -F_Ax*sin(theta) + F_Ay*cos(theta)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from mech390.physics import kinematics
from mech390.physics._utils import get_or_warn

# No module-level defaults for config-sourced constants.
# delta, Kt_lug, Kt_hole_torsion must be present in the design dict
# (injected from baseline.yaml). Missing keys raise KeyError.

### Saint-Venant torsion coefficient (Roark approximation)

# beta = 1/3 - 0.21 * (t/w) * [1 - t^4 / (12 * w^4)], valid for w >= t
def _beta_torsion(w: float, t: float) -> float:
    return 1.0 / 3.0 - 0.21 * (t / w) * (1.0 - t**4 / (12.0 * w**4))


### Force frame decomposition helpers

# decomposes F_B and F_C into rod-local axial (r) and transverse (t) components
# F_B is force on crank; rod sees -F_B at pin B - see module docstring for sign convention
# returns (F_r_rod_B, F_t_rod_B, F_r_rod_C, F_t_rod_C) in Newtons
def _rod_frame_forces(
    F_B: np.ndarray,
    F_C: np.ndarray,
    phi: float,
) -> Tuple[float, float, float, float]:
    cos_p = math.cos(phi)
    sin_p = math.sin(phi)

    F_r_rod_B = -F_B[0] * cos_p - F_B[1] * sin_p
    F_t_rod_B =  F_B[0] * sin_p - F_B[1] * cos_p
    F_r_rod_C =  F_C[0] * cos_p + F_C[1] * sin_p
    F_t_rod_C = -F_C[0] * sin_p + F_C[1] * cos_p

    return F_r_rod_B, F_t_rod_B, F_r_rod_C, F_t_rod_C


# decomposes F_A and F_B into crank-local axial (r) and transverse (t) components
# F_A = bearing reaction on crank; F_B = force on crank - see module docstring for sign convention
# returns (F_r_crank_B, F_t_crank_B, F_r_crank_A, F_t_crank_A) in Newtons
def _crank_frame_forces(
    F_A: np.ndarray,
    F_B: np.ndarray,
    theta: float,
) -> Tuple[float, float, float, float]:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    F_r_crank_B =  F_B[0] * cos_t + F_B[1] * sin_t
    F_t_crank_B = -F_B[0] * sin_t + F_B[1] * cos_t
    F_r_crank_A =  F_A[0] * cos_t + F_A[1] * sin_t
    F_t_crank_A = -F_A[0] * sin_t + F_A[1] * cos_t

    return F_r_crank_B, F_t_crank_B, F_r_crank_A, F_t_crank_A


### Rod stresses

# worst-case normal and shear stress in the connecting rod at one crank angle
# covers: corner body stress at pins B and C (axial + in-plane + OOP bending),
# axial hole stress, torsional shear at body and holes, transverse shear, lug shear
def _rod_stresses(
    design: Dict[str, Any],
    F_B: np.ndarray,
    F_C: np.ndarray,
    theta: float,
) -> Tuple[float, float]:
    # geometry
    w    = design['width_l']        # rod width (in-plane)
    t    = design['thickness_l']    # rod thickness (out-of-plane)
    L    = design['l']              # rod length (centre distance)
    D_pB = design['pin_diameter_B']
    D_pC = design['pin_diameter_C']
    I_yr = design['I_area_rod_yy']  # w*t^3/12 - out-of-plane bending (weak axis)
    I_zr = design['I_area_rod_zz']  # t*w^3/12 - in-plane bending (strong axis)

    # configurable stress-analysis constants - must be injected from baseline.yaml
    try:
        delta   = float(design['delta'])
        Kt_lug  = float(design['Kt_lug'])
        Kt_hole = float(design['Kt_hole_torsion'])
    except KeyError as exc:
        raise KeyError(
            f"stresses._rod_stresses: required key {exc} missing from design dict. "
            f"Ensure generate.py / validate_candidate.py injects all stress_analysis "
            f"constants from baseline.yaml before calling engine.evaluate_design()."
        ) from exc

    # i_offset: OOP eccentricity at Pin B - crank/rod touch face-to-face
    # Zero eccentricity at Pin C - rod/slider share z-centreline -> M_eta = 0 at C
    i_offset = (design['thickness_l'] + design['thickness_r']) / 2.0

    # gross cross-section area
    A_r = w * t

    # extreme fibre distances
    c_yr = t / 2.0   # out-of-plane (weak axis): eta = +-t/2
    c_zr = w / 2.0   # in-plane (strong axis):   zeta = +-w/2

    # rod angle
    phi = kinematics.rod_angle(theta, design['r'], L, design['e'])

    # force decomposition into rod frame
    F_r_B, F_t_B, F_r_C, F_t_C = _rod_frame_forces(F_B, F_C, phi)

    # bending moments
    # OOP (M_eta): eccentric axial force F_r_B acts through i_offset in zeta.
    #   No zeta pin reactions (system fully planar, dynamics 2D only) -> no lateral
    #   force balance along xi -> M_eta = F_r_B * i_offset CONSTANT along rod.
    # In-plane (M_zeta): transverse force F_t drives linear bending diagram;
    #   representative peak at xi = L/4.
    M_zeta_max_B = abs(F_t_B) * L / 4.0
    M_zeta_max_C = abs(F_t_C) * L / 4.0
    M_eta_rod    = abs(F_r_B) * i_offset   # constant along rod (no zeta reactions)

    # corner body stress at pin B end
    # corner (eta = +-t/2, zeta = +-w/2): in-plane AND OOP bending simultaneously
    # sigma = |F_r_B|/A_r + 6*M_zeta/(t*w^2) + 6*M_eta/(w*t^2)
    sigma_body_B_corner = (abs(F_r_B) / A_r
                           + M_zeta_max_B * c_zr / I_zr
                           + M_eta_rod    * c_yr / I_yr)

    # corner body stress at pin C end
    # M_eta same as at B (constant - no zeta reactions)
    # sigma = |F_r_C|/A_r + 6*M_zeta/(t*w^2) + 6*M_eta/(w*t^2)
    sigma_body_C_corner = (abs(F_r_C) / A_r
                           + M_zeta_max_C * c_zr / I_zr
                           + M_eta_rod    * c_yr / I_yr)

    # axial hole stress
    # Z_r,B = (w_rod - D_B_hole) * t_rod
    D_B_hole = D_pB + delta
    D_C_hole = D_pC + delta
    Z_r_B = max(w - D_B_hole, 1e-9) * t
    Z_r_C = max(w - D_C_hole, 1e-9) * t
    sigma_ax_hole_B = Kt_lug * abs(F_r_B) / Z_r_B
    sigma_ax_hole_C = Kt_lug * abs(F_r_C) / Z_r_C

    # torsion
    # T_rod = F_t,rod,C * i_offset: transverse force at C acts through i_offset
    T_rod = max(abs(F_t_B), abs(F_t_C)) * i_offset
    # Saint-Venant: tau_max = T / (beta*b*c^2), b = max(w,t), c = min(w,t)
    # tau_T_max occurs at (eta=0, zeta=+-c_r/2) - midpoint of the longer face
    b_r = max(w, t)
    c_r = min(w, t)
    beta_r = _beta_torsion(b_r, c_r)

    # tau_T,rod body - maximum at (eta=0, zeta=+-c_r/2)
    tau_T_rod_body = T_rod / (beta_r * b_r * c_r**2)

    # torsional shear at holes
    hole_factor_B = max(1.0 - math.pi * D_B_hole**2 / (4.0 * w * t), 0.01)
    hole_factor_C = max(1.0 - math.pi * D_C_hole**2 / (4.0 * w * t), 0.01)
    tau_nom_hole_B = T_rod / (beta_r * b_r * c_r**2 * hole_factor_B)
    tau_nom_hole_C = T_rod / (beta_r * b_r * c_r**2 * hole_factor_C)

    # peak hole torsional shear with stress concentration
    tau_max_hole_B = Kt_hole * tau_nom_hole_B
    tau_max_hole_C = Kt_hole * tau_nom_hole_C

    # transverse shear
    # tau_xeta peak = (3/2)*F_t/A at eta=0. Acts same direction as tau_T_max
    # at (eta=0, zeta=+-c_r/2) -> ADD, not max
    tau_V_rod = 1.5 * max(abs(F_t_B), abs(F_t_C)) / A_r
    tau_body_crit = tau_T_rod_body + tau_V_rod   # combined shear at (eta=0, zeta=+-c_r/2)

    # smallest-area lug shear
    # net-section lug shear: F_t/A_net (no 3/2; standard lug analysis)
    denom_B = max(w - D_B_hole, 1e-9) * t
    denom_C = max(w - D_C_hole, 1e-9) * t
    tau_sma_B = abs(F_t_B) / denom_B
    tau_sma_C = abs(F_t_C) / denom_C

    # collect worst-case
    # sigma: corner B - axial + in-plane + OOP bending at (+-t/2, +-w/2)
    #        corner C - axial + in-plane bending only (M_eta same constant value)
    #        holes    - axial net-section with Kt
    sigma_rod = max(
        sigma_body_B_corner,
        sigma_body_C_corner,
        sigma_ax_hole_B,
        sigma_ax_hole_C,
    )

    # tau: tau_body_crit = tau_T + tau_V summed at (eta=0, zeta=+-c_r/2)
    #      tau_max_hole_* = Kt-amplified torsion at holes
    #      tau_sma_*      = net-section lug shear
    tau_rod = max(
        tau_body_crit,
        tau_max_hole_B,
        tau_max_hole_C,
        tau_sma_B,
        tau_sma_C,
    )

    return sigma_rod, tau_rod


### Crank stresses

# worst-case normal and shear stress in the crank arm at one crank angle
# covers: corner body stress at pins B and A (axial + in-plane + OOP bending),
# axial hole stress, torsional shear (offset only), transverse shear, lug shear
#
# OOP bending: M_eta_crank = F_r_crank_B * i_offset, CONSTANT along crank.
# System fully planar (2D dynamics) -> no zeta pin reactions -> no moment gradient.
#
# T_in is NOT Saint-Venant bar torsion: T_in * crank_axis = 0 -> zero bar torsion.
# T_in acts as in-plane bending at section A: M_A = T_in, sigma = T_in * c_zc / I_zc.
#
# Gravity bending removed: crank self-weight already in F_A, F_B via Newton-Euler.
def _crank_stresses(
    design: Dict[str, Any],
    F_A: np.ndarray,
    F_B: np.ndarray,
    theta: float,
) -> Tuple[float, float]:
    # geometry
    w    = design['width_r']         # crank width (in-plane)
    t    = design['thickness_r']     # crank thickness (out-of-plane)
    L    = design['r']               # crank arm length (centre distance)
    d_shaft_A = design['d_shaft_A']
    D_pB = design['pin_diameter_B']
    I_yc = design['I_area_crank_yy'] # w*t^3/12 - out-of-plane bending (weak axis)
    I_zc = design['I_area_crank_zz'] # t*w^3/12 - in-plane bending (strong axis)
    _ctx = 'stresses._crank_stresses'
    T_in = get_or_warn(design, 'tau_A', 0.0, context=_ctx)

    # configurable stress-analysis constants - must be injected from baseline.yaml
    try:
        delta   = float(design['delta'])
        Kt_lug  = float(design['Kt_lug'])
        Kt_hole = float(design['Kt_hole_torsion'])
    except KeyError as exc:
        raise KeyError(
            f"stresses._crank_stresses: required key {exc} missing from design dict. "
            f"Ensure generate.py / validate_candidate.py injects all stress_analysis "
            f"constants from baseline.yaml before calling engine.evaluate_design()."
        ) from exc

    # i_offset: OOP eccentricity at Pin B - crank/rod touch face-to-face
    # No eccentricity at Pin A (motor shaft on crank neutral axis)
    # M_eta constant along crank (no zeta reactions - system fully planar)
    i_offset = (design['thickness_l'] + design['thickness_r']) / 2.0

    # gross cross-section area
    A_c = w * t

    # extreme fibre distances
    c_yc = t / 2.0   # out-of-plane (weak axis): eta = +-t/2
    c_zc = w / 2.0   # in-plane (strong axis):   zeta = +-w/2

    # force decomposition into crank frame
    F_r_crank_B, F_t_crank_B, F_r_crank_A, F_t_crank_A = _crank_frame_forces(
        F_A, F_B, theta
    )

    # bending moments
    # OOP (M_eta_crank): F_r_crank_B acts through i_offset at Pin B.
    #   No zeta reactions -> constant along crank. Same value at A and B corners.
    # In-plane (M_zeta): transverse force drives linear bending; peak at xi = L/4.
    # T_in at section A: reclassified as in-plane bending (not bar torsion).
    M_zeta_max_B = abs(F_t_crank_B) * L / 4.0
    M_zeta_max_A = abs(F_t_crank_A) * L / 4.0
    M_eta_crank  = abs(F_r_crank_B) * i_offset   # constant along crank

    # corner body stress at pin B end
    # corner (eta = +-t/2, zeta = +-w/2): axial + in-plane + OOP bending
    # sigma = |F_r_B|/A_c + 6*M_zeta/(t*w^2) + 6*M_eta/(w*t^2)
    sigma_body_B_corner = (abs(F_r_crank_B) / A_c
                           + abs(M_zeta_max_B) * c_zc / I_zc
                           + abs(M_eta_crank)  * c_yc / I_yc)

    # corner body stress at pin A end
    # T_in enters as additional in-plane bending at section A (M_A = T_in)
    # OOP term same as B - M_eta constant along crank
    # sigma = |F_r_A|/A_c + 6*M_zeta_A/(t*w^2) + T_in*c_zc/I_zc + 6*M_eta/(w*t^2)
    sigma_body_A_corner = (abs(F_r_crank_A) / A_c
                           + M_zeta_max_A * c_zc / I_zc
                           + abs(T_in)    * c_zc / I_zc
                           + M_eta_crank  * c_yc / I_yc)

    # axial hole stress
    D_B_hole = D_pB + delta
    D_A_hole = d_shaft_A + delta
    Z_c_B = max(w - D_B_hole, 1e-9) * t
    Z_c_A = max(w - D_A_hole, 1e-9) * t
    sigma_ax_hole_B = Kt_lug * abs(F_r_crank_B) / Z_c_B
    sigma_ax_hole_A = Kt_lug * abs(F_r_crank_A) / Z_c_A

    # torsion
    # T_in NOT bar torsion (T_in * crank_axis = 0)
    # only T_offset = F_t_crank_B * i_offset drives bar torsion
    T_offset = abs(max(F_t_crank_B, F_t_crank_A)) * i_offset
    b_c = max(w, t)
    c_c = min(w, t)
    beta_c = _beta_torsion(b_c, c_c)
    tau_T_offset = T_offset / (beta_c * b_c * c_c**2)

    # torsion at holes - T_offset only
    hole_factor_B = max(1.0 - math.pi * D_B_hole**2 / (4.0 * w * t), 0.01)
    hole_factor_A = max(1.0 - math.pi * D_A_hole**2  / (4.0 * w * t), 0.01)
    tau_nom_hole_B = T_offset / (beta_c * b_c * c_c**2 * hole_factor_B)
    tau_nom_hole_A = T_offset / (beta_c * b_c * c_c**2 * hole_factor_A)
    tau_max_hole_B = Kt_hole * tau_nom_hole_B
    tau_max_hole_A = Kt_hole * tau_nom_hole_A

    # transverse shear
    # tau_V peak = (3/2)*F_t/A at eta=0. Same point as tau_T_max -> ADD
    tau_V_crank = 1.5 * max(abs(F_t_crank_B), abs(F_t_crank_A)) / A_c
    tau_body_crit = tau_T_offset + tau_V_crank

    # smallest-area lug shear
    denom_B = max(w - D_B_hole, 1e-9) * t
    denom_A = max(w - D_A_hole, 1e-9) * t
    tau_sma_B = abs(F_t_crank_B) / denom_B
    tau_sma_A = abs(F_t_crank_A) / denom_A

    # collect worst-case
    # sigma: corner B - axial + in-plane + OOP
    #        corner A - axial + in-plane + T_in + OOP
    #        holes    - axial net-section with Kt
    sigma_crank = max(
        sigma_body_B_corner,
        sigma_body_A_corner,
        sigma_ax_hole_B,
        sigma_ax_hole_A,
    )

    # tau: tau_body_crit = tau_T_offset + tau_V at (eta=0, zeta=+-c_c/2)
    #      tau_max_hole_* = Kt-amplified torsion at holes
    #      tau_sma_*      = net-section lug shear
    tau_crank = max(
        tau_body_crit,
        tau_max_hole_B,
        tau_max_hole_A,
        tau_sma_B,
        tau_sma_A,
    )

    return sigma_crank, tau_crank


### Pin stresses

# worst-case normal and shear stress in the pins at one crank angle
# shaft A: transverse shear + torsional shear (T_in) combined
# pin B: single shear bending from max(t_crank, t_rod) arm
# pin C: double shear bending from t_rod/4 arm
# bearing stress computed for all three pins on both sides
def _pin_stresses(
    design: Dict[str, Any],
    F_A: np.ndarray,
    F_B: np.ndarray,
    F_C: np.ndarray,
    theta: float,
) -> Tuple[float, float]:
    d_shaft_A = design['d_shaft_A']
    D_pB = design['pin_diameter_B']
    D_pC = design['pin_diameter_C']
    t_rod    = design['thickness_l']
    t_crank  = design['thickness_r']
    _ctx = 'stresses._pin_stresses'
    T_in         = get_or_warn(design, 'tau_A',        0.0,  context=_ctx)
    slider_height = get_or_warn(design, 'slider_height', 0.02, context=_ctx)

    # axial force components for bearing stress
    phi = kinematics.rod_angle(theta, design['r'], design['l'], design['e'])
    F_r_rod_B, _, F_r_rod_C, _ = _rod_frame_forces(F_B, F_C, phi)
    F_r_crank_B, _, F_r_crank_A, _ = _crank_frame_forces(F_A, F_B, theta)

    # resultant force magnitudes for transverse shear and bending
    mag_A = float(np.linalg.norm(F_A))
    mag_B = float(np.linalg.norm(F_B))
    mag_C = float(np.linalg.norm(F_C))

    # pin cross-sectional areas (circular)
    A_sA = math.pi * d_shaft_A**2 / 4.0
    A_pB = math.pi * D_pB**2 / 4.0
    A_pC = math.pi * D_pC**2 / 4.0

    # transverse shear
    tau_pin_A = mag_A / A_sA          # shaft A - single shear
    tau_pin_B = mag_B / A_pB          # pin B - single shear
    tau_pin_C = mag_C / (2.0 * A_pC) # pin C - double shear

    # shaft A torsional shear
    # T_in applied directly at shaft A; tau = T*c/J = 16*T / (pi*d^3)
    tau_shaft_A_torsion = 16.0 * abs(T_in) / (math.pi * d_shaft_A**3)

    # combined shear at shaft A: transverse + torsional ADD
    tau_pin_A_total = tau_pin_A + tau_shaft_A_torsion

    # bending stress
    # shaft A: moment arm = t_crank / 2 (bore mid-plane)
    M_pin_A = mag_A * t_crank / 2.0
    # pin B: spans both crank and rod lugs; larger arm governs
    M_pin_B = mag_B * max(t_crank, t_rod) / 2.0
    # pin C: double shear -> moment arm = t_rod / 4
    M_pin_C = mag_C * t_rod / 4.0

    sigma_b_A = 32.0 * M_pin_A / (math.pi * d_shaft_A**3)
    sigma_b_B = 32.0 * M_pin_B / (math.pi * D_pB**3)
    sigma_b_C = 32.0 * M_pin_C / (math.pi * D_pC**3)

    # bearing stress
    sigma_br_A         = abs(F_r_crank_A) / (d_shaft_A * t_crank)
    sigma_br_B_rod     = abs(F_r_rod_B)   / (D_pB * t_rod)
    sigma_br_B_crank   = abs(F_r_crank_B) / (D_pB * t_crank)
    sigma_br_C_rod     = abs(F_r_rod_C)   / (2.0 * D_pC * t_rod)     # double shear, rod lugs
    sigma_br_C_slider  = abs(F_r_rod_C)   / (D_pC * slider_height)   # single, slider lug

    # collect worst-case
    sigma_pin = max(
        sigma_b_A,
        sigma_b_B,
        sigma_b_C,
        sigma_br_A,
        sigma_br_B_rod,
        sigma_br_B_crank,
        sigma_br_C_rod,
        sigma_br_C_slider,
    )

    tau_pin = max(
        tau_pin_A_total,   # shaft A - transverse + torsion combined
        tau_pin_B,         # pin B - transverse only
        tau_pin_C,         # pin C - transverse only (double shear)
    )

    return sigma_pin, tau_pin


### Public interface

def evaluate(
    design: Dict[str, Any],
    F_A: np.ndarray,
    F_B: np.ndarray,
    F_C: np.ndarray,
) -> Dict[str, Any]:
    """Evaluates rod, crank, and pin stresses at one crank angle. Returns per-component worst-case values."""
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
