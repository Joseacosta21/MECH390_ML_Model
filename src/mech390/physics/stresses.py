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
  d_shaft_A = design['d_shaft_A']
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
from mech390.physics._utils import get_or_warn

# No module-level defaults for config-sourced constants.
# delta, Kt_lug, Kt_hole_torsion must be present in the design dict
# (injected from baseline.yaml). Missing keys raise KeyError.

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
    _ctx = 'stresses._rod_stresses'
    g    = get_or_warn(design, 'g', 9.81, context=_ctx)
    m_rod = get_or_warn(design, 'mass_rod', 0.0, context=_ctx)

    # Configurable stress-analysis constants — must be injected from baseline.yaml
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
    sigma_ax_body_B = (abs(F_r_B) / A_r) + (abs(M_rod_max) * c_zr / I_zr)
    sigma_ax_body_C = (abs(F_r_C) / A_r) + (abs(M_rod_max) * c_zr / I_zr)

    # --- Axial hole stress (Mother Doc Eq 4.7) ---
    # Z_r,B = (w_rod - D_B_hole) * t_rod
    D_B_hole = D_pB + delta
    D_C_hole = D_pC + delta
    Z_r_B = max(w - D_B_hole, 1e-9) * t
    Z_r_C = max(w - D_C_hole, 1e-9) * t
    sigma_ax_hole_B = Kt_lug * abs(F_r_B) / Z_r_B
    sigma_ax_hole_C = Kt_lug * abs(F_r_C) / Z_r_C

    # --- Critical Point 2: out-of-plane extreme fibre (η=0, ζ=±c_r/2) ---
    # The governing combined-stress location on the rod cross-section:
    #   · OOP bending (from i_offset at B) is maximum here — extreme OOP fibre
    #   · Saint-Venant τ_T_max also peaks here — midpoint of longer face (Roark)
    #   · In-plane transverse shear τ_V also peaks here — τ_xη is uniform in ζ,
    #     maximum at η=0 (neutral axis for in-plane loading)
    # Full normal stress at Point 2 = axial + OOP bending (no in-plane bending at η=0).
    # Compare with sigma_ax_body_{B,C} (Point 1: axial + in-plane bending at η=±w/2).
    M_eta_rod_B = abs(F_r_B) * i_offset
    sigma_body_B_pt2 = (abs(F_r_B) / A_r) + (M_eta_rod_B * c_yr / I_yr)

    # --- Torsion (Mother Doc Sections 5.5 / 5.6) ---
    # T_rod = F_t,rod,C * i_offset  (Eq 5.6)
    T_rod = abs(F_t_C) * i_offset
    # Saint-Venant: τ_max = T / (β·b·c²), b = max(w,t), c = min(w,t)  (Roark Table 10.7).
    # c is ALWAYS min(w,t) regardless of torsion direction; torsion direction only affects sign.
    # τ_T_max occurs at (η=0, ζ=±c_r/2) — midpoint of the longer face.
    b_r = max(w, t)
    c_r = min(w, t)
    beta_r = _beta_torsion(b_r, c_r)

    # tau_T,rod body — maximum at Point 2 (η=0, ζ=±c_r/2)
    tau_T_rod_body = T_rod / (beta_r * b_r * c_r**2)

    # tau_nom,hole at Pin B and C (Eq 5.9)
    hole_factor_B = max(1.0 - math.pi * D_B_hole**2 / (4.0 * w * t), 0.01)
    hole_factor_C = max(1.0 - math.pi * D_C_hole**2 / (4.0 * w * t), 0.01)
    tau_nom_hole_B = T_rod / (beta_r * b_r * c_r**2 * hole_factor_B)
    tau_nom_hole_C = T_rod / (beta_r * b_r * c_r**2 * hole_factor_C)

    # Peak hole torsional shear — Kt_hole_torsion (Peterson 4.9.1, conservative)
    tau_max_hole_B = Kt_hole * tau_nom_hole_B
    tau_max_hole_C = Kt_hole * tau_nom_hole_C

    # --- Transverse shear at Point 2 (Mother Doc Eq 4.8; corrected) ---
    # For a rectangular section, τ_xη(η) is uniform across ζ; peak = (3/2)·F_t/A at η=0
    # (Mott Sec. 13-2; Shigley Sec. 3-11).  At Point 2 (η=0) this is the maximum transverse
    # shear, acting in the η-direction — SAME direction as τ_T_max at that point.
    # → τ_T and τ_V ADD at Point 2; they must not be compared with max().
    tau_V_rod = 1.5 * max(abs(F_t_B), abs(F_t_C)) / A_r
    tau_body_crit = tau_T_rod_body + tau_V_rod   # combined critical shear at Point 2

    # --- Smallest-area lug shear (Mother Doc Eq 4.9) ---
    # Net-section lug shear uses F_t/A_net (no 3/2; standard lug analysis nominal shear).
    denom_B = max(w - D_B_hole, 1e-9) * t
    denom_C = max(w - D_C_hole, 1e-9) * t
    tau_sma_B = abs(F_t_B) / denom_B
    tau_sma_C = abs(F_t_C) / denom_C

    # --- Collect worst-case ---
    # sigma: Point 1 (η=±w/2) → sigma_ax_body = axial + in-plane bending
    #        Point 2 (η=0, ζ=±c_r/2) → sigma_body_B_pt2 = axial + OOP bending
    sigma_rod = max(
        sigma_ax_body_B,
        sigma_ax_body_C,
        sigma_ax_hole_B,
        sigma_ax_hole_C,
        sigma_body_B_pt2,
    )

    # tau: tau_body_crit = τ_T + τ_V summed at Point 2 (same location, same direction)
    #      tau_max_hole_* = torsion-dominated at holes with Kt (τ_V secondary at holes)
    #      tau_sma_* = net-section lug shear (no 3/2; lug analysis convention)
    tau_rod = max(
        tau_body_crit,
        tau_max_hole_B,
        tau_max_hole_C,
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

    Covers (Mother Doc Section 6, corrected):
      - Axial body stress at Pin A and B ends           (sigma_ax,crank,body)
      - Axial hole stress at Pin A and B                (sigma_ax,crank,hole)
      - Gravity-induced peak bending stress             (sigma from M_crank,max)
      - T_in in-plane bending at section A              (sigma_ax,body_A includes T_in term)
      - Out-of-plane bending at Pin B — Point 2         (sigma_body_B_pt2 = axial + OOP bending)
      - Torsional shear from T_offset only              (tau_T,offset; T_in is NOT bar torsion)
      - Transverse shear + torsion combined at Point 2  (tau_body_crit = tau_T + (3/2)F_t/A)
      - Smallest-area lug shear at Pin A and B          (tau_sma)

    Returns:
        (sigma_crank, tau_crank) worst-case pair (Pa)
    """
    # --- Geometry ---
    w    = design['width_r']         # crank width (in-plane)
    t    = design['thickness_r']     # crank thickness (out-of-plane)
    L    = design['r']               # crank arm length (centre distance)
    d_shaft_A = design['d_shaft_A']
    D_pB = design['pin_diameter_B']
    I_yc = design['I_area_crank_yy'] # w*t^3/12 — out-of-plane bending (weak axis)
    I_zc = design['I_area_crank_zz'] # t*w^3/12 — in-plane bending (strong axis)
    # T_in: motor input torque from Newton-Euler solver at this theta (N*m).
    # Injected into design dict by engine.py as design['tau_A'] = forces['tau_A'].
    _ctx = 'stresses._crank_stresses'
    T_in = get_or_warn(design, 'tau_A', 0.0, context=_ctx)
    g    = get_or_warn(design, 'g', 9.81, context=_ctx)
    m_crank = get_or_warn(design, 'mass_crank', 0.0, context=_ctx)

    # Configurable stress-analysis constants — must be injected from baseline.yaml
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
    w_crank_g = (m_crank * g / L) * math.cos(theta)
    M_crank_max = w_crank_g * L**2 / 8.0

    # --- Axial body stress (Mother Doc Eq 6.7) ---
    # Point 1 (η=±w/2, ζ=0): in-plane extreme fibre — axial + in-plane bending.
    sigma_ax_body_B = abs(F_r_crank_B) / A_c + abs(M_crank_max) * c_zc / I_zc
    # At section A: T_in enters as in-plane bending, NOT Saint-Venant bar torsion.
    #   T_in = T_in·ẑ;  bar axis ê₁ = (cosθ, sinθ, 0);  T_in·ẑ·ê₁ = 0  → zero bar torsion.
    #   T_in·ẑ·ẑ = T_in → in-plane bending moment M_A = T_in at section A (zero at B).
    #   σ = M·c_z/I_zc = T_in·(w/2) / (t·w³/12) = 6·T_in / (t·w²).
    sigma_ax_body_A = (abs(F_r_crank_A) / A_c
                       + abs(M_crank_max) * c_zc / I_zc
                       + abs(T_in) * c_zc / I_zc)   # T_in in-plane bending at section A

    # --- Axial hole stress (Mother Doc Eq 6.8) ---
    D_B_hole = D_pB + delta
    D_A_hole = d_shaft_A + delta   # shaft bore at section A
    Z_c_B = max(w - D_B_hole, 1e-9) * t
    Z_c_A = max(w - D_A_hole, 1e-9) * t
    sigma_ax_hole_B = Kt_lug * abs(F_r_crank_B) / Z_c_B
    sigma_ax_hole_A = Kt_lug * abs(F_r_crank_A) / Z_c_A

    # --- Critical Point 2: out-of-plane extreme fibre (η=0, ζ=±c_c/2) ---
    # Same governing point as for rod: max τ_T, max τ_V, max OOP bending all coincide here.
    # Full normal stress = axial + OOP bending (no in-plane bending contribution at η=0).
    M_eta_crank_B = abs(F_r_crank_B) * i_offset
    sigma_body_B_pt2 = abs(F_r_crank_B) / A_c + M_eta_crank_B * c_yc / I_yc

    # --- Torsion (Mother Doc Section 6.7) ---
    # T_in does NOT create Saint-Venant torsion of the crank bar:
    #   T_in·ẑ · ê₁ = 0  (see sigma_ax_body_A note above).
    # Only T_offset = F_t,crank,B · i_offset drives bar torsion (Eq 6.16).
    T_offset = abs(F_t_crank_B) * i_offset
    # τ_max = T / (β·b·c²), b = max(w,t), c = min(w,t).
    # τ_T_max occurs at (η=0, ζ=±c_c/2) — midpoint of the longer face.
    b_c = max(w, t)
    c_c = min(w, t)
    beta_c = _beta_torsion(b_c, c_c)

    # tau_T,offset — only genuine bar torsion (Eq 6.17); T_in removed
    tau_T_offset = T_offset / (beta_c * b_c * c_c**2)

    # Combined torsion for holes: T_total = T_offset only (T_in is not bar torsion)
    T_total = T_offset
    hole_factor_B = max(1.0 - math.pi * D_B_hole**2 / (4.0 * w * t), 0.01)
    hole_factor_A = max(1.0 - math.pi * D_A_hole**2  / (4.0 * w * t), 0.01)
    tau_nom_hole_B = T_total / (beta_c * b_c * c_c**2 * hole_factor_B)
    tau_nom_hole_A = T_total / (beta_c * b_c * c_c**2 * hole_factor_A)

    # Peak hole torsional shear — Kt_hole_torsion (Peterson 4.9.1, conservative)
    tau_max_hole_B = Kt_hole * tau_nom_hole_B
    tau_max_hole_A = Kt_hole * tau_nom_hole_A

    # --- Transverse shear at Point 2 (corrected) ---
    # τ_xη at (η=0, ζ=±c_c/2) = (3/2)·F_t/A — maximum value, same point as τ_T_max.
    # Both act in η-direction → ADD (not max).
    tau_V_crank = 1.5 * max(abs(F_t_crank_B), abs(F_t_crank_A)) / A_c
    tau_body_crit = tau_T_offset + tau_V_crank   # combined critical shear at Point 2

    # --- Smallest-area lug shear (Mother Doc Eq 6.13) ---
    denom_B = max(w - D_B_hole, 1e-9) * t
    denom_A = max(w - D_A_hole, 1e-9) * t
    tau_sma_B = abs(F_t_crank_B) / denom_B
    tau_sma_A = abs(F_t_crank_A) / denom_A

    # --- Collect worst-case ---
    # sigma: Point 1 (η=±w/2) → sigma_ax_body_{A,B} = axial + in-plane bending (+ T_in at A)
    #        Point 2 (η=0, ζ=±c_c/2) → sigma_body_B_pt2 = axial + OOP bending
    sigma_crank = max(
        sigma_ax_body_B,
        sigma_ax_body_A,
        sigma_ax_hole_B,
        sigma_ax_hole_A,
        sigma_body_B_pt2,
    )

    # tau: tau_body_crit = τ_T_offset + τ_V summed at Point 2
    #      T_in removed from torsion (reclassified as in-plane bending at section A)
    #      tau_max_hole_* = torsion-dominated at holes with Kt (T_total = T_offset only)
    #      tau_sma_* = net-section lug shear (no 3/2; lug analysis convention)
    tau_crank = max(
        tau_body_crit,
        tau_max_hole_B,
        tau_max_hole_A,
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
    d_shaft_A = design['d_shaft_A']
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
    A_sA = math.pi * d_shaft_A**2 / 4.0   # shaft A (circular cross-section)
    A_pB = math.pi * D_pB**2 / 4.0
    A_pC = math.pi * D_pC**2 / 4.0

    # --- Pin/shaft shear (Mother Doc Section 7.1) ---
    tau_pin_A = mag_A / A_sA               # shaft A — single shear (transverse load at bore)
    tau_pin_B = mag_B / A_pB               # single shear
    tau_pin_C = mag_C / (2.0 * A_pC)      # double shear

    # --- Pin/shaft bending (Mother Doc Section 7.2) ---
    M_pin_A = mag_A * t_crank / 2.0       # shaft bending at bore mid-plane
    M_pin_B = mag_B * t_crank / 2.0
    M_pin_C = mag_C * t_rod   / 4.0

    sigma_b_A = 32.0 * M_pin_A / (math.pi * d_shaft_A**3)
    sigma_b_B = 32.0 * M_pin_B / (math.pi * D_pB**3)
    sigma_b_C = 32.0 * M_pin_C / (math.pi * D_pC**3)

    # --- Bearing stress (Mother Doc Section 7.3) ---
    sigma_br_A       = abs(F_r_crank_A) / (d_shaft_A * t_crank)
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
                   'd_shaft_A'                            — motor shaft diameter (m)
                   'pin_diameter_B', 'pin_diameter_C'    — lug pin diameters (m)
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
