# THE MOTHER DOC

**Fatigue Analysis of an Offset Slider-Crank Mechanism**  
*Rectangular Bar Links (2024-T3 Aluminium) | 2024-T3 Aluminium Pins | Full 2D and 3D Stress States*  
*Version: March 2026*

> **Note:** This is the Markdown export of The Mother Doc v7. All equations reference SI units unless stated otherwise. Sections 4–8 cover stress analysis; Sections 9–13 cover fatigue; Section 14 covers buckling.



---

# 1. Mechanism Geometry and Nomenclature

The mechanism is an offset slider-crank with eccentricity **e > 0**. All coordinates originate at the crank pivot O = (0, 0). The slider translates along the line y = −e. Each link is a flat rectangular bar of 2024-T3 aluminium with cross-section width w (in-plane, η-direction) and thickness t (out-of-plane, ζ-direction). The orientation is fixed as **w > t** for both links throughout the full crank cycle; no reorientation occurs.

## 1.1 Symbol Definitions

| **Symbol** | **Definition**                                                                                           | **Notes**                                               |
|------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| O (Pin A)  | Crank pivot, fixed at (0,0)                                                                              | Ground bearing; motor torque applied here               |
| B (Pin B)  | Crankpin connecting crank arm to rod                                                                     | F_B = [F_Bx, F_By] from Newton-Euler solver acts here |
| C (Pin C)  | Rod-slider joint                                                                                         | Double shear; slider brackets rod on both sides         |
| l_rod      | Connecting rod length                                                                                    | mm                                                      |
| L_crank    | Crank arm length (Pin A to Pin B)                                                                        | mm                                                      |
| e          | Eccentricity (in-plane offset of slider line)                                                            | Slider line at y = −e; e > 0, no upper bound           |
| i_offset   | Out-of-plane offset at Pin B joint = (t_rod + t_crank)/2; source of rod torsion and out-of-plane bending | < 10% of rod length                                    |
| theta      | Crank angle (CCW from +x axis)                                                                           | deg                                                     |
| phi(theta) | Connecting rod orientation angle from +x axis; computed by kinematics.rod_angle(theta, r, l, e)          | deg                                                     |
| omega      | Crank angular velocity (constant)                                                                        | rad/s                                                   |
| w_rod      | Rod in-plane width (η-direction, longer)                                                                 | mm; w_rod > t_rod                                      |
| t_rod      | Rod out-of-plane thickness (ζ-direction, shorter)                                                        | mm                                                      |
| w_crank    | Crank in-plane width                                                                                     | mm; w_crank > t_crank                                  |
| t_crank    | Crank out-of-plane thickness                                                                             | mm                                                      |
| d_p        | Pin diameter (same for all three pins)                                                                   | mm; 2024-T3 aluminium                                   |
| L_p        | Pin length (same for all three pins)                                                                     | mm                                                      |

## 1.2 Pin Configuration

| **Pin** | **Joint**                             | **Shear Type** | **Notes**                                       |
|---------|---------------------------------------|----------------|-------------------------------------------------|
| Pin A   | Ground pivot (crank rotates about it) | Single shear   | Cantilevered into ground bearing at one end     |
| Pin B   | Crank-rod joint                       | Single shear   | Rod and crank lugs side by side; negligible gap |
| Pin C   | Rod-slider joint                      | Double shear   | Rod lug bracketed on both sides by slider body  |

## 1.3 Lug Geometry

All four joints use **round-ended lugs**. The round end is a full-width semicircle: end radius = w/2 for each link. In the mechanism’s notation: rod end radius = w_rod/2; crank end radius = w_crank/2. Round-ended lugs were chosen over square-ended for fatigue superiority (longer shear plane, no tip corner stress raiser).


---

# 2. Force Decomposition at Pin B

All forces are taken directly from the Newton-Euler solver outputs: F_A = [F_Ax, F_Ay] at Pin A, F_B = [F_Bx, F_By] at Pin B, F_C = [F_Cx, F_Cy] at Pin C, and T_in (input torque). Each link sees independent end forces; the rod is loaded by F_B at Pin B and F_C at Pin C; the crank is loaded by F_A at Pin A and F_B at Pin B.

## 2.1 Resultant Force Magnitudes

| **(2.1a)** | F_B(theta) = sqrt[ F_Bx² + F_By² ] | *Resultant force magnitude at Pin B (global frame)* |
|------------|--------------------------------------|-----------------------------------------------------|
| **(2.1b)** | F_C(theta) = sqrt[ F_Cx² + F_Cy² ] | *Resultant force magnitude at Pin C (global frame)* |
| **(2.1c)** | F_A(theta) = sqrt[ F_Ax² + F_Ay² ] | *Resultant force magnitude at Pin A (global frame)* |

## 2.2 Crank Arm Frame Forces

| **(2.3)** | F_t,crank,B(theta) = −F_Bx·sin(theta) + F_By·cos(theta) | *Tangential force at Pin B end of crank (perpendicular to crank arm); from F_B* |
|-----------|---------------------------------------------------------|---------------------------------------------------------------------------------|
| **(2.4)** | F_r,crank,B(theta) = F_Bx·cos(theta) + F_By·sin(theta)  | *Axial force at Pin B end of crank (along crank arm axis); from F_B*            |
| **(2.5)** | F_t,crank,A(theta) = F_Ax·sin(theta) − F_Ay·cos(theta)  | *Tangential force at Pin A end of crank (perpendicular to crank arm); from F_A* |
| **(2.6)** | F_r,crank,A(theta) = −F_Ax·cos(theta) − F_Ay·sin(theta) | *Axial force at Pin A end of crank (along crank arm axis); from F_A*            |

## 2.3 Input Torque at Pin A

| **(2.7)** | T_in(theta) | *Input torque at Pin A; taken directly from Newton-Euler solver output* |
|-----------|-------------|-------------------------------------------------------------------------|

## 2.4 Bearing Reactions at Pin A

| **(2.7)** | F_Ax(theta) | *Bearing reaction at Pin A, x-direction; taken directly from Newton-Euler solver output* |
|-----------|-------------|------------------------------------------------------------------------------------------|
| **(2.8)** | F_Ay(theta) | *Bearing reaction at Pin A, y-direction; taken directly from Newton-Euler solver output* |

## 2.5 Rod Frame Force Components

| **(2.9)**  | F_r,rod,B(theta) = F_Bx·cos(phi(theta)) + F_By·sin(phi(theta))  | *Axial force on rod at Pin B end (+ = tension, − = compression); from F_B*                   |
|------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| **(2.10)** | F_t,rod,B(theta) = −F_Bx·sin(phi(theta)) + F_By·cos(phi(theta)) | *Tangential force on rod at Pin B end (drives bending, torsion, transverse shear); from F_B* |
| **(2.11)** | F_r,rod,C(theta) = −F_Cx·cos(phi(theta)) − F_Cy·sin(phi(theta)) | *Axial force on rod at Pin C end (sign: rod sees −F_C); from F_C*                            |
| **(2.12)** | F_t,rod,C(theta) = F_Cx·sin(phi(theta)) − F_Cy·cos(phi(theta))  | *Tangential force on rod at Pin C end; from F_C*                                             |

> *F_B and F_C are independent Newton-Euler solver outputs. F_r,rod,B and F_t,rod,B feed into Section 4 and 5 equations at the Pin B hole and body. F_r,rod,C and F_t,rod,C feed into Section 4 and 5 equations at the Pin C hole. F_r,crank,B, F_t,crank,B, F_r,crank,A, and F_t,crank,A feed into Section 6.*


---

# 3. Cross-Section Properties (Rectangular Bar w × t)

## 3.1 Area

| **(3.1)** | A_r = w_rod·t_rod     | *Rod cross-sectional area (mm²)*   |
|-----------|-----------------------|------------------------------------|
| **(3.2)** | A_c = w_crank·t_crank | *Crank cross-sectional area (mm²)* |

## 3.2 Second Moment of Area

| **(3.3)** | I_zr = t_rod·w_rod³ / 12           | *Rod, z-axis (in-plane bending, stronger axis)*   |
|-----------|------------------------------------|---------------------------------------------------|
| **(3.4)** | I_yr = w_rod·t_rod³ / 12           | *Rod, y-axis (out-of-plane bending, weak axis)*   |
| **(3.5)** | I_zc = t_crank·w_crank³ / 12       | *Crank, z-axis (in-plane bending)*                |
| **(3.6)** | I_yc = w_crank·t_crank³ / 12       | *Crank, y-axis (out-of-plane bending, weak axis)* |
| **(3.7)** | I_min,r = I_yr = w_rod·t_rod³ / 12 | *Rod, weak axis for buckling check*               |

> *Since w > t: I_zr > I_yr and I_zc > I_yc. Out-of-plane bending is the critical bending mode for both links.*

## 3.3 Extreme Fibre Distances

| **(3.8)** | c_zr = w_rod/2, c_yr = t_rod/2     | *Rod extreme fibre distances*   |
|-----------|------------------------------------|---------------------------------|
| **(3.9)** | c_zc = w_crank/2, c_yc = t_crank/2 | *Crank extreme fibre distances* |

## 3.4 First Moment of Area (Q)

Derived from Q = A̅·ȳ: for a rectangle of width b and depth h cut at the neutral axis, Q_max = b·(h/2)·(h/4) = b·h²/8.

| **(3.10)** | Q_zr,max = w_rod·t_rod² / 8     | *Rod, first moment at neutral axis (z-axis)*   |
|------------|---------------------------------|------------------------------------------------|
| **(3.11)** | Q_zc,max = w_crank·t_crank² / 8 | *Crank, first moment at neutral axis (z-axis)* |


---

# 4. Two-Dimensional Stress Analysis (Rod)

For in-plane loading only. Let ξ ∈ [0, l_rod] be the position along the rod from Pin B (ξ = 0) to Pin C (ξ = l_rod).

## 4.1 Bending Moment Along the Rod (Gravity Load)

The rod is a pin-pin beam. Torsion at each end is the only moment; the bending moment is zero at both pin holes (ξ = 0 and ξ = l_rod) by the pin-pin boundary condition. Bending moment arises from the distributed gravitational load, which varies with the instantaneous angle phi(theta) of the rod.

| **(4.1)** | w_rod,g(theta) = (m_rod·g / l_rod)·cos(phi(theta)) | *Gravity distributed load per unit length on the rod; phi(theta) is the instantaneous inclination of the rod from horizontal; g = 9810 mm/s²* |
|-----------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **(4.2)** | R_C,rod(theta) = w_rod,g(theta)·l_rod / 2          | *Transverse pin reaction at Pin C (by symmetry of uniform load on pin-pin beam)*                                                              |
| **(4.3)** | R_B,rod(theta) = w_rod,g(theta)·l_rod / 2          | *Transverse pin reaction at Pin B (by symmetry of uniform load on pin-pin beam)*                                                              |
| **(4.4)** | M_rod(ξ, theta) = w_rod,g(theta)·ξ·(l_rod − ξ) / 2 | *Parabolic bending moment distribution; M = 0 at ξ = 0 and ξ = l_rod (pin-pin boundary condition satisfied)*                                  |
| **(4.5)** | M_rod,max(theta) = w_rod,g(theta)·l_rod² / 8       | *Peak bending moment at ξ = l_rod/2 (midspan of the rod)*                                                                                     |

## 4.2 Normal Stress

| **(4.6)** | σ_ax,rod,body,B(theta) = F_r,rod,B(theta)/A_r ± M_rod,max(theta)·c_zr/I_zr; σ_ax,rod,body,C(theta) = F_r,rod,C(theta)/A_r ± M_rod,max(theta)·c_zr/I_zr                             | *Peak normal stress at critical body section; B-end uses F_r,rod,B, C-end uses F_r,rod,C; take governing (larger) value*                                                                                                                             |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(4.7)** | σ_ax,rod,hole,B(theta) = Kt_fixed·F_r,rod,B(theta)/Z_r,B; σ_ax,rod,hole,C(theta) = Kt_fixed·F_r,rod,C(theta)/Z_r,C; where Z_r,B = (w_rod − D_B)·t_rod; Z_r,C = (w_rod − D_C)·t_rod | *Peak normal stress at hole: use F_r,rod,B at Pin B hole, F_r,rod,C at Pin C hole; M = 0 at holes by pin-pin condition; Kt_fixed = 2.34 applied to net-section area Z_r (Peterson Chart 5.12 not used — applies to double-shear configuration only)* |

> *Kt_u2 (Peterson 4.8.1, bending at hole) evaluates to zero at both hole locations in the current pin-pin model. It is retained as a conservative documented bound only. See Section 11.2.*

## 4.3 Transverse Shear Stress

| **(4.8)** | τ_xy,max,B(theta) = 3·F_t,rod,B(theta)/(2·w_rod·t_rod); τ_xy,max,C(theta) = 3·F_t,rod,C(theta)/(2·w_rod·t_rod) | *Peak transverse shear at neutral axis (simplified from Q formula); evaluate at both ends and take governing value*                                                                          |
|-----------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(4.9)** | τ_sma,B,rod = F_t,rod,B(theta) / ((w_rod − D_B)·t_rod); τ_sma,C,rod = F_t,rod,C(theta) / ((w_rod − D_C)·t_rod) | *Shear stress at the smallest lug cross-section due to transverse force; single shear plane on the side the pin pushes toward; not shear tear-out. Pin B hole uses D_B; Pin C hole uses D_C* |

## 4.4 Principal Stresses and Von Mises (2D)

| **(4.10)** | σ₁₂ = σ_xx/2 ± sqrt[ (σ_xx/2)² + τ_xy² ] | *In-plane principal stresses*                |
|------------|--------------------------------------------|----------------------------------------------|
| **(4.11)** | τ_max = sqrt[ (σ_xx/2)² + τ_xy² ]        | *Maximum shear stress (2D)*                  |
| **(4.12)** | σ_VM = sqrt[ σ_xx² + 3·τ_xy² ]           | *Von Mises equivalent stress (2D, σ_yy = 0)* |


---

# 5. Three-Dimensional Stress Analysis (Rod)

Applicable because of the out-of-plane spatial offset i_offset between the crank plane and slider plane (< 10% of rod length). The rod carries axial force, in-plane bending, out-of-plane bending from eccentric loading at Pin B, and torsion simultaneously. Coordinate ξ ∈ [0, l_rod] measured from Pin B.

## 5.1 Normal Stress (3D, Biaxial Bending)

At a point (η, ζ) in the cross-section (η ∈ [−w/2, +w/2], ζ ∈ [−t/2, +t/2]):

| **(5.1)** | σ_ξξ,B = F_r,rod,B(theta)/(w_rod·t_rod) + M_ζ(ξ,theta)·ζ/I_zr + M_η(ξ,theta)·η/I_yr; σ_ξξ,C = F_r,rod,C(theta)/(w_rod·t_rod) + M_ζ(ξ,theta)·ζ/I_zr + M_η(ξ,theta)·η/I_yr | *Axial + biaxial bending normal stress*             |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| **(5.2)** | σ_ξξ,max,B = F_r,rod,B/(w_rod·t_rod) ± 6·M_ζ/(w_rod·t_rod²) ± 6·M_η/(t_rod·w_rod²); σ_ξξ,max,C = F_r,rod,C/(w_rod·t_rod) ± 6·M_ζ/(w_rod·t_rod²) ± 6·M_η/(t_rod·w_rod²)   | *Peak 3D normal stress at corners (η=±w/2, ζ=±t/2)* |

## 5.2 In-Plane Bending Moment M_ζ

| **(5.3)** | M_ζ(ξ, theta) = M_rod(ξ, theta) | *In-plane bending moment; same as Eq. (4.4)* |
|-----------|---------------------------------|----------------------------------------------|

## 5.3 Out-of-Plane Bending at Pin B

The axial force F_r,rod transmitted through Pin B acts at an offset i_offset = (t_rod + t_crank)/2 from the rod’s neutral axis in the ζ direction, producing an out-of-plane bending moment at the Pin B cross-section. No stress concentration factor applies here since Pin B is a plain rectangular cross-section adjacent to the hole, not the hole itself.

| **(5.4)** | M_η,rod,B(theta) = F_r,rod,B(theta)·i_offset                                                   | *Out-of-plane bending moment at Pin B from eccentric axial force; uses F_r,rod,B only (offset is a Pin B geometry)* |
|-----------|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **(5.5)** | σ_oop,rod,B(theta) = M_η,rod,B(theta)·c_yr / I_yr = F_r,rod,B(theta)·i_offset·(t_rod/2) / I_yr | *Out-of-plane bending stress at Pin B (plain cross-section)*                                                        |

> *Out-of-plane bending at Pin C of the rod is neglected.*

## 5.4 Torsional Moment on the Rod

The transverse force F_t,rod,C acts at Pin C with moment arm i_offset from the rod’s neutral axis. This generates a twisting moment along the rod’s longitudinal axis. The Pin C end force is used because torsion is generated by the force acting at the slider end of the rod.

| **(5.6)** | T_rod(theta) = F_t,rod,C(theta)·i_offset | *Torsional moment on rod about its own axis* |
|-----------|------------------------------------------|----------------------------------------------|

## 5.5 Torsional Shear Stress (Saint-Venant, Rectangular Bar)

**CRITICAL:** Do NOT use τ = Tc/J with J = w·t·(w²+t²)/12. That formula applies only to circular sections. Use Saint-Venant formulae below.

| **(5.7)** | β_r = 1/3 − 0.21·(t_rod/w_rod)·[1 − t_rod⁴/(12·w_rod⁴)] | *Saint-Venant torsion coefficient for rod (Roark approximation; w_rod > t_rod; less accurate for w_rod/t_rod close to 1)* |
|-----------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **(5.8)** | τ_T,rod(theta) = T_rod(theta) / (β_r·w_rod²·t_rod)        | *Peak torsional shear on plain body (short face, ζ = ±t_rod/2); governs over long face since w > t*                       |

## 5.6 Torsional Shear at the Hole Cross-Section

At the pin hole cross-section, the hole removes material and reduces the torsional constant. The nominal torsional shear stress at the hole is approximated using the area reduction ratio as a proxy for torsional stiffness:

| **(5.9)**  | τ_nom,hole,rod,B(theta) = T_rod(theta) / [β_r·w_rod²·t_rod·(1 − π·D_B²/(4·w_rod·t_rod))]; τ_nom,hole,rod,C(theta) = T_rod(theta) / [β_r·w_rod²·t_rod·(1 − π·D_C²/(4·w_rod·t_rod))] | *Nominal torsional shear at rod hole cross-section. Valid for D_B/w_rod < 0.3 (Pin B) and D_C/w_rod < 0.3 (Pin C). D_B, D_C are hole diameters = D_pB + δ_B, D_pC + δ_C.* |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(5.10)** | τ_max,hole,rod(theta) = Kt_u1·τ_nom,hole,rod(theta)                                                                                                                                    | *Peak torsional shear at hole edge; Kt_u1 = 4 from Peterson Section 4.9.1*                                                                                                  |

> **CAUTION:** Kt_u1 = 4 is from Peterson Section 4.9.1 (open hole, infinite plate, pure shear). It is a conservative upper bound for the current pin-loaded finite-width lug geometry.
>
> The round end geometry and finite bar width reduce the effective Kt below 4. Use as a worst-case bound only.

## 5.7 Combined Shear and Von Mises (3D)

| **(5.11)** | τ_ξη,total,B = F_t,rod,B(theta)·Q_zr,max / (I_zr·w_rod); τ_ξη,total,C = F_t,rod,C(theta)·Q_zr,max / (I_zr·w_rod) | *Transverse shear on long face (η-direction); long-face torsion removed (non-governing)*                                   |
|------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **(5.12)** | τ_ξζ,total = τ_T,rod(theta)                                                                                      | *Combined shear on short face (ζ-direction); transverse shear in ζ-direction assumed zero (KT = 1.0 pending verification)* |
| **(5.13)** | σ_VM = sqrt[ σ_ξξ² + 3(τ_ξη,total² + τ_ξζ,total²) ]                                                            | *3D Von Mises for slender links (σ_ηη = σ_ζζ = 0)*                                                                         |


---

# 6. Crank Arm Stress Analysis

The crank arm is a flat rectangular bar (w_crank × t_crank) between Pin A (ξ = 0) and Pin B (ξ = L_crank). Stresses are expressed using F_r,crank,A(theta), F_t,crank,A(theta) at the Pin A end and F_r,crank,B(theta), F_t,crank,B(theta) at the Pin B end, from Eqs. (2.3)–(2.6).

## 6.1 In-Plane Bending Moment (Gravity Load)

The crank is a pin-pin beam. Torsion at each end is the only moment; the bending moment is zero at both pin holes (ξ = 0 and ξ = L_crank) by the pin-pin boundary condition. Bending moment arises from the distributed gravitational load, which varies with the instantaneous crank angle theta.

| **(6.1)** | w_crank,g(theta) = (m_crank·g / L_crank)·cos(theta)      | *Gravity distributed load per unit length on the crank; theta is the crank angle measured from horizontal; g = 9810 mm/s²* |
|-----------|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **(6.2)** | R_B,crank(theta) = w_crank,g(theta)·L_crank / 2          | *Transverse pin reaction at Pin B (by symmetry of uniform load on pin-pin beam)*                                           |
| **(6.3)** | R_A,crank(theta) = w_crank,g(theta)·L_crank / 2          | *Transverse pin reaction at Pin A (by symmetry of uniform load on pin-pin beam)*                                           |
| **(6.4)** | M_crank(ξ, theta) = w_crank,g(theta)·ξ·(L_crank − ξ) / 2 | *Parabolic bending moment distribution; M = 0 at ξ = 0 and ξ = L_crank (pin-pin boundary condition satisfied)*             |
| **(6.5)** | M_crank,max(theta) = w_crank,g(theta)·L_crank² / 8       | *Peak bending moment at ξ = L_crank/2 (midspan of the crank)*                                                              |

## 6.2 Axial Stress

| **(6.6)** | σ_ax,crank,B(theta) = F_r,crank,B(theta)/(w_crank·t_crank); σ_ax,crank,A(theta) = F_r,crank,A(theta)/(w_crank·t_crank) | *Axial stress in crank arm; evaluate at both ends and take governing value* |
|-----------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|

## 6.3 Normal Stress at Critical Body Section

| **(6.7)** | σ_ax,crank,body,B(theta) = F_r,crank,B(theta)/A_c ± M_crank,max(theta)·c_zc/I_zc; σ_ax,crank,body,A(theta) = F_r,crank,A(theta)/A_c ± M_crank,max(theta)·c_zc/I_zc                                 | *Peak normal stress at critical body section; B-end uses F_r,crank,B, A-end uses F_r,crank,A; take governing value*         |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **(6.8)** | σ_ax,crank,hole,B(theta) = Kt_fixed·F_r,crank,B(theta)/Z_c,B; σ_ax,crank,hole,A(theta) = Kt_fixed·F_r,crank,A(theta)/Z_c,A; where Z_c,B = (w_crank − D_B)·t_crank; Z_c,A = (w_crank − D_A)·t_crank | *Peak normal stress at hole: use F_r,crank,B at Pin B hole, F_r,crank,A at Pin A hole; M = 0 at holes by pin-pin condition* |

## 6.4 Out-of-Plane Bending at Pin B

The axial force F_r,crank,B at Pin B acts at an offset i_offset = (t_rod + t_crank)/2 from the crank arm neutral axis in the ζ direction. Out-of-plane bending at Pin A is neglected (reacted by ground bearing). F_r,crank,B is used here because the offset geometry is specific to the Pin B joint.

| **(6.9)**  | M_η,crank,B(theta) = F_r,crank,B(theta)·i_offset                      | *Out-of-plane bending moment at Pin B*                               |
|------------|-----------------------------------------------------------------------|----------------------------------------------------------------------|
| **(6.10)** | σ_oop,crank,B(theta) = F_r,crank,B(theta)·i_offset·(t_crank/2) / I_yc | *Out-of-plane bending stress at Pin B (plain cross-section; no SCF)* |

## 6.5 Transverse Shear Stress

| **(6.11)** | τ_xy,crank,B(theta) = F_t,crank,B(theta)·Q_zc,max/(I_zc·w_crank); τ_xy,crank,A(theta) = F_t,crank,A(theta)·Q_zc,max/(I_zc·w_crank)     | *Transverse shear at neutral axis (general form); evaluate at both ends and take governing value*                                                                                            |
|------------|----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(6.12)** | τ_xy,max,crank,B(theta) = 3·F_t,crank,B(theta)/(2·w_crank·t_crank); τ_xy,max,crank,A(theta) = 3·F_t,crank,A(theta)/(2·w_crank·t_crank) | *Peak transverse shear (simplified)*                                                                                                                                                         |
| **(6.13)** | τ_sma,B,crank = F_t,crank,B(theta) / ((w_crank − D_B)·t_crank); τ_sma,A,crank = F_t,crank,A(theta) / ((w_crank − D_A)·t_crank)         | *Shear stress at the smallest lug cross-section due to transverse force; single shear plane on the side the pin pushes toward; not shear tear-out. Pin B hole uses D_B; Pin A hole uses D_A* |

## 6.6 Torsional Shear — T_in (From Pin A Side Cut)

| **(6.14)** | T_in(theta)                                          | *Motor input torque at Pin A; taken directly from Newton-Euler solver output . This is NOT Saint-Venant torsion of the crank arm about its own axis.* |
|------------|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(6.15)** | τ_T,in(theta) = T_in(theta) / (β_c·w_crank·t_crank²) | *Saint-Venant torsional shear from T_in on short face (t_crank face); β_c = 1/3 − 0.21·(t_crank/w_crank)·[1 − t_crank⁴/(12·w_crank⁴)]*              |

## 6.7 Torsional Shear — T_offset (From Link Thickness Offset at Pin B)

The transverse force F_t,rod,B at Pin B acts at offset i_offset = (t_rod + t_crank)/2 from the crank arm neutral axis, generating a torque about the crank arm’s own longitudinal axis. The Pin B end force is used because the offset geometry is specific to the Pin B joint.

| **(6.16)** | T_offset(theta) = F_t,rod,B(theta)·i_offset                  | *Torsion of crank arm from link thickness offset at Pin B*                                                         |
|------------|--------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **(6.17)** | τ_T,offset(theta) = T_offset(theta) / (β_c·w_crank·t_crank²) | *Saint-Venant torsional shear from T_offset on short face; same denominator as Eq. (6.14); only numerator differs* |

> *β_c appears in both Eqs. (6.14) and (6.16) with the same denominator. This is intentional: both torsion sources act about the same crank arm axis; only the torque magnitude differs.*

## 6.8 Torsional Shear at Crank Hole Cross-Sections

| **(6.18)** | τ_nom,hole,crank,B(theta) = T / [β_c·w_crank·t_crank²·(1 − π·D_B²/(4·w_crank·t_crank))]; τ_nom,hole,crank,A(theta) = T / [β_c·w_crank·t_crank²·(1 − π·D_A²/(4·w_crank·t_crank))] | *Nominal torsional shear at crank hole cross-section (valid for D_B/w_crank < 0.3 at Pin B, D_A/w_crank < 0.3 at Pin A). D_A, D_B are hole diameters = D_pA + δ_A, D_pB + δ_B. T = T_in or T_offset as appropriate.* |
|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(6.19)** | τ_max,hole,crank(theta) = Kt_u1·τ_nom,hole,crank(theta)                                                                                                                              | *Peak torsional shear at crank hole edge; Kt_u1 = 4 from Peterson 4.9.1 (conservative upper bound)*                                                                                                                    |

## 6.9 Combined Von Mises (Crank Arm)

| **(6.20)** | σ_VM,crank(theta) = sqrt[ (σ_ax,crank,body,B + σ_ax,crank,B)² + 3·(τ_T,in + τ_T,offset)² ] | *Combined Von Mises at critical section*       |
|------------|----------------------------------------------------------------------------------------------|------------------------------------------------|
| **(6.21)** | M_max = max_theta[ M_crank,max(theta) ] over theta = 0° to 360°                            | *Maximum bending moment over full crank cycle* |


---

# 7. Pin Stress Analysis

All pins: solid cylinder, length L_p, 2024-T3 aluminium (same material as links). Gap between lugs is negligible. Pin diameters: D_pA (Pin A), D_pB (Pin B), D_pC (Pin C).

## 7.1 Pin Shear Stress

| **(7.1)** | τ_pin,A(theta) = F_A(theta) / (A_pin,A) = 4·F_A(theta) / (π·D_pA²)   | *Pin A shear stress (single shear)*                                        |
|-----------|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| **(7.2)** | τ_pin,B(theta) = F_B(theta) / (A_pin,B) = 4·F_B(theta) / (π·D_pB²)   | *Pin B shear stress (single shear)*                                        |
| **(7.3)** | τ_pin,C(theta) = F_C(theta) / (2·A_pin,C) = 2·F_C(theta) / (π·D_pC²) | *Pin C shear stress (double shear; force shared between two shear planes)* |

## 7.2 Pin Bending Stress

| **(7.4)** | M_pin,A(theta) = |F_A(theta)|·t_crank/2                                                            | *Pin A bending moment (cantilever; crank lug bears at t_crank/2 from fixed end)* |
|-----------|------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| **(7.5)** | M_pin,B(theta) = |F_B(theta)|·t_crank/2                                                            | *Pin B bending moment (single shear; maximum at rod-crank lug interface)*        |
| **(7.6)** | M_pin,C(theta) = |F_C(theta)|·t_rod/4                                                              | *Pin C bending moment (double shear; maximum at centre of rod lug)*              |
| **(7.7)** | σ_b,pin(theta) = 32·M_pin(theta) / (π·D_p³); use D_pA, D_pB, or D_pC for Pin A, B, or C respectively | *Peak pin bending stress (applies to A, B, or C with respective M_pin)*          |

## 7.3 Bearing Stress at Pin Holes

| **(7.8)**  | σ_br,A(theta) = F_A(theta) / (D_pA·t_crank)       | *Bearing stress at Pin A (crank lug)*                                                                          |
|------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| **(7.9)**  | σ_br,B,rod(theta) = F_B(theta) / (D_pB·t_rod)     | *Bearing stress at Pin B, rod lug side*                                                                        |
| **(7.10)** | σ_br,B,crank(theta) = F_B(theta) / (D_pB·t_crank) | *Bearing stress at Pin B, crank lug side*                                                                      |
| **(7.11)** | σ_br,C(theta) = F_C(theta) / (2·D_pC·t_rod)       | *Bearing stress at Pin C (double shear; force shared equally between two slider lugs each of thickness t_rod)* |


---

# 8. Lug Joint Stress Concentration

## 8.1 Stress Concentration at Pin Hole Perimeter

A fixed conservative stress concentration factor Kt_fixed = 2.34 is applied to the net-section area at each pin hole. Peterson Chart 5.12 (Section 5.8) is not used because it is calibrated for double-shear lug configurations; this mechanism uses single shear at Pins A and B and double shear at Pin C. Kt_fixed = 2.34 is a conservative engineering estimate pending a validated single-shear lug SCF formula. Eqs 8.1 to 8.3 below are retained for reference but are not active in the current stress computation.

| **(8.1)** | e%(theta) = (δ / D_p) × 100; use D_pA, D_pB, or D_pC for Pin A, B, or C respectively; δ = diametral clearance at that pin | *Pin clearance as percentage of hole diameter; δ = diametral clearance*                                                                                                                                             |
|-----------|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(8.2)** | Kte = Kt0.2 + f·(Kt100 − Kt0.2) [NOT CURRENTLY ACTIVE — double-shear chart only]                                        | *Retained for reference. Not active: Peterson Chart 5.12 applies to double-shear configurations only. Kt_fixed = 2.34 is used instead in Eqs 4.7 and 6.8.*                                                          |
| **(8.3)** | Kte* = (Kte*/Kte)·Kte [NOT CURRENTLY ACTIVE — double-shear chart only]                                                | *Retained for reference. Not active: thick lug correction from Peterson Chart 5.13 applies to double-shear configurations only. Kt_fixed = 2.34 covers both thin and thick lugs as a single conservative estimate.* |

> *No closed-form formula exists for Peterson Charts 5.12 and 5.13. Recommended approach: digitize with WebPlotDigitizer and fit 3rd/4th degree polynomials per curve (accuracy within 2–3%). Heywood/Frocht-Hill fallback (conservative, c/H > 1.5 only): Kt = (3 − d/W)/(1 − d/W).*

## 8.2 Net-Section Stress Concentration (Peterson Section 4.5.8)

Section 4.5.8 governs the tensile failure path across the hole diameter perpendicular to the load.

| **(8.4)** | σ_nd(theta) = F_r,rod,B(theta) / Z_r,B (rod, Pin B); F_r,rod,C(theta) / Z_r,C (rod, Pin C); F_r,crank,B(theta) / Z_c,B (crank, Pin B); F_r,crank,A(theta) / Z_c,A (crank, Pin A)                                     | *Nominal net-section stress at pin hole; axial force component used (not resultant); Z_r,B = (w_rod − D_B)·t_rod; Z_r,C = (w_rod − D_C)·t_rod; Z_c,B = (w_crank − D_B)·t_crank; Z_c,A = (w_crank − D_A)·t_crank* |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(8.5)** | Kt_nd = σ_max / σ_nd                                                                                                                                                                                                 | *SCF normalized to net-section stress (Chart 4.67)*                                                                                                                                                              |
| **(8.6)** | σ_nb,A(theta) = F_r,crank,A(theta) / (D_pA·t_crank); σ_nb,B,rod(theta) = F_r,rod,B(theta) / (D_pB·t_rod); σ_nb,B,crank(theta) = F_r,crank,B(theta) / (D_pB·t_crank); σ_nb,C(theta) = F_r,rod,C(theta) / (D_pC·t_rod) | *Nominal bearing area stress; pin diameter used (not hole diameter) since bearing acts on pin surface; each pin and lug face has its own equation consistent with Section 7 bearing stress convention*           |
| **(8.7)** | Kt_nb = σ_max / σ_nb                                                                                                                                                                                                 | *SCF normalized to bearing area stress (Chart 4.67); generally used when d/H < 1/4*                                                                                                                             |

> *Clearance increases Kt_nb significantly. At d/H = 0.15, clearances of 0.7%, 1.3%, and 2.7% give Kt_nb of approximately 1.1, 1.3, and 1.8 respectively. For the active hole stress model use Kt_fixed = 2.34 with Z_r or Z_c (Eqs 4.7 and 6.8).*

## 8.3 Shear Tear-Out

For a round-ended lug with end radius = w/2, the distance from the hole centre to the lug boundary is w/2 in every direction. Therefore R = w/2 is constant for all theta. Z = 0 for a full-width semicircular end (no loss in shear plane length due to curvature).

| **(8.8)**  | a = w/2 − D_A/2 (Pin A), w/2 − D_B/2 (Pin B), or w/2 − D_C/2 (Pin C); substitute hole diameter at the pin being checked                                                                                                    | *Edge distance beyond hole*                                                                                                                                                                                     |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **(8.9)**  | Lsp = w/2 − D_A/2 + (D_A/2)·tan(40°) (Pin A); w/2 − D_B/2 + (D_B/2)·tan(40°) (Pin B); w/2 − D_C/2 + (D_C/2)·tan(40°) (Pin C); substitute hole diameter at the pin being checked                                            | *Shear plane length (40° refined method); valid for round-ended lug with R = w/2*                                                                                                                               |
| **(8.10)** | As = 2·Lsp·t                                                                                                                                                                                                               | *Total shear area; t = t_rod (rod lug) or t_crank (crank lug)*                                                                                                                                                  |
| **(8.11)** | Psu = Ssu·As                                                                                                                                                                                                               | *Ultimate shear load; Ssu ≈ 0.6·Sut for 2024-T3*                                                                                                                                                                |
| **(8.12)** | FS_tear,A(theta) = Psu,A / |F_r,crank,A(theta)|; FS_tear,B,rod(theta) = Psu,B,rod / |F_r,rod,B(theta)|; FS_tear,B,crank(theta) = Psu,B,crank / |F_r,crank,B(theta)|; FS_tear,C(theta) = Psu,C / |F_r,rod,C(theta)| | *Factor of safety against shear tear-out at each pin hole; axial force component used; Psu computed separately per pin using corresponding w and t; critical theta is where axial force is maximum at each pin* |

> *Shear tear-out must be checked at all four pin holes: Pin A, Pin B rod side, Pin B crank side, and Pin C.*


---

# 9. Stress Cycling: Mean and Alternating Components

| **(9.1)** | σ_max = max_theta[ σ(theta) ] over theta = 0° to 360° | *Maximum stress over one revolution*       |
|-----------|---------------------------------------------------------|--------------------------------------------|
| **(9.2)** | σ_min = min_theta[ σ(theta) ] over theta = 0° to 360° | *Minimum stress over one revolution*       |
| **(9.3)** | σ_m = (σ_max + σ_min) / 2                               | *Mean (midrange) stress*                   |
| **(9.4)** | σ_a = (σ_max − σ_min) / 2                               | *Alternating (amplitude) stress*           |
| **(9.5)** | R = σ_min / σ_max                                       | *Stress ratio (R = −1 for fully reversed)* |
| **(9.6)** | σ_a,eq = sqrt[ σ_a² + 3·τ_a² ]                        | *Von Mises equivalent alternating stress*  |
| **(9.7)** | σ_m,eq = sqrt[ σ_m² + 3·τ_m² ]                        | *Von Mises equivalent mean stress*         |

> *For an offset slider-crank (e > 0), R ≠ −1 and σ_m ≠ 0 at most crank angles. This is captured automatically when F_r,rod(theta) and F_t,rod(theta) are evaluated at every degree of theta.*


---

# 10. Endurance Limit and Marin Correction Factors

2024-T3 aluminium has no true endurance limit. The reference fatigue strength at 10⁸ cycles is used as the practical fatigue limit.

## 10.1 Material Reference Values (2024-T3 Aluminium)

| **Property**                       | **Symbol** | **Value**          | **Units** |
|------------------------------------|------------|--------------------|-----------|
| Ultimate tensile strength          | S_ut       | 483                | MPa       |
| Yield strength (0.2% offset)       | S_y        | 345                | MPa       |
| Ref. fatigue strength (10⁸ cycles) | S’_e      | 130                | MPa       |
| Elastic modulus                    | E          | 73.1               | GPa       |
| Shear modulus                      | G          | 28                 | GPa       |
| Poisson’s ratio                    | ν          | 0.33               | —         |
| Density                            | ρ          | 2780               | kg/m³     |
| Fatigue strength coefficient       | σ’_f      | ≈1.67 × S_ut ≈ 807 | MPa       |

## 10.2 Marin Correction Factors

| **(10.1)**          | S_e = k_a·k_b·k_c·k_d·k_e·k_f·S’_e                                            |            |                             | *Corrected endurance limit*                              |                                            |
|---------------------|--------------------------------------------------------------------------------|------------|-----------------------------|----------------------------------------------------------|--------------------------------------------|
| **(10.2)**          | d_e = 0.808·sqrt(w_rod·t_rod) [rod] or 0.808·sqrt(w_crank·t_crank) [crank] |            |                             | *Equivalent diameter for rectangular section in bending* |                                            |
| **(10.3)**          | A_95 = 0.05·w_rod·t_rod [rod] or 0.05·w_crank·t_crank [crank]              |            |                             | *95% stressed area for rectangular section*              |                                            |
| **Factor**          |                                                                                | **Symbol** | **Formula / Value**         |                                                          | **Notes**                                  |
| Surface (machined)  |                                                                                | k_a        | 4.51 × S_ut^(−0.265) ≈ 0.82 |                                                          | S_ut in MPa                                |
| Size (8–51 mm)      |                                                                                | k_b        | 0.879·d_e^(−0.107)          |                                                          | d_e from Eq. (10.2)                        |
| Size (51–254 mm)    |                                                                                | k_b        | 1.24·d_e^(−0.107)           |                                                          | Choose based on d_e range                  |
| Load (bending)      |                                                                                | k_c        | 1.0                         |                                                          | Use 1.0; apply Von Mises separately        |
| Load (axial)        |                                                                                | k_c        | 0.85                        |                                                          | —                                          |
| Temperature (≤70°C) |                                                                                | k_d        | 1.0                         |                                                          | —                                          |
| Reliability (99%)   |                                                                                | k_e        | 0.814                       |                                                          | k_e = 1 − 0.08·z_a                         |
| Reliability (99.9%) |                                                                                | k_e        | 0.753                       |                                                          | Recommended for fatigue-critical aluminium |
| Miscellaneous       |                                                                                | k_f        | 1.0 (no fretting)           |                                                          | Reduce for press-fit bushings              |


---

# 11. Stress Concentration Factors

## 11.1 Summary of Stress Concentration Factors

| **Symbol**      | **Source**                                                  | **Applied to**                                                   | **Status**                                           |
|-----------------|-------------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------|
| Kt_fixed = 2.34 | Conservative fixed estimate (single-shear); see Section 8.1 | Normal stress at hole perimeter; both σ_a and σ_m                | Exact for pin-loaded lug                             |
| Kt_nd / Kt_nb   | Peterson 4.5.8 Chart 4.67                                   | Net-section tensile stress across hole                           | Exact for closely fitting pin                        |
| Kt_u1 = 4       | Peterson 4.9.1 Chart 4.97                                   | Torsional shear at hole (conservative upper bound)               | Conservative; see Section 11.2                       |
| Kt_u2           | Peterson 4.8.1 Chart 4.88                                   | Bending at hole from transverse force (conservative upper bound) | Evaluates to zero in current model; see Section 11.2 |
| KT = 1.0        | Assumed                                                     | Transverse shear at hole                                         | Pending verification                                 |

## 11.2 Conservative Approximations (Open Items)

### Kt_u2: Bending at the Pin Hole (Peterson Section 4.8.1)

Provides Ktg and Ktn for a flat rectangular beam with an open circular hole under in-plane bending (Chart 4.88):

| **(11.1)** | Ktg = σ_max / [6M / (H²·h)]          | *Gross section SCF; H = beam depth, h = thickness, M = bending moment* |
|------------|----------------------------------------|------------------------------------------------------------------------|
| **(11.2)** | Ktn = σ_max / [6M·d / ((H³ − d³)·h)] | *Net section SCF; d = hole diameter*                                   |

> **WARNING:** 4.8.1 models an open unloaded hole in a beam body. Pin contact changes the stress distribution significantly.
>
> The hole is at the round lug end, not in the beam body, so the local geometry differs from the 4.8.1 model.
>
> ADDITIONAL CAUTION: In the current pin-pin model, M = 0 at both hole locations (ξ = 0 and ξ = l) by boundary condition. Kt_u2 therefore evaluates to zero and does not contribute to hole stress. It is retained for completeness only. Do not apply without verifying that a bending moment actually exists at the hole.

### Kt_u1 = 4: Torsional Shear at the Pin Hole (Peterson Section 4.9.1)

> **WARNING:** 4.9.1 models an open hole in pure shear on an infinite plate. Kt = 4 is a worst-case upper bound.
>
> The pin contact, finite bar width, and round end geometry all reduce the effective Kt below 4.
>
> Use as a conservative bound only. The value may be revised if a more specific source is identified.


---

# 12. Fatigue Failure Criteria

Each stress component at the hole carries its own SCF (see Section 11). The Von Mises equivalent alternating and mean stresses σ_a,eq and σ_m,eq are computed with per-component factors applied before the Von Mises combination.

## 12.1 Modified Goodman Criterion

| **(12.1)** | σ_a,eq/S_e + σ_m,eq/S_ut = 1/n_f         | *Modified Goodman equation*       |
|------------|------------------------------------------|-----------------------------------|
| **(12.2)** | n_f = 1 / [ σ_a,eq/S_e + σ_m,eq/S_ut ] | *Fatigue safety factor (Goodman)* |

## 12.2 ECY (Early Cycle Yielding) Line

| **(12.3)** | σ_a,eq + σ_m,eq = S_y / n_y   | *ECY (Early Cycle Yielding) line*                   |
|------------|-------------------------------|-----------------------------------------------------|
| **(12.4)** | n_y = S_y / (σ_a,eq + σ_m,eq) | *Static yield safety factor*                        |
| **(12.5)** | n = min[ n_f, n_y ]         | *Governing (minimum) safety factor at each section* |

> *Both criteria must hold simultaneously. Goodman prevents fatigue fracture; ECY prevents yielding on first load application. n_f and n_y are computed independently.*


---

# 13. Finite Life Analysis (S-N Curve, Basquin Relation)

| **(13.1)** | σ’_a = σ’_f·(2N_f)^b_B                      | *Basquin power law (log-log linear S-N curve)* |
|------------|-----------------------------------------------|------------------------------------------------|
| **(13.2)** | σ’_f ≈ 1.67·S_ut = 807 MPa                   | *Fatigue strength coefficient (2024-T3)*       |
| **(13.3)** | b_B = −[ log(0.9·S_ut / S_e) ] / log(2×10⁶) | *Basquin exponent (two-point method)*          |
| **(13.4)** | N_f(ξ) = (1/2)·[σ’_a(ξ) / σ’_f]^(1/b_B)   | *Cycles to failure at section ξ*               |
| **(13.5)** | t_f(ξ) = N_f(ξ) / (n_rpm / 60)                | *Life in seconds (1 cycle per revolution)*     |
| **(13.6)** | D(ξ) = Σ_i [ n_i / N_{f,i}(ξ) ]            | *Miner’s rule cumulative damage sum*           |
| **(13.7)** | Failure when D(ξ) = 1                         | *Failure criterion*                            |


---

# 14. Buckling Check (Connecting Rod Under Compression)

The connecting rod is in compression when F_r,rod(theta) < 0 (Eq. 2.9). Euler buckling must be checked about the weaker axis at those crank angles.

| **(14.1)** | P_cr = π²·E·I_min,r / (K_c·l_rod)²                               | *Euler critical buckling load*                                                     |
|------------|------------------------------------------------------------------|------------------------------------------------------------------------------------|
| **(14.2)** | I_min,r = I_yr = w_rod·t_rod³ / 12                               | *Second moment about weak axis; buckling occurs about y-axis since w_rod > t_rod* |
| **(14.3)** | K_c = 1.0 (pin-pin ends, both free to rotate)                    | *Effective length factor*                                                          |
| **(14.4)** | N_max,comp = max_theta[ |F_r,rod(theta)| ] when F_r,rod < 0 | *Maximum compressive axial force from Eq. (2.9)*                                   |
| **(14.5)** | n_buck = P_cr / N_max,comp ≥ 3.0                                 | *Buckling safety factor (target ≥ 3.0 for machinery)*                              |


---

# 15. Consolidated Design Check Table

| **Check**                  | **Criterion**                        | **Target**   | **Equation**           |
|----------------------------|--------------------------------------|--------------|------------------------|
| Fatigue life (Goodman)     | σ_a,eq/S_e + σ_m,eq/S_ut             | n_f ≥ 1.5    | Eq. (12.2)             |
| ECY (Early Cycle Yielding) | σ_a,eq + σ_m,eq ≤ S_y/n_y            | n_y ≥ 1.2    | Eq. (12.4)             |
| Von Mises (3D, theta_crit) | σ_VM ≤ S_y/n                         | n ≥ 1.2      | Eq. (5.13) or (6.19)   |
| Finite life                | N_f ≥ N_design                       | N_f/N_d ≥ 1  | Eq. (13.4)             |
| Buckling (rod)             | F_r,rod < 0 and P_cr > |F_r,rod| | n_buck ≥ 3.0 | Eq. (14.5)             |
| Pin shear stress           | τ_pin ≤ S_y/(sqrt(3)·n)              | n ≥ 2.0      | Eqs. (7.1)–(7.3)       |
| Pin bending stress         | σ_b,pin ≤ S_y/n                      | n ≥ 2.0      | Eq. (7.7)              |
| Bearing stress             | σ_br ≤ S_y/n                         | n ≥ 2.0      | Eqs. (7.8)–(7.11)      |
| Shear tear-out             | FS_tear = Psu / |F_r,axial(theta)| | FS ≥ 2.0     | Eq. (8.12)             |
| Lug contact (Goodman)      | σ_a,eq/S_e + σ_m,eq/S_ut at hole     | n_f ≥ 1.5    | Eq. (12.2) + Eq. (8.3) |


---

# 16. Step-by-Step Analysis Algorithm

| **Step** | **Action**                                                                                                                                                                                                                                    | **Equations Used**                  |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| 1        | At each theta = 0° to 360°: read F_A, F_B, F_C, T_in from Newton-Euler solver; compute F_r,rod,B, F_t,rod,B, F_r,rod,C, F_t,rod,C (Eqs. 2.9–2.12), F_r,crank,B, F_t,crank,B, F_r,crank,A, F_t,crank,A (Eqs. 2.3–2.6)                          | Known inputs                        |
| 2        | Compute T_offset(theta), T_rod(theta), F_B(theta) at each theta                                                                                                                                                                               | (2.1)–(2.10), (5.6), (6.13), (6.15) |
| 3        | Compute gravity distributed loads w_rod,g(theta) and w_crank,g(theta); evaluate parabolic bending moment distributions; compute M_rod,max(theta) = w_rod,g(theta)·l_rod²/8 and M_crank,max(theta) = w_crank,g(theta)·L_crank²/8 at each theta | (4.1)–(4.5), (6.1)–(6.5)            |
| 4        | Compute rod stresses: σ_ax,rod,body, σ_ax,rod,hole, τ_xy, τ_xz, σ_oop at body section and at hole locations                                                                                                                                   | (4.6)–(4.8), (5.1)–(5.13)           |
| 5        | Compute crank arm stresses: σ_ax,crank,body, σ_ax,crank,hole, σ_ax,crank, τ_xy, τ_T,in, τ_T,offset, σ_oop at body section and at hole locations                                                                                               | (6.6)–(6.19)                        |
| 6        | Compute pin stresses: shear, bending, bearing at each pin at each theta                                                                                                                                                                       | (7.1)–(7.11)                        |
| 7        | Compute lug nominal stress σ_nd, Kt_fixed, Kt_nd, Kt_nb; apply Kt_fixed to hole stress via Z_r or Z_c; check shear tear-out FS_tear(theta)                                                                                                    | (8.1)–(8.12)                        |
| 8        | Extract σ_max, σ_min over full cycle; compute σ_m, σ_a, σ_a,eq, σ_m,eq                                                                                                                                                                        | (9.1)–(9.7)                         |
| 9        | Compute S_e with Marin factors                                                                                                                                                                                                                | (10.1)–(10.3)                       |
| 10       | Apply Goodman + ECY at each section                                                                                                                                                                                                           | (12.1)–(12.5)                       |
| 11       | Compute N_f if σ_a,eq > S_e                                                                                                                                                                                                                  | (13.1)–(13.7)                       |
| 12       | Check buckling at theta values where F_r,rod(theta) < 0                                                                                                                                                                                      | (14.1)–(14.5)                       |
| 13       | Find min n(ξ) over all sections = critical section                                                                                                                                                                                            | min over all ξ                      |
| 14       | Iterate w, t, D_pA, D_pB, D_pC until n ≥ n_target                                                                                                                                                                                             | Target n_f ≥ 1.5–2.5 for machinery  |


---

# 17. Important Design Notes

- All stress equations assume linear elastic behaviour. Plasticity corrections (Neuber’s rule: K_f·K_ε = K_t²) are needed if local strains exceed yield.

- F_A, F_B, and F_C already include inertia forces from the full Newton-Euler dynamic analysis. Do not use quasi-static force estimates.

- The offset e > 0 ensures R ≠ −1 at most crank angles, producing a non-zero mean stress. This is captured automatically when σ(theta) is evaluated using F_r,rod,B, F_r,rod,C, F_r,crank,B, and F_r,crank,A at every degree.

- Both the forward stroke (theta = 0°–180°) and return stroke (theta = 180°–360°) must be checked. The asymmetry of F_r,rod,B, F_r,rod,C, F_t,rod,B, and F_t,rod,C over the cycle captures this automatically.

- Rectangular bars are weaker in torsion than round shafts. With w > t fixed for both links, out-of-plane bending is the critical bending mode. Increase t to improve out-of-plane bending resistance; increase w to improve in-plane stiffness.

- Pin holes are the primary stress concentration sites. Minimise Kt_fixed·F/Z by using the largest feasible pin radius relative to lug width and minimising the applied transverse force.

- Do NOT use τ = Tc/J with J = w·t·(w²+t²)/12 for torsion of a rectangular bar. Use the Saint-Venant formula: τ = T/(β·w²·t) for the rod and T/(β_c·w·t²) for the crank.

- The torsion coefficient β_c appears in both T_in (Section 6.6) and T_offset (Section 6.7) with the same denominator. This is intentional; only the torque in the numerator differs.

- Aluminium fatigue data shows higher scatter than steel. Apply k_e corresponding to 99.9% reliability for fatigue-critical aluminium components.

- Aluminium has no true endurance limit. Always specify a design life N_d and verify N_f ≥ N_d at the critical section.

- Kt_u1 = 4 (torsion at hole, Section 4.9.1) and Kt_u2 (bending at hole, Section 4.8.1) are conservative upper bounds only. Kt_u2 evaluates to zero in the current pin-pin model. Both should be reviewed if the boundary conditions or geometry change.

- KT = 1.0 (transverse shear at hole) is an unverified assumption. It requires confirmation that transverse shear is negligible at the governing critical point where Kt_fixed is applied.


---

# 18. Symbol Index

| **Symbol**                                       | **Description**                                                                                                                                                                                             | **Units** |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| F_A                                              | Joint reaction force vector at Pin A = [F_Ax, F_Ay]; Newton-Euler solver output                                                                                                                           | N         |
| F_B                                              | Joint reaction force vector at Pin B = [F_Bx, F_By]; Newton-Euler solver output                                                                                                                           | N         |
| F_C                                              | Joint reaction force vector at Pin C = [F_Cx, F_Cy]; Newton-Euler solver output                                                                                                                           | N         |
| T_in(theta)                                      | Input torque at Pin A; taken directly from Newton-Euler solver output                                                                                                                                       | N·mm      |
| F(theta)x                                        | Retired; superseded by F_Bx (x-component of F_B from Newton-Euler solver)                                                                                                                                   | N         |
| F(theta)y                                        | Retired; superseded by F_By (y-component of F_B from Newton-Euler solver)                                                                                                                                   | N         |
| F(theta)                                         | Resultant force magnitude at Pin B = sqrt[F_r,rod² + F_t,rod²]                                                                                                                                            | N         |
| F_r,rod,B / F_r,rod,C                            | Axial force on rod at Pin B end = F_Bx·cos(phi(theta)) + F_By·sin(phi(theta)); at Pin C end = −F_Cx·cos(phi(theta)) − F_Cy·sin(phi(theta))                                                                  | N         |
| F_t,rod,B / F_t,rod,C                            | Tangential force on rod at Pin B end = −F_Bx·sin(phi(theta)) + F_By·cos(phi(theta)); at Pin C end = F_Cx·sin(phi(theta)) − F_Cy·cos(phi(theta))                                                             | N         |
| F_r,crank,B / F_r,crank,A                        | Axial force on crank at Pin B end = F_Bx·cos(theta) + F_By·sin(theta); at Pin A end = −F_Ax·cos(theta) − F_Ay·sin(theta)                                                                                    | N         |
| F_t,crank,B / F_t,crank,A                        | Tangential force on crank at Pin B end = −F_Bx·sin(theta) + F_By·cos(theta); at Pin A end = F_Ax·sin(theta) − F_Ay·cos(theta)                                                                               | N         |
| F_r,rod,B(theta)                                 | Axial force on rod at Pin B end; from F_B (Eq. 2.9)                                                                                                                                                         | N         |
| F_r,rod,C(theta)                                 | Axial force on rod at Pin C end; from F_C (Eq. 2.11)                                                                                                                                                        | N         |
| F_t,rod,B(theta)                                 | Tangential force on rod at Pin B end; from F_B (Eq. 2.10)                                                                                                                                                   | N         |
| F_t,rod,C(theta)                                 | Tangential force on rod at Pin C end; from F_C (Eq. 2.12)                                                                                                                                                   | N         |
| F_r,crank,B(theta)                               | Axial force on crank at Pin B end; from F_B (Eq. 2.4)                                                                                                                                                       | N         |
| F_r,crank,A(theta)                               | Axial force on crank at Pin A end; from F_A (Eq. 2.6)                                                                                                                                                       | N         |
| F_t,crank,B(theta)                               | Tangential force on crank at Pin B end; from F_B (Eq. 2.3)                                                                                                                                                  | N         |
| F_t,crank,A(theta)                               | Tangential force on crank at Pin A end; from F_A (Eq. 2.5)                                                                                                                                                  | N         |
| T_in(theta)                                      | Input torque at Pin A; taken directly from Newton-Euler solver output                                                                                                                                       | N·mm      |
| T_rod(theta)                                     | Torsional moment on rod = F_t,rod,C(theta)·i_offset                                                                                                                                                         | N·mm      |
| T_offset(theta)                                  | Torsion of crank arm from link offset = F_t,rod,B(theta)·i_offset                                                                                                                                           | N·mm      |
| L_crank                                          | Crank arm length (Pin A to Pin B); maps to argument r in kinematics.py and dynamics.py; renamed from r in this document to avoid conflict with lug edge distance r                                          | mm        |
| l_rod                                            | Connecting rod length; maps to argument l in kinematics.py and dynamics.py                                                                                                                                  | mm        |
| e                                                | Eccentricity of slider line (e > 0)                                                                                                                                                                        | mm        |
| i_offset                                         | Out-of-plane offset at Pin B joint = (t_rod + t_crank)/2; source of rod torsion (T_rod) and out-of-plane bending (M_η,rod,B, M_η,crank,B)                                                                   | mm        |
| theta                                            | Crank angle from ground to crank arm (CCW)                                                                                                                                                                  | deg       |
| phi(theta)                                       | Rod inclination angle from +x axis                                                                                                                                                                          | deg       |
| omega                                            | Crank angular velocity (constant); maps to argument omega in kinematics.py and dynamics.py                                                                                                                  | rad/s     |
| w_rod, t_rod                                     | Rod in-plane width and out-of-plane thickness (w_rod > t_rod)                                                                                                                                              | mm        |
| w_crank, t_crank                                 | Crank in-plane width and out-of-plane thickness (w_crank > t_crank)                                                                                                                                        | mm        |
| d_p                                              | Retired; superseded by D_pA, D_pB, D_pC (pin diameters are not assumed equal)                                                                                                                               | mm        |
| D_pA                                             | Diameter of Pin A (ground bearing on crank); D_A = D_pA + δ_A                                                                                                                                               | mm        |
| D_pB                                             | Diameter of Pin B (crank-rod joint); D_B = D_pB + δ_B                                                                                                                                                       | mm        |
| D_pC                                             | Diameter of Pin C (rod-slider joint); D_C = D_pC + δ_C                                                                                                                                                      | mm        |
| D_A                                              | Hole diameter at Pin A (ground bearing on crank)                                                                                                                                                            | mm        |
| D_B                                              | Hole diameter at Pin B (crank-rod joint); appears in both rod and crank sma equations                                                                                                                       | mm        |
| D_C                                              | Hole diameter at Pin C (rod-slider joint)                                                                                                                                                                   | mm        |
| L_p                                              | Pin length (same for all three pins)                                                                                                                                                                        | mm        |
| A_r, A_c                                         | Rod and crank cross-sectional area                                                                                                                                                                          | mm²       |
| I_zr, I_zc                                       | Second moment of area, z-axis (in-plane bending)                                                                                                                                                            | mm⁴       |
| I_yr, I_yc                                       | Second moment of area, y-axis (out-of-plane bending, weak axis)                                                                                                                                             | mm⁴       |
| c_zr, c_yr                                       | Rod extreme fibre distances (z and y axes)                                                                                                                                                                  | mm        |
| c_zc, c_yc                                       | Crank extreme fibre distances (z and y axes)                                                                                                                                                                | mm        |
| Q_zr,max, Q_zc,max                               | First moment of area at neutral axis (rod and crank)                                                                                                                                                        | mm³       |
| β_r                                              | Saint-Venant torsion coefficient for rod = 1/3 − 0.21·(t_rod/w_rod)·[1 − t_rod⁴/(12·w_rod⁴)]; Roark approximation, less accurate for w_rod/t_rod close to 1                                               | —         |
| β_c                                              | Saint-Venant torsion coefficient for crank = 1/3 − 0.21·(t_crank/w_crank)·[1 − t_crank⁴/(12·w_crank⁴)]; Roark approximation, less accurate for w_crank/t_crank close to 1                                 | —         |
| Kte                                              | Retired; replaced by Kt_fixed = 2.34. Peterson Chart 5.12 Kte applies to double-shear configurations only; not valid for single-shear lugs in this mechanism.                                               | —         |
| Kte*                                            | Retired; replaced by Kt_fixed = 2.34. Peterson Chart 5.13 thick-lug correction applies to double-shear configurations only.                                                                                 | —         |
| Kt_nd, Kt_nb                                     | Net-section SCFs from Peterson 4.5.8 (Chart 4.67)                                                                                                                                                           | —         |
| Kt_fixed                                         | Fixed conservative stress concentration factor at pin hole = 2.34; applied to net-section area Z_r or Z_c in Eqs 4.7 and 6.8; replaces Peterson Chart 5.12 Kte which applies to double-shear only           | —         |
| Z_r                                              | Net-section area at rod pin hole = (w_rod − D_p)·t_rod; D_p = hole diameter (D_A, D_B, or D_C); Z_r,B uses D_B; Z_r,C uses D_C. Note: D_A/B/C = D_pA/B/C + δ_A/B/C                                          | m²        |
| Z_c                                              | Net-section area at crank pin hole = (w_crank − D_p)·t_crank; D_p = hole diameter (D_A, D_B, or D_C); Z_c,B uses D_B; Z_c,A uses D_A. Note: D_A/B/C = D_pA/B/C + δ_A/B/C                                    | m²        |
| Kt_u1                                            | Conservative torsion SCF at hole = 4, from Peterson 4.9.1                                                                                                                                                   | —         |
| Kt_u2                                            | Conservative bending SCF at hole from Peterson 4.8.1; evaluates to zero in current model                                                                                                                    | —         |
| KT                                               | Transverse shear SCF at hole = 1.0 (assumed; pending verification)                                                                                                                                          | —         |
| S_ut                                             | Ultimate tensile strength (483 MPa for 2024-T3)                                                                                                                                                             | MPa       |
| S_y                                              | Yield strength (345 MPa for 2024-T3)                                                                                                                                                                        | MPa       |
| S_e                                              | Corrected endurance limit                                                                                                                                                                                   | MPa       |
| S’_e                                            | Reference fatigue strength at 10⁸ cycles (130 MPa)                                                                                                                                                          | MPa       |
| σ’_f                                            | Fatigue strength coefficient (≈807 MPa for 2024-T3)                                                                                                                                                         | MPa       |
| b_B                                              | Basquin exponent                                                                                                                                                                                            | —         |
| σ_m, σ_a                                         | Mean and alternating stress components                                                                                                                                                                      | MPa       |
| σ_a,eq, σ_m,eq                                   | Von Mises equivalent alternating and mean stresses                                                                                                                                                          | MPa       |
| n_f                                              | Fatigue safety factor (Goodman)                                                                                                                                                                             | —         |
| n_y                                              | Static yield safety factor (ECY line)                                                                                                                                                                       | —         |
| n_buck                                           | Buckling safety factor                                                                                                                                                                                      | —         |
| N_f                                              | Cycles to failure                                                                                                                                                                                           | cycles    |
| D                                                | Miner cumulative damage sum                                                                                                                                                                                 | —         |
| R                                                | Stress ratio (σ_min/σ_max)                                                                                                                                                                                  | —         |
| ξ                                                | Position along link axis (0 = near pin, l = far pin)                                                                                                                                                        | mm        |
| ξ*                                              | Position along link of peak bending moment (dM/dξ = 0)                                                                                                                                                      | mm        |
| η                                                | In-plane transverse coordinate (−w/2 to +w/2)                                                                                                                                                               | mm        |
| ζ                                                | Out-of-plane coordinate (−t/2 to +t/2)                                                                                                                                                                      | mm        |
| m_rod                                            | Mass of connecting rod; maps to argument mass_rod in dynamics.py                                                                                                                                            | kg        |
| m_crank                                          | Mass of crank arm; maps to argument mass_crank in dynamics.py                                                                                                                                               | kg        |
| a_η,rod(ξ,theta)                                 | No longer used (D’Alembert approach removed); formerly transverse acceleration at position ξ along rod from kinematic analysis                                                                              | mm/s²     |
| a_η,crank(ξ,theta)                               | No longer used (D’Alembert approach removed); formerly transverse acceleration at position ξ along crank from kinematic analysis                                                                            | mm/s²     |
| W_rod(ξ, theta)                                  | Replaced by w_rod,g(theta); formerly distributed inertia load per unit length on rod (D’Alembert approach removed)                                                                                          | N/mm      |
| W_crank(ξ, theta)                                | Replaced by w_crank,g(theta); formerly distributed inertia load per unit length on crank (D’Alembert approach removed)                                                                                      | N/mm      |
| w_rod,g(theta)                                   | Gravity distributed load per unit length on rod = (m_rod·g / l_rod)·cos(phi(theta))                                                                                                                         | N/mm      |
| w_crank,g(theta)                                 | Gravity distributed load per unit length on crank = (m_crank·g / L_crank)·cos(theta)                                                                                                                        | N/mm      |
| g                                                | Gravitational acceleration = 9810 mm/s²                                                                                                                                                                     | mm/s²     |
| R_B,rod(theta)                                   | Transverse pin reaction at Pin B on rod from D’Alembert load balance                                                                                                                                        | N         |
| R_C,rod(theta)                                   | Transverse pin reaction at Pin C on rod                                                                                                                                                                     | N         |
| R_A,crank(theta)                                 | Transverse pin reaction at Pin A on crank                                                                                                                                                                   | N         |
| R_B,crank(theta)                                 | Transverse pin reaction at Pin B on crank                                                                                                                                                                   | N         |
| M_rod(ξ, theta)                                  | In-plane bending moment distribution along rod                                                                                                                                                              | N·mm      |
| M_crank(ξ, theta)                                | In-plane bending moment distribution along crank                                                                                                                                                            | N·mm      |
| M_rod,max(theta)                                 | Peak in-plane bending moment in rod at ξ*                                                                                                                                                                  | N·mm      |
| M_crank,max(theta)                               | Peak in-plane bending moment in crank at ξ*                                                                                                                                                                | N·mm      |
| M_η,rod(theta)                                   | Out-of-plane bending moment on rod at Pin B = F_r,rod,B·i_offset                                                                                                                                            | N·mm      |
| M_η,crank(theta)                                 | Out-of-plane bending moment on crank at Pin B = F_r,crank·i_offset                                                                                                                                          | N·mm      |
| σ_ax,rod,body(theta)                             | Peak normal stress at critical body section ξ* (no hole; no SCF)                                                                                                                                           | MPa       |
| σ_ax,rod,hole(theta)                             | Peak normal stress at pin hole = Kt_fixed·F_r,rod,B/Z_r,B or Kt_fixed·F_r,rod,C/Z_r,C; Kt_fixed = 2.34                                                                                                      | MPa       |
| σ_oop,rod(theta)                                 | Out-of-plane bending stress on rod at Pin B plain cross-section (no SCF)                                                                                                                                    | MPa       |
| σ_oop,crank(theta)                               | Out-of-plane bending stress on crank at Pin B plain cross-section (no SCF)                                                                                                                                  | MPa       |
| σ_VM                                             | Von Mises equivalent stress                                                                                                                                                                                 | MPa       |
| τ_T,rod(theta)                                   | Torsional shear stress on rod plain body, short face = T_rod/(β_r·w_rod²·t_rod)                                                                                                                             | MPa       |
| τ_T,in(theta)                                    | Torsional shear stress on crank from T_in, short face = T_in/(β_c·w_crank·t_crank²)                                                                                                                         | MPa       |
| τ_T,offset(theta)                                | Torsional shear stress on crank from T_offset, short face = T_offset/(β_c·w_crank·t_crank²)                                                                                                                 | MPa       |
| τ_nom,hole(theta)                                | Nominal torsional shear stress at hole cross-section (area-reduced)                                                                                                                                         | MPa       |
| τ_max,hole(theta)                                | Peak torsional shear at hole edge = Kt_u1·τ_nom,hole; Kt_u1 = 4 (conservative)                                                                                                                              | MPa       |
| τ_ξη,total                                       | Combined transverse shear on long face (η-direction)                                                                                                                                                        | MPa       |
| τ_ξζ,total                                       | Combined shear on short face (ζ-direction)                                                                                                                                                                  | MPa       |
| A_pin                                            | Pin cross-sectional area: A_pin,A = π·D_pA²/4; A_pin,B = π·D_pB²/4; A_pin,C = π·D_pC²/4                                                                                                                     | mm²       |
| τ_pin,A / τ_pin,B / τ_pin,C                      | Shear stress in Pin A, B, C respectively                                                                                                                                                                    | MPa       |
| M_pin,A / M_pin,B / M_pin,C                      | Bending moment in Pin A, B, C respectively                                                                                                                                                                  | N·mm      |
| σ_b,pin(theta)                                   | Peak pin bending stress = 32·M_pin/(π·D_p³); use D_pA, D_pB, or D_pC                                                                                                                                        | MPa       |
| σ_br,A                                           | Nominal bearing stress at Pin A = F_A(theta)/(D_pA·t_crank)                                                                                                                                                 | MPa       |
| σ_br,B,rod / σ_br,B,crank                        | Nominal bearing stress at Pin B, rod and crank sides                                                                                                                                                        | MPa       |
| σ_br,C                                           | Nominal bearing stress at Pin C = F_C(theta)/(2·D_pC·t_rod)                                                                                                                                                 | MPa       |
| H                                                | Total lug width in Peterson notation; H = w_rod or w_crank                                                                                                                                                  | mm        |
| c                                                | Distance from hole centre to free edge of lug in load direction (Peterson Chart 5.12); c = w/2 for round-ended lug (end radius = w/2)                                                                       | mm        |
| c/H                                              | Lug geometry ratio for Peterson Chart 5.12; c/H = (w/2)/w = 0.5 for all round-ended lugs in this mechanism                                                                                                  | —         |
| d/H                                              | Lug geometry ratio for Peterson Chart 5.12; d/H = D_p/H where D_p = D_pA, D_pB, or D_pC and H = w_rod or w_crank                                                                                            | —         |
| D_p                                              | Generic pin diameter placeholder used in Section 8 equations; substitute D_pA, D_pB, or D_pC for the pin being checked                                                                                      | mm        |
| h (lug)                                          | Lug thickness in Peterson notation; h = t_rod or t_crank                                                                                                                                                    | mm        |
| d (lug)                                          | Hole diameter in Peterson notation; d = D_pA, D_pB, or D_pC at the pin being checked                                                                                                                        | mm        |
| δ                                                | Diametral clearance between pin and hole; D_A/B/C = D_pA/B/C + δ_A/B/C. Not yet a design variable in Stage 2 — stresses.py uses a fixed default δ = 0.1 mm for all pins pending per-pin clearance sampling. | mm        |
| e%                                               | Pin clearance as percentage = (δ/D_p) × 100; D_p = D_pA, D_pB, or D_pC                                                                                                                                      | —         |
| Kt0.2 / Kt100 [Retired — not currently active] | Retired — not currently active. Kte boundary values from Peterson Chart 5.12; applies to double-shear only.                                                                                                 | —         |
| f                                                | Retired — not currently active. Clearance interpolation factor for Kte; read from Chart 5.12 sub-chart (double-shear only).                                                                                 | —         |
| r                                                | Edge distance = distance from hole centre to lug boundary (= w/2 for full-width round end)                                                                                                                  | mm        |
| a                                                | Edge distance beyond hole = w/2 − D_A/2, w/2 − D_B/2, or w/2 − D_C/2; substitute hole diameter at the pin being checked                                                                                     | mm        |
| Lsp                                              | Shear plane length (40° refined method)                                                                                                                                                                     | mm        |
| As                                               | Total shear area = 2·Lsp·t                                                                                                                                                                                  | mm²       |
| Psu                                              | Ultimate shear load = Ssu·As                                                                                                                                                                                | N         |
| Ssu                                              | Ultimate shear strength ≈ 0.6·Sut for 2024-T3                                                                                                                                                               | MPa       |
| FS_tear(theta)                                   | Factor of safety against shear tear-out = Psu / |F_r,axial(theta)|; evaluated per pin using axial force component                                                                                         | —         |
| σ_nd(theta)                                      | Nominal net-section stress at pin hole = F_r,axial / Z; axial force component at the relevant pin and lug; Z = Z_r,B/C (rod) or Z_c,B/A (crank)                                                             | MPa       |
| σ_nb(theta)                                      | Nominal bearing area stress = F_r,axial / (D_p·t); pin diameter D_pA/B/C used (not hole diameter); see Eq 8.6 for per-pin breakdown                                                                         | MPa       |
| Ktg / Ktn                                        | Gross and net section SCFs from Peterson 4.8.1 (Kt_u2 context)                                                                                                                                              | —         |
| k_a, k_b, k_c, k_d, k_e, k_f                     | Marin correction factors: surface, size, load, temperature, reliability, miscellaneous                                                                                                                      | —         |
| d_e                                              | Equivalent diameter for rectangular section in bending = 0.808·sqrt(w·t)                                                                                                                                    | mm        |
| A_95                                             | 95% stressed area for rectangular section = 0.05·w·t                                                                                                                                                        | mm²       |
| z_a                                              | Reliability z-factor (standard normal deviate); z_a = 2.326 for 99.9%                                                                                                                                       | —         |
| N_d                                              | Design life (required number of cycles)                                                                                                                                                                     | cycles    |
| n_rpm                                            | Crank rotational speed                                                                                                                                                                                      | rpm       |
| t_f(ξ)                                           | Life in seconds at section ξ = N_f(ξ)/(n_rpm/60)                                                                                                                                                            | s         |
| n_i                                              | Number of cycles at stress level i (Miner’s rule)                                                                                                                                                           | cycles    |
| P_cr                                             | Euler critical buckling load                                                                                                                                                                                | N         |
| K_c                                              | Effective length factor for buckling; K_c = 1.0 for pin-pin                                                                                                                                                 | —         |
| N_max,comp                                       | Maximum compressive axial force in rod over full crank cycle                                                                                                                                                | N         |
| ρ                                                | Density of 2024-T3 aluminium = 2780 kg/m³                                                                                                                                                                   | kg/m³     |
| E                                                | Elastic modulus of 2024-T3 aluminium = 73.1 GPa                                                                                                                                                             | GPa       |
| ν                                                | Poisson’s ratio of 2024-T3 = 0.33                                                                                                                                                                           | —         |
| G                                                | Shear modulus of 2024-T3 = 28 GPa                                                                                                                                                                           | GPa       |
| σ_max / σ_min                                    | Maximum and minimum stress over one crank revolution                                                                                                                                                        | MPa       |
