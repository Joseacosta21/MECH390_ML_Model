"""
Dynamics for the offset crank-slider mechanism.
Implements Newton-Euler equations for joint reaction forces.

All force/position vectors are np.ndarray of shape (2,) representing [x, y].
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from mech390.physics import kinematics, mass_properties

_COND_LIMIT = 1e12


def _cross_z(r_vec: np.ndarray, f_vec: np.ndarray) -> float:
    """2D cross product z-component: r x F."""
    return float(r_vec[0] * f_vec[1] - r_vec[1] * f_vec[0])


def _slider_friction_sign(v_sx: float, eps: float) -> float:
    """Kinetic friction direction, with deadband around zero velocity."""
    if abs(v_sx) <= eps:
        return 0.0
    return 1.0 if v_sx > 0.0 else -1.0


def solve_joint_reactions_newton_euler(
    theta: float,
    omega: float,
    r: float,
    l: float,
    e: float,
    mass_crank: float,
    mass_rod: float,
    mass_slider: float,
    I_crank: float = 1.0,
    I_rod: float = 1.0,
    mu: float = 0.0,
    g: float = 9.81,
    alpha_r: float = 0.0,
    v_eps: float = 1e-9,
    m_block: float = 0.0,
) -> Dict[str, Any]:
    """
    Solve the planar Newton-Euler joint-reaction system at one crank angle.

    Unknown vector: [F_Ax, F_Ay, F_Bx, F_By, F_Cx, F_Cy, N, tau_A].

    m_block is the payload mass sitting on the slider. It adds to slider
    inertia and weight only - N is solved by the system, so increased
    weight automatically raises friction via mu*N.
    """
    m_r = float(mass_crank)
    m_l = float(mass_rod)
    m_s = float(mass_slider)
    m_eff = m_s + float(m_block)   # effective slider + block mass
    i_r = float(I_crank)
    i_l = float(I_rod)
    mu_val = float(mu)
    grav = float(g)
    alpha_crank = float(alpha_r)

    # kinematics used by the force balance model
    r_A = np.array([0.0, 0.0], dtype=float)
    r_B = kinematics.crank_pin_position(theta, r)
    r_C = kinematics.slider_position(theta, r, l, e)

    r_Gr = mass_properties.crank_cog(theta, r)
    r_Gl = mass_properties.rod_cog(theta, r, l, e)

    a_B = kinematics.crank_pin_acceleration(theta, omega, r)
    a_C = kinematics.slider_acceleration(theta, omega, r, l, e)
    a_Gr = 0.5 * a_B
    a_Gl = 0.5 * (a_B + a_C)
    a_Gs = a_C

    alpha_l = kinematics.rod_angular_acceleration(theta, omega, r, l, e, alpha2=alpha_crank)
    v_sx = float(kinematics.slider_velocity(theta, omega, r, l, e)[0])
    s = _slider_friction_sign(v_sx, float(v_eps))

    r_A_Gr = r_A - r_Gr
    r_B_Gr = r_B - r_Gr
    r_B_Gl = r_B - r_Gl
    r_C_Gl = r_C - r_Gl

    # unknown vector indices
    f_ax, f_ay, f_bx, f_by, f_cx, f_cy, n_idx, tau_idx = range(8)

    A = np.zeros((8, 8), dtype=float)
    b = np.zeros(8, dtype=float)
    ex = np.array([1.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0], dtype=float)

    # 1) Crank Fx: F_Ax + F_Bx = m_r * a_Grx
    A[0, f_ax] = 1.0
    A[0, f_bx] = 1.0
    b[0] = m_r * a_Gr[0]

    # 2) Crank Fy: F_Ay + F_By = m_r * a_Gry + m_r * g
    A[1, f_ay] = 1.0
    A[1, f_by] = 1.0
    b[1] = m_r * a_Gr[1] + m_r * grav

    # 3) Crank moment about G_r.
    A[2, f_ax] = _cross_z(r_A_Gr, ex)
    A[2, f_ay] = _cross_z(r_A_Gr, ey)
    A[2, f_bx] = _cross_z(r_B_Gr, ex)
    A[2, f_by] = _cross_z(r_B_Gr, ey)
    A[2, tau_idx] = 1.0
    b[2] = i_r * alpha_crank

    # 4) Rod Fx: -F_Bx + F_Cx = m_l * a_Glx
    A[3, f_bx] = -1.0
    A[3, f_cx] = 1.0
    b[3] = m_l * a_Gl[0]

    # 5) Rod Fy: -F_By + F_Cy = m_l * a_Gly + m_l * g
    A[4, f_by] = -1.0
    A[4, f_cy] = 1.0
    b[4] = m_l * a_Gl[1] + m_l * grav

    # 6) Rod moment about G_l.
    A[5, f_bx] = _cross_z(r_B_Gl, -ex)
    A[5, f_by] = _cross_z(r_B_Gl, -ey)
    A[5, f_cx] = _cross_z(r_C_Gl, ex)
    A[5, f_cy] = _cross_z(r_C_Gl, ey)
    b[5] = i_l * alpha_l

    # 7) Slider Fx: -F_Cx - mu*s*N = m_eff * a_Gsx
    A[6, f_cx] = -1.0
    A[6, n_idx] = -mu_val * s
    b[6] = m_eff * a_Gs[0]

    # 8) Slider Fy: -F_Cy + N = m_eff * a_Gsy + m_eff * g
    #    Larger m_eff -> larger N -> larger friction in eq 7 (coupled automatically).
    A[7, f_cy] = -1.0
    A[7, n_idx] = 1.0
    b[7] = m_eff * a_Gs[1] + m_eff * grav

    cond_number = float(np.linalg.cond(A))
    if not np.isfinite(cond_number) or cond_number > _COND_LIMIT:
        raise ValueError(
            f"Newton-Euler solve ill-conditioned at theta={theta:.8f} rad "
            f"(cond={cond_number:.3e}, limit={_COND_LIMIT:.1e})."
        )

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Newton-Euler solve failed at theta={theta:.8f} rad: {exc}"
        ) from exc

    F_A = np.array([x[f_ax], x[f_ay]], dtype=float)
    F_B = np.array([x[f_bx], x[f_by]], dtype=float)
    F_C = np.array([x[f_cx], x[f_cy]], dtype=float)
    N = float(x[n_idx])
    F_f = float(-mu_val * s * N)
    tau_A = float(x[tau_idx])

    return {
        "F_A": F_A,
        "F_B": F_B,
        "F_C": F_C,
        "F_O": F_A.copy(),  # compatibility alias
        "N": N,
        "F_f": F_f,
        "tau_A": tau_A,
    }


# thin wrapper - positional args unchanged, optional kwargs passed through
def joint_reaction_forces(
    theta: float,
    omega: float,
    r: float,
    l: float,
    e: float,
    mass_crank: float,
    mass_rod: float,
    mass_slider: float,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calls solve_joint_reactions_newton_euler with positional args plus any kwargs."""
    return solve_joint_reactions_newton_euler(
        theta=theta,
        omega=omega,
        r=r,
        l=l,
        e=e,
        mass_crank=mass_crank,
        mass_rod=mass_rod,
        mass_slider=mass_slider,
        I_crank=float(kwargs.get("I_crank", 1.0)),
        I_rod=float(kwargs.get("I_rod", 1.0)),
        mu=float(kwargs.get("mu", 0.0)),
        g=float(kwargs.get("g", 9.81)),
        alpha_r=float(kwargs.get("alpha_r", 0.0)),
        v_eps=float(kwargs.get("v_eps", 1e-9)),
        m_block=float(kwargs.get("m_block", 0.0)),
    )
