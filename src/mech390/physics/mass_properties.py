"""
Mass properties for the offset crank-slider mechanism.

Computes masses, mass moments of inertia, area moments of inertia,
and kinematic center-of-gravity positions for all links.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from mech390.physics import kinematics


@dataclass(frozen=True)
class MassPropertiesResult:
    """Typed container for computed design-level mass properties."""
    rho: float
    mass_crank: float
    mass_rod: float
    mass_slider: float
    I_mass_crank_cg_z: float
    I_mass_rod_cg_z: float
    I_mass_slider_cg_z: float
    I_area_crank_yy: float
    I_area_crank_zz: float
    I_area_rod_yy: float
    I_area_rod_zz: float
    I_area_slider_yy: float
    I_area_slider_zz: float

    def to_dict(self) -> Dict[str, float]:
        """Returns all mass properties as a dict, including legacy slider area-moment key aliases."""
        out = asdict(self)
        out["I_area_slider_x"] = self.I_area_slider_yy
        out["I_area_slider_y"] = self.I_area_slider_zz
        return out


def _require_positive(name: str, value: Any) -> float:
    """Convert to float and enforce strict positivity."""
    val = float(value)
    if val <= 0.0:
        raise ValueError(f"{name} must be > 0. Got {value!r}.")
    return val


def _circle_area(diameter: float) -> float:
    """Area of a circle from diameter."""
    d = _require_positive("diameter", diameter)
    return 0.25 * np.pi * d * d


def _hole_mass(diameter: float, thickness: float, rho: float) -> float:
    """Mass removed by a through-hole in a plate/prism of thickness."""
    t = _require_positive("thickness", thickness)
    density = float(rho)
    return _circle_area(diameter) * t * density


def _disc_moi_cg_z(mass: float, diameter: float) -> float:
    """Mass moment of inertia of a solid disk about its own centroidal z-axis."""
    d = _require_positive("diameter", diameter)
    m = _require_positive("mass", mass)
    return 0.5 * m * (0.5 * d) ** 2


def _rect_prism_mass_and_moi_cg_z(
    length: float,
    width: float,
    thickness: float,
    rho: float,
) -> Tuple[float, float]:
    """Return (mass, I_cg_z) for a rectangular prism."""
    l = _require_positive("length", length)
    w = _require_positive("width", width)
    t = _require_positive("thickness", thickness)
    density = float(rho)

    m_rect = density * l * w * t
    i_rect_cg_z = (m_rect / 12.0) * (l * l + w * w)
    return m_rect, i_rect_cg_z


def _fixed_rho_from_config(config: Mapping[str, Any]) -> float:
    """
    Extracts fixed density from config.material.rho.

    Accepts a scalar number or a {'min': x, 'max': x} dict where min == max.
    """
    material = config.get("material")
    if not isinstance(material, Mapping):
        raise ValueError("Config must include 'material' mapping with 'rho'.")

    rho_def = material.get("rho")
    if isinstance(rho_def, (int, float)):
        return float(rho_def)

    if isinstance(rho_def, Mapping) and "min" in rho_def and "max" in rho_def:
        rho_min = float(rho_def["min"])
        rho_max = float(rho_def["max"])
        if rho_min != rho_max:
            raise ValueError(
                "material.rho must be fixed for mass-properties computation "
                f"(min == max). Got min={rho_min}, max={rho_max}."
            )
        return rho_min

    raise ValueError(
        "Unsupported material.rho format. Use scalar or {min,max} with min==max."
    )


def link_body_length(center_distance: float, d_left: float, d_right: float) -> float:
    """
    Compute rectangular link body length using minimal tangency rule:
        L = center_distance + 0.5*d_left + 0.5*d_right
    """
    c = _require_positive("center_distance", center_distance)
    dl = _require_positive("d_left", d_left)
    dr = _require_positive("d_right", d_right)
    return c + 0.5 * dl + 0.5 * dr


def link_plan_area_net(
    center_distance: float,
    width: float,
    d_left: float,
    d_right: float,
) -> float:
    """Net plan area of a rectangular link with two circular through-holes."""
    w = _require_positive("width", width)
    length = link_body_length(center_distance, d_left, d_right)
    area_net = length * w - (_circle_area(d_left) + _circle_area(d_right))
    if area_net <= 0.0:
        raise ValueError(
            "Non-physical net link plan area (<= 0). "
            f"center_distance={center_distance}, width={width}, "
            f"d_left={d_left}, d_right={d_right}."
        )
    return area_net


def link_volume_net(
    center_distance: float,
    width: float,
    thickness: float,
    d_left: float,
    d_right: float,
) -> float:
    """Net link volume after subtracting two through-holes."""
    t = _require_positive("thickness", thickness)
    return link_plan_area_net(center_distance, width, d_left, d_right) * t


def link_mass(
    center_distance: float,
    width: float,
    thickness: float,
    d_left: float,
    d_right: float,
    rho: float,
) -> float:
    """Net link mass from net volume and density."""
    density = float(rho)
    return link_volume_net(center_distance, width, thickness, d_left, d_right) * density


def link_mass_moi_cg_z(
    center_distance: float,
    width: float,
    thickness: float,
    d_left: float,
    d_right: float,
    rho: float,
) -> float:
    """Net mass moment of inertia about link CG z-axis using composite subtraction."""
    c = _require_positive("center_distance", center_distance)
    w = _require_positive("width", width)
    t = _require_positive("thickness", thickness)
    dl = _require_positive("d_left", d_left)
    dr = _require_positive("d_right", d_right)
    density = float(rho)

    length = link_body_length(c, dl, dr)
    _, i_rect_cg_z = _rect_prism_mass_and_moi_cg_z(length, w, t, density)

    m_h_left = _hole_mass(dl, t, density)
    m_h_right = _hole_mass(dr, t, density)

    i_h_left_center_z = _disc_moi_cg_z(m_h_left, dl)
    i_h_right_center_z = _disc_moi_cg_z(m_h_right, dr)

    # Exact offsets of each pin hole center relative to the rectangular body CG.
    # Link body length L = c + 0.5*dl + 0.5*dr; CG is at L/2 from the left end.
    # Left hole at 0.5*dl from left end  -> offset = 0.5*dl - L/2 = -0.5*c + 0.25*(dl - dr)
    # Right hole at c + 0.5*dl from left -> offset = c + 0.5*dl - L/2 =  0.5*c + 0.25*(dl - dr)
    x_left = -0.5 * c + 0.25 * (dl - dr)
    x_right = 0.5 * c + 0.25 * (dl - dr)

    i_net = (
        i_rect_cg_z
        - (i_h_left_center_z + m_h_left * x_left * x_left)
        - (i_h_right_center_z + m_h_right * x_right * x_right)
    )
    if i_net <= 0.0:
        raise ValueError(
            "Non-physical link mass moment of inertia (<= 0). "
            f"center_distance={center_distance}, width={width}, thickness={thickness}, "
            f"d_left={d_left}, d_right={d_right}, rho={rho}."
        )
    return i_net


def link_area_moments_gross(width: float, thickness: float) -> Dict[str, float]:
    """
    Gross rectangular area moments for link cross-section (holes ignored).

    Iyy is about the axis parallel to thickness; Izz is about the axis parallel to width.
    """
    w = _require_positive("width", width)
    t = _require_positive("thickness", thickness)
    return {
        "Iyy": w * t**3 / 12.0,
        "Izz": t * w**3 / 12.0,
    }


def slider_volume_net(length: float, width: float, height: float, d_c: float) -> float:
    """Net slider volume from length*width plan minus one circular hole, extruded by height."""
    l = _require_positive("length", length)
    w = _require_positive("width", width)
    h = _require_positive("height", height)
    dc = _require_positive("d_c", d_c)

    plan_area_net = l * w - _circle_area(dc)
    if plan_area_net <= 0.0:
        raise ValueError(
            "Non-physical net slider plan area (<= 0). "
            f"length={length}, width={width}, d_c={d_c}."
        )
    return plan_area_net * h


def slider_mass(length: float, width: float, height: float, d_c: float, rho: float) -> float:
    """Net slider mass."""
    density = float(rho)
    return slider_volume_net(length, width, height, d_c) * density


def slider_mass_moi_cg_z(
    length: float,
    width: float,
    height: float,
    d_c: float,
    rho: float,
) -> float:
    """Net slider mass moment of inertia about CG z-axis."""
    l = _require_positive("length", length)
    w = _require_positive("width", width)
    h = _require_positive("height", height)
    dc = _require_positive("d_c", d_c)
    density = float(rho)

    _, i_rect_cg_z = _rect_prism_mass_and_moi_cg_z(l, w, h, density)

    m_hole = _hole_mass(dc, h, density)
    i_hole_center_z = _disc_moi_cg_z(m_hole, dc)

    i_net = i_rect_cg_z - i_hole_center_z
    if i_net <= 0.0:
        raise ValueError(
            "Non-physical slider mass moment of inertia (<= 0). "
            f"length={length}, width={width}, height={height}, d_c={d_c}, rho={rho}."
        )
    return i_net


def slider_area_moments_gross(width: float, height: float) -> Dict[str, float]:
    """
    Gross rectangular area moments for slider section (holes ignored).
    Returns Iyy, Izz, and legacy aliases I_area_x / I_area_y.
    """
    w = _require_positive("width", width)
    h = _require_positive("height", height)
    iyy = w * h**3 / 12.0
    izz = h * w**3 / 12.0
    return {
        "Iyy": iyy,
        "Izz": izz,
        "I_area_x": iyy,
        "I_area_y": izz,
    }


def _required_positive_field(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> float:
    """Fetch required key from mapping and validate as positive float."""
    try:
        return _require_positive(label, mapping[key])
    except KeyError as exc:
        raise ValueError(f"Missing field required for mass properties: {label}") from exc


def _load_design_link_inputs(design: Mapping[str, Any]) -> Dict[str, float]:
    """Load and validate required design dimensions for crank/rod/hole geometry."""
    required_fields = (
        ("r", "design.r"),
        ("l", "design.l"),
        ("width_r", "design.width_r"),
        ("width_l", "design.width_l"),
        ("thickness_r", "design.thickness_r"),
        ("thickness_l", "design.thickness_l"),
        ("d_shaft_A", "design.d_shaft_A"),
        ("pin_diameter_B", "design.pin_diameter_B"),
        ("pin_diameter_C", "design.pin_diameter_C"),
    )
    return {
        key: _required_positive_field(design, key, label)
        for key, label in required_fields
    }


def _load_slider_inputs(config: Mapping[str, Any]) -> Dict[str, float]:
    """Load and validate slider geometry from config.geometry.slider."""
    geometry = config.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("Config must include 'geometry' mapping.")
    slider = geometry.get("slider")
    if not isinstance(slider, Mapping):
        raise ValueError("Config must include 'geometry.slider' mapping.")

    return {
        "length": _required_positive_field(slider, "length", "geometry.slider.length"),
        "width": _required_positive_field(slider, "width", "geometry.slider.width"),
        "height": _required_positive_field(slider, "height", "geometry.slider.height"),
    }


def compute_design_mass_properties(
    design: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, float]:
    """Compute all mass-properties outputs for one sampled design and global config."""
    rho = _fixed_rho_from_config(config)

    link_inputs = _load_design_link_inputs(design)
    slider_inputs = _load_slider_inputs(config)

    mass_crank = link_mass(
        link_inputs["r"],
        link_inputs["width_r"],
        link_inputs["thickness_r"],
        link_inputs["d_shaft_A"],
        link_inputs["pin_diameter_B"],
        rho,
    )
    mass_rod = link_mass(
        link_inputs["l"],
        link_inputs["width_l"],
        link_inputs["thickness_l"],
        link_inputs["pin_diameter_B"],
        link_inputs["pin_diameter_C"],
        rho,
    )
    mass_slider = slider_mass(
        slider_inputs["length"],
        slider_inputs["width"],
        slider_inputs["height"],
        link_inputs["pin_diameter_C"],
        rho,
    )

    i_mass_crank = link_mass_moi_cg_z(
        link_inputs["r"],
        link_inputs["width_r"],
        link_inputs["thickness_r"],
        link_inputs["d_shaft_A"],
        link_inputs["pin_diameter_B"],
        rho,
    )
    i_mass_rod = link_mass_moi_cg_z(
        link_inputs["l"],
        link_inputs["width_l"],
        link_inputs["thickness_l"],
        link_inputs["pin_diameter_B"],
        link_inputs["pin_diameter_C"],
        rho,
    )
    i_mass_slider = slider_mass_moi_cg_z(
        slider_inputs["length"],
        slider_inputs["width"],
        slider_inputs["height"],
        link_inputs["pin_diameter_C"],
        rho,
    )

    i_area_crank = link_area_moments_gross(link_inputs["width_r"], link_inputs["thickness_r"])
    i_area_rod = link_area_moments_gross(link_inputs["width_l"], link_inputs["thickness_l"])
    i_area_slider = slider_area_moments_gross(slider_inputs["width"], slider_inputs["height"])

    return MassPropertiesResult(
        rho=rho,
        mass_crank=mass_crank,
        mass_rod=mass_rod,
        mass_slider=mass_slider,
        I_mass_crank_cg_z=i_mass_crank,
        I_mass_rod_cg_z=i_mass_rod,
        I_mass_slider_cg_z=i_mass_slider,
        I_area_crank_yy=i_area_crank["Iyy"],
        I_area_crank_zz=i_area_crank["Izz"],
        I_area_rod_yy=i_area_rod["Iyy"],
        I_area_rod_zz=i_area_rod["Izz"],
        I_area_slider_yy=i_area_slider["Iyy"],
        I_area_slider_zz=i_area_slider["Izz"],
    ).to_dict()


### Kinematic COG helpers

# crank COG approximated at midpoint between origin O and crank pin B
def crank_cog(theta: float, r: float) -> np.ndarray:
    """Returns crank center of gravity (midpoint between O and pin B)."""
    return 0.5 * kinematics.crank_pin_position(theta, r)


# rod COG approximated at midpoint between pin B and pin C
def rod_cog(theta: float, r: float, l: float, e: float) -> np.ndarray:
    """Returns rod center of gravity (midpoint between pin B and pin C)."""
    pos_B = kinematics.crank_pin_position(theta, r)
    pos_C = kinematics.slider_position(theta, r, l, e)
    return 0.5 * (pos_B + pos_C)


# slider COG coincides with pin C since slider is constrained to x-axis
def slider_cog(theta: float, r: float, l: float, e: float) -> np.ndarray:
    """Returns slider center of gravity (coincides with pin C)."""
    return kinematics.slider_position(theta, r, l, e)
