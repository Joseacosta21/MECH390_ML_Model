"""
Stage 2: 3D embodiment expansion.
Expands valid 2D mechanisms into multiple 3D variants with geometric constraints.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

from mech390 import config as config_utils
from mech390.datagen import sampling

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"random", "latin_hypercube"}
_CONSTRAINT_TEXT = (
    "width_r > d_shaft_A + diametral_clearance_m + 2*min_wall, "
    "width_r > pin_diameter_B + diametral_clearance_m + 2*min_wall, "
    "width_l > pin_diameter_B + diametral_clearance_m + 2*min_wall, "
    "width_l > pin_diameter_C + diametral_clearance_m + 2*min_wall"
)

# parameters that use standard geometry resolution (resolution_mm)
_GEO_PARAMS = {"width_r", "width_l", "thickness_r", "thickness_l"}
# parameters that use pin/shaft resolution (pin_resolution_mm)
_PIN_PARAMS = {"d_shaft_A", "pin_diameter_B", "pin_diameter_C"}


# rounds a value to the nearest manufacturing step size
def _round_to_res(value: float, resolution_m: float) -> float:
    if resolution_m <= 0.0:
        return value
    decimal_places = max(0, round(-math.log10(resolution_m)))
    raw = round(value / resolution_m) * resolution_m
    return round(raw, decimal_places)


# checks that every pin hole has enough material around it (width - pin_diameter > min_net)
def _passes_width_pin_constraints(
    candidate: Dict[str, float],
    min_net_section: float,
) -> bool:
    return (
        candidate["width_r"] - candidate["d_shaft_A"]     > min_net_section
        and candidate["width_r"] - candidate["pin_diameter_B"] > min_net_section
        and candidate["width_l"] - candidate["pin_diameter_B"] > min_net_section
        and candidate["width_l"] - candidate["pin_diameter_C"] > min_net_section
    )


# yields up to max_attempts raw 3D parameter sets for one 2D design
def _iter_stage2_candidates(
    method: str,
    param_ranges: Dict[str, Dict[str, float]],
    max_attempts: int,
    seed: int,
) -> Iterator[Dict[str, float]]:
    if method == "random":
        rng = np.random.default_rng(seed)
        names = sorted(param_ranges.keys())
        for _ in range(max_attempts):
            yield {
                name: float(rng.uniform(param_ranges[name]["min"], param_ranges[name]["max"]))
                for name in names
            }
        return

    if method == "latin_hypercube":
        lhs_rows = sampling.get_sampler(
            method="latin_hypercube",
            param_ranges=param_ranges,
            n_samples=max_attempts,
            seed=seed,
        )
        for row in lhs_rows:
            yield {k: float(v) for k, v in row.items()}
        return

    raise ValueError(
        f"Unsupported Stage-2 sampling method '{method}'. "
        f"Supported methods: {sorted(_SUPPORTED_METHODS)}."
    )


def iter_expand_to_3d(
    valid_2d_designs: Iterable[Dict[str, float]],
    config: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    """For each valid 2D design, yields multiple 3D variants that pass width/pin constraints."""
    param_ranges = config_utils.get_stage2_param_ranges(config)
    stage2_sampling = config_utils.get_stage2_sampling_settings(config)
    n_variants_per_2d = stage2_sampling["n_variants_per_2d"]
    max_attempts = stage2_sampling["stage2_max_attempts_per_2d"]

    sampling_cfg = config.get("sampling", {})
    if sampling_cfg is None:
        sampling_cfg = {}
    sampling_method = sampling_cfg.get("method", "random")
    if sampling_method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported Stage-2 sampling method '{sampling_method}'. "
            f"Use one of: {sorted(_SUPPORTED_METHODS)}."
        )

    base_seed = int(config.get("random_seed", 42))

    # manufacturing resolutions (0 = no rounding)
    mfg = config.get("manufacturing") or {}
    res_m     = float(mfg.get("resolution_mm",     0.0)) * 1e-3
    pin_res_m = float(mfg.get("pin_resolution_mm", 0.0)) * 1e-3

    # net-section constraint: width - D_pin > min_net_section
    # min_net_section = delta + 2 * min_wall (ensures non-zero material around every hole)
    stress_cfg   = config.get("stress_analysis") or {}
    delta_m      = float(stress_cfg.get("diametral_clearance_m", 1e-4))
    min_wall_m   = float(stress_cfg.get("min_wall_m", 0.5e-3))
    min_net_section = delta_m + 2.0 * min_wall_m

    for design_idx, design_2d in enumerate(valid_2d_designs):
        accepted = 0
        design_seed = base_seed + design_idx * 9973

        for candidate in _iter_stage2_candidates(
            method=sampling_method,
            param_ranges=param_ranges,
            max_attempts=max_attempts,
            seed=design_seed,
        ):
            # round to manufacturing resolution before constraint check
            rounded: Dict[str, float] = {}
            for name, value in candidate.items():
                if name in _PIN_PARAMS:
                    rounded[name] = _round_to_res(value, pin_res_m)
                elif name in _GEO_PARAMS:
                    rounded[name] = _round_to_res(value, res_m)
                else:
                    rounded[name] = value

            if not _passes_width_pin_constraints(rounded, min_net_section):
                continue

            design_3d: Dict[str, Any] = dict(design_2d)
            design_3d.update(rounded)

            # TODO: compute masses and inertias with mech390.physics.mass_properties.
            # TODO: compute stresses with mech390.physics.stresses.

            accepted += 1
            yield design_3d
            if accepted >= n_variants_per_2d:
                break

        if accepted < n_variants_per_2d:
            logger.warning(
                "Stage 2 design %d: only %d/%d variants accepted after %d attempts "
                "(constraints: %s) - continuing with fewer variants.",
                design_idx, accepted, n_variants_per_2d, max_attempts, _CONSTRAINT_TEXT,
            )


# runs iter_expand_to_3d and collects all results into a list
def expand_to_3d(
    valid_2d_designs: List[Dict[str, float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return list(iter_expand_to_3d(valid_2d_designs, config))
