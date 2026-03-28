"""
Stage 2: 3D embodiment expansion.
Expands valid 2D mechanisms into multiple 3D variants with geometric constraints.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

import math

from mech390 import config as config_utils
from mech390.datagen import sampling

_SUPPORTED_METHODS = {"random", "latin_hypercube"}
_CONSTRAINT_TEXT = (
    "width_r > pin_diameter_A, width_r > pin_diameter_B, "
    "width_l > pin_diameter_B, width_l > pin_diameter_C"
)

# Parameters that use standard geometry resolution (resolution_mm)
_GEO_PARAMS = {"width_r", "width_l", "thickness_r", "thickness_l"}
# Parameters that use pin resolution (pin_resolution_mm)
_PIN_PARAMS = {"pin_diameter_A", "pin_diameter_B", "pin_diameter_C"}


def _round_to_res(value: float, resolution_m: float) -> float:
    """
    Round *value* to the nearest multiple of *resolution_m*.
    When resolution_m <= 0 no rounding is applied.

    Uses Python's built-in round() to eliminate binary floating-point
    representation noise.
    """
    if resolution_m <= 0.0:
        return value
    decimal_places = max(0, round(-math.log10(resolution_m)))
    raw = round(value / resolution_m) * resolution_m
    return round(raw, decimal_places)


def _passes_width_pin_constraints(candidate: Dict[str, float]) -> bool:
    """Enforce strict Stage-2 width/pin feasibility constraints."""
    return (
        candidate["width_r"] > candidate["pin_diameter_A"]
        and candidate["width_r"] > candidate["pin_diameter_B"]
        and candidate["width_l"] > candidate["pin_diameter_B"]
        and candidate["width_l"] > candidate["pin_diameter_C"]
    )


def _iter_stage2_candidates(
    method: str,
    param_ranges: Dict[str, Dict[str, float]],
    max_attempts: int,
    seed: int,
) -> Iterator[Dict[str, float]]:
    """Yield at most max_attempts candidate parameter sets for one 2D design."""
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
    """
    Streaming Stage-2 expansion: for each valid 2D design, emit many valid 3D variants.
    """
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

    # Manufacturing resolutions (0 = no rounding)
    mfg = config.get("manufacturing") or {}
    res_m     = float(mfg.get("resolution_mm",     0.0)) * 1e-3
    pin_res_m = float(mfg.get("pin_resolution_mm", 0.0)) * 1e-3

    for design_idx, design_2d in enumerate(valid_2d_designs):
        accepted = 0
        design_seed = base_seed + design_idx * 9973

        for candidate in _iter_stage2_candidates(
            method=sampling_method,
            param_ranges=param_ranges,
            max_attempts=max_attempts,
            seed=design_seed,
        ):
            # --- Round to manufacturing resolution BEFORE constraint check ---
            rounded: Dict[str, float] = {}
            for name, value in candidate.items():
                if name in _PIN_PARAMS:
                    rounded[name] = _round_to_res(value, pin_res_m)
                elif name in _GEO_PARAMS:
                    rounded[name] = _round_to_res(value, res_m)
                else:
                    rounded[name] = value

            if not _passes_width_pin_constraints(rounded):
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
            raise ValueError(
                "Stage 2 could not generate enough feasible 3D variants for "
                f"design index {design_idx}: accepted {accepted}/{n_variants_per_2d} "
                f"after {max_attempts} attempts. Constraints: {_CONSTRAINT_TEXT}. "
                f"Ranges: {param_ranges}."
            )


def expand_to_3d(
    valid_2d_designs: List[Dict[str, float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compatibility wrapper that materializes the streaming Stage-2 iterator."""
    return list(iter_expand_to_3d(valid_2d_designs, config))
