"""
Stage 2: 3D embodiment expansion.
Expands valid 2D mechanisms into multiple 3D variants with geometric constraints.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List

import numpy as np

from mech390 import config as config_utils
from mech390.datagen import sampling

_SUPPORTED_METHODS = {"random", "latin_hypercube"}


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

    sampling_method = config.get("sampling", {}).get("method", "random")
    if sampling_method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported Stage-2 sampling method '{sampling_method}'. "
            f"Use one of: {sorted(_SUPPORTED_METHODS)}."
        )

    base_seed = int(config.get("random_seed", 42))

    for design_idx, design_2d in enumerate(valid_2d_designs):
        accepted = 0
        design_seed = base_seed + design_idx * 9973

        for candidate in _iter_stage2_candidates(
            method=sampling_method,
            param_ranges=param_ranges,
            max_attempts=max_attempts,
            seed=design_seed,
        ):
            if not _passes_width_pin_constraints(candidate):
                continue

            design_3d: Dict[str, Any] = dict(design_2d)
            design_3d.update(candidate)

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
                f"after {max_attempts} attempts. Constraints: width_r > pin_diameter_A, "
                "width_r > pin_diameter_B, width_l > pin_diameter_B, "
                f"width_l > pin_diameter_C. Ranges: {param_ranges}."
            )


def expand_to_3d(
    valid_2d_designs: List[Dict[str, float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compatibility wrapper that materializes the streaming Stage-2 iterator."""
    return list(iter_expand_to_3d(valid_2d_designs, config))
