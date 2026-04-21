"""
Stage 1: 2D kinematic synthesis and filtering.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from mech390.datagen import sampling
from mech390.physics import kinematics

logger = logging.getLogger(__name__)

_CONSTRAINT_FUNCS = {
    "abs": abs,
    "sqrt": np.sqrt,
    "min": min,
    "max": max,
    "pow": pow,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
}


# rounds a value to the nearest manufacturing step size
def _round_to_res(value: float, resolution_m: float) -> float:
    if resolution_m <= 0.0:
        return value
    # e.g. 0.001 m -> 3 decimal places, 0.0001 m -> 4 decimal places
    decimal_places = max(0, round(-math.log10(resolution_m)))
    raw = round(value / resolution_m) * resolution_m
    return round(raw, decimal_places)


# converts a range definition (dict or list) into a plain (min, max) tuple
def _range_bounds(range_def: Any, name: str) -> Tuple[float, float]:
    if isinstance(range_def, dict) and "min" in range_def and "max" in range_def:
        return float(range_def["min"]), float(range_def["max"])
    if isinstance(range_def, (list, tuple)) and len(range_def) == 2:
        return float(range_def[0]), float(range_def[1])
    raise ValueError(f"Range for '{name}' must be {{min,max}} or [min,max]. Got: {range_def!r}")


# returns the valid range of e values for a given l that still allow the target ROM
def _feasible_e_interval_for_l(
    l: float,
    e_min: float,
    e_max: float,
    target_rom: float,
    strict_eps: float,
) -> Optional[Tuple[float, float]]:
    if target_rom >= 2.0 * l:
        return None

    e_cap_sq = l * l - (target_rom * target_rom) / 4.0
    if e_cap_sq <= 0.0:
        return None
    e_cap = np.sqrt(e_cap_sq)

    low = max(e_min, -l + strict_eps, -e_cap + strict_eps)
    high = min(e_max, l - strict_eps, e_cap - strict_eps)
    if low >= high:
        return None
    return low, high


# samples (l, e) pairs, keeping only ones where a valid r could exist
def _generate_constrained_stage1_candidates(
    method: str,
    n_samples: int,
    seed: int,
    l_range: Any,
    e_range: Any,
    target_rom: float,
    max_draws: int,
    strict_eps: float,
    resolution_m: float = 0.0,
) -> List[Dict[str, float]]:
    l_min_cfg, l_max_cfg = _range_bounds(l_range, "l")
    e_min_cfg, e_max_cfg = _range_bounds(e_range, "e")

    l_low = max(l_min_cfg, (target_rom / 2.0) + strict_eps)
    l_high = l_max_cfg
    if l_low >= l_high:
        logger.warning(
            "No feasible l-domain for target ROM %.6g inside config bounds [%.6g, %.6g].",
            target_rom, l_min_cfg, l_max_cfg,
        )
        return []

    if n_samples <= 0:
        return []

    candidates: List[Dict[str, float]] = []

    if method == "random":
        rng = np.random.default_rng(seed)
        for _ in range(max_draws):
            if len(candidates) >= n_samples:
                break
            l = _round_to_res(float(rng.uniform(l_low, l_high)), resolution_m)
            e_interval = _feasible_e_interval_for_l(l, e_min_cfg, e_max_cfg, target_rom, strict_eps)
            if e_interval is None:
                continue
            e = _round_to_res(float(rng.uniform(e_interval[0], e_interval[1])), resolution_m)
            candidates.append({"l": l, "e": e})
    elif method == "latin_hypercube":
        draws_used = 0
        batch_id = 0
        while len(candidates) < n_samples and draws_used < max_draws:
            remaining = n_samples - len(candidates)
            batch_n = min(max(remaining * 2, 64), max_draws - draws_used)
            unit_samples = sampling.get_sampler(
                method="latin_hypercube",
                param_ranges={
                    "u_l": {"min": 0.0, "max": 1.0},
                    "u_e": {"min": 0.0, "max": 1.0},
                },
                n_samples=batch_n,
                seed=seed + batch_id,
            )
            draws_used += batch_n
            batch_id += 1

            for row in unit_samples:
                if len(candidates) >= n_samples:
                    break
                l = _round_to_res(
                    float(l_low + row["u_l"] * (l_high - l_low)), resolution_m
                )
                e_interval = _feasible_e_interval_for_l(l, e_min_cfg, e_max_cfg, target_rom, strict_eps)
                if e_interval is None:
                    continue
                e = _round_to_res(
                    float(e_interval[0] + row["u_e"] * (e_interval[1] - e_interval[0])),
                    resolution_m,
                )
                candidates.append({"l": l, "e": e})
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    if len(candidates) < n_samples:
        logger.warning(
            "Constrained sampling produced %d/%d feasible (l,e) candidates. "
            "Consider widening l/e bounds or increasing n_attempts.",
            len(candidates), n_samples,
        )
    return candidates


# turns constraint strings into compiled code objects so they run fast inside the sampling loop
def _compile_constraints(constraints: List[str]) -> List[Tuple[str, Any]]:
    compiled: List[Tuple[str, Any]] = []
    for expr in constraints:
        if not isinstance(expr, str):
            raise TypeError(f"Constraint must be a string. Got: {type(expr)}")
        compiled.append((expr, compile(expr, "<sampling_constraint>", "eval")))
    return compiled


# builds the variable dict that constraint expressions can read (r, l, e, ROM target)
def _build_constraint_context(r: float, l: float, e: float, target_rom: float) -> Dict[str, float]:
    return {
        "r": r,
        "l": l,
        "e": e,
        "S": target_rom,
        "ROM_target": target_rom,
    }


# returns True if all constraint expressions pass for the given geometry
def _constraints_satisfied(compiled_constraints: List[Tuple[str, Any]], context: Dict[str, float]) -> bool:
    if not compiled_constraints:
        return True

    locals_env = dict(_CONSTRAINT_FUNCS)
    locals_env.update(context)
    for expr, code in compiled_constraints:
        try:
            if not bool(eval(code, {"__builtins__": {}}, locals_env)):
                return False
        except Exception as exc:
            logger.debug("Constraint evaluation failed for '%s': %s", expr, exc)
            return False
    return True


# solves analytically for the crank radius r that gives exactly the target ROM at fixed l and e
def solve_for_r_given_rom(
    l: float,
    e: float,
    target_rom: float,
    r_min: float,
    r_max: float,
    rom_tolerance: float = 1e-5,
) -> Optional[float]:
    """Returns None when no feasible r exists."""
    s_val = target_rom

    if abs(e) >= l:
        logger.debug("solve_for_r: e=%.4g >= l=%.4g", e, l)
        return None

    if s_val >= 2 * l:
        logger.debug("solve_for_r: S=%.4g >= 2l=%.4g", s_val, 2 * l)
        return None

    max_rom = 2 * np.sqrt(l**2 - e**2)
    if s_val >= max_rom:
        logger.debug("solve_for_r: S=%.4g >= max_rom=%.4g", s_val, max_rom)
        return None

    r_sol = (s_val / 2.0) * np.sqrt((4 * (l**2 - e**2) - s_val**2) / (4 * l**2 - s_val**2))

    if not (r_min <= r_sol <= r_max):
        logger.debug("solve_for_r: r=%.4g outside [%.4g, %.4g]", r_sol, r_min, r_max)
        return None

    if l <= r_sol + abs(e):
        return None

    term_ext = (l + r_sol) ** 2 - e**2
    term_ret = (l - r_sol) ** 2 - e**2
    if term_ext < 0.0 or term_ret < 0.0:
        return None

    rom_from_original = np.sqrt(term_ext) - np.sqrt(term_ret)
    if not np.isfinite(rom_from_original):
        return None

    branch_rhs = l**2 + r_sol**2 - e**2 - 0.5 * s_val**2
    if branch_rhs < -rom_tolerance:
        logger.debug(
            "solve_for_r: branch infeasible l=%.4g e=%.4g r=%.4g S=%.4g rhs=%.4g",
            l, e, r_sol, s_val, branch_rhs,
        )
        return None

    if abs(rom_from_original - s_val) > rom_tolerance:
        logger.debug(
            "solve_for_r: ROM residual too high l=%.4g e=%.4g r=%.4g "
            "(target=%.6g achieved=%.6g tol=%.3g)",
            l, e, r_sol, s_val, rom_from_original, rom_tolerance,
        )
        return None

    return r_sol


# checks a single (l, e) candidate: solves for r, rounds it, verifies ROM and QRR, returns result dict or None
def _accept_candidate(
    l: float,
    e: float,
    r_min: float,
    r_max: float,
    target_rom: float,
    qrr_min: float,
    qrr_max: float,
    rom_tolerance: float,
    compiled_constraints: List[Tuple[str, Any]],
    resolution_m: float = 0.0,
) -> Optional[Dict[str, Any]]:
    r_sol = solve_for_r_given_rom(l, e, target_rom, r_min, r_max, rom_tolerance=rom_tolerance)
    if r_sol is None:
        return None

    # round r to manufacturing resolution
    r_sol = _round_to_res(r_sol, resolution_m)

    # rounding can push r out of bounds, check again
    if not (r_min <= r_sol <= r_max):
        return None

    if not _constraints_satisfied(
        compiled_constraints,
        _build_constraint_context(r=r_sol, l=l, e=e, target_rom=target_rom),
    ):
        return None

    metrics = kinematics.calculate_metrics(r_sol, l, e)
    if not metrics["valid"]:
        return None

    if abs(metrics["ROM"] - target_rom) > rom_tolerance:
        return None

    qrr = metrics["QRR"]
    if not (qrr_min <= qrr <= qrr_max):
        return None

    return {
        "r": r_sol,
        "l": l,
        "e": e,
        "ROM": metrics["ROM"],
        "QRR": qrr,
        "theta_min": metrics["theta_retracted"],
        "theta_max": metrics["theta_extended"],
    }


def iter_valid_2d_mechanisms(
    config: Dict[str, Any],
    n_attempts: int = 100000,
) -> Iterator[Dict[str, Any]]:
    """Yields valid Stage-1 2D mechanism dicts until n_samples is reached or the draw budget runs out."""
    geo_ranges = config.get("geometry")
    op_settings = config.get("operating")
    samp_config = config.get("sampling")

    if geo_ranges is None or op_settings is None or samp_config is None:
        logger.error("Configuration missing required sections: geometry/operating/sampling")
        return

    l_range = geo_ranges["l"]
    e_range = geo_ranges["e"]
    r_range = geo_ranges["r"]
    r_min, r_max = _range_bounds(r_range, "r")

    target_rom = op_settings["ROM"]
    qrr_min, qrr_max = _range_bounds(op_settings["QRR"], "QRR")
    rom_tolerance = op_settings.get("ROM_tolerance", 1e-5)

    target_n_valid = int(samp_config.get("n_samples", 1000))
    if target_n_valid <= 0:
        return

    sampling_method = samp_config.get("method", "random")
    sampling_seed = config.get("random_seed", 42)
    constraint_exprs = samp_config.get("constraints", []) or []
    compiled_constraints = _compile_constraints(constraint_exprs)

    # manufacturing resolution (0 = no rounding)
    mfg = config.get("manufacturing") or {}
    resolution_m = float(mfg.get("resolution_mm", 0.0)) * 1e-3

    strict_eps = max(1e-12, rom_tolerance * 1e-3)

    # generate candidates in batches: 10x what we still need, minimum 64
    total_draws = 0
    n_valid_yielded = 0
    batch_seed_offset = 0

    while n_valid_yielded < target_n_valid:
        remaining = target_n_valid - n_valid_yielded
        batch_size = max(64, remaining * 10)

        if total_draws + batch_size > n_attempts:
            batch_size = n_attempts - total_draws

        if batch_size <= 0:
            logger.warning(
                "Stage 1: draw budget exhausted after %d draws - "
                "%d / %d valid designs produced.",
                total_draws, n_valid_yielded, target_n_valid,
            )
            break

        candidates = _generate_constrained_stage1_candidates(
            method=sampling_method,
            n_samples=batch_size,
            seed=sampling_seed + batch_seed_offset,
            l_range=l_range,
            e_range=e_range,
            target_rom=target_rom,
            max_draws=batch_size * 5,
            strict_eps=strict_eps,
            resolution_m=resolution_m,
        )
        total_draws += batch_size
        batch_seed_offset += batch_size

        for cand in candidates:
            if n_valid_yielded >= target_n_valid:
                break
            accepted = _accept_candidate(
                l=cand["l"],
                e=cand["e"],
                r_min=r_min,
                r_max=r_max,
                target_rom=target_rom,
                qrr_min=qrr_min,
                qrr_max=qrr_max,
                rom_tolerance=rom_tolerance,
                compiled_constraints=compiled_constraints,
                resolution_m=resolution_m,
            )
            if accepted is not None:
                n_valid_yielded += 1
                yield accepted


# runs iter_valid_2d_mechanisms and returns all results as a list
def generate_valid_2d_mechanisms(
    config: Dict[str, Any],
    n_attempts: int = 100000,
) -> List[Dict[str, Any]]:
    return list(iter_valid_2d_mechanisms(config, n_attempts=n_attempts))
