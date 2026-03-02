"""  # Module docstring start.
Stage 1: 2D Kinematic Synthesis and Filtering.  # Summarizes this stage's purpose.
Implements the "Sample 2, Solve 1" strategy to enforce ROM constraints.  # Notes the core method used.
"""  # Module docstring end.

import numpy as np  # Imports NumPy utilities (currently unused here).
# brentq no longer needed — r is solved analytically from the ROM formula.
from typing import Dict, List, Optional, Tuple, Any  # Imports typing aliases for signatures.
import logging  # Imports logging for runtime diagnostics.

from mech390.physics import kinematics  # Imports kinematic metric computations.
from mech390.datagen import sampling  # Imports sampling utilities for candidate generation.

# Set up logger  # Marks logger initialization section.
logger = logging.getLogger(__name__)  # Creates a module-scoped logger.


def solve_for_r_given_rom(
    l: float,
    e: float,
    target_rom: float,
    r_min: float,
    r_max: float,
) -> Optional[float]:
    """
    Analytically solves for crank radius r that produces the target ROM,
    given rod length l and offset e.

    Derivation
    ----------
    From the dead-centre geometry, the slider positions at the two dead centres are:

        x_max = sqrt((r + l)^2 - e^2)   (extended)
        x_min = sqrt((l - r)^2 - e^2)   (retracted)

    Setting S = ROM = x_max - x_min and solving algebraically yields the
    exact closed-form:

        r = (S / 2) * sqrt( (4*(l^2 - e^2) - S^2) / (4*l^2 - S^2) )

    Args:
        l: Rod length.
        e: Offset (D).
        target_rom: Desired Range of Motion (S).
        r_min, r_max: Allowable bounds for r (read from config geometry.r).

    Returns:
        r_solution: The exact radius r, or None if infeasible for these inputs.
    """
    S = target_rom

    # --- Geometric feasibility conditions ---
    # These must all hold for the closed-form formula to yield a real, positive r.

    # 1. Offset must be less than rod length (otherwise no valid triangle exists)
    if abs(e) >= l:
        logger.debug("solve_for_r: e=%.4g >= l=%.4g — offset too large", e, l)
        return None

    # 2. ROM must be less than 2l  (ensures denominator 4l²−S² > 0)
    if S >= 2 * l:
        logger.debug("solve_for_r: S=%.4g >= 2l=%.4g — ROM exceeds rod-length limit", S, 2 * l)
        return None

    # 3. ROM must be less than 2*sqrt(l²−e²)  (ensures numerator 4(l²−e²)−S² > 0)
    max_rom = 2 * np.sqrt(l**2 - e**2)
    if S >= max_rom:
        logger.debug(
            "solve_for_r: S=%.4g >= 2*sqrt(l²−e²)=%.4g — ROM exceeds geometric maximum",
            S, max_rom,
        )
        return None

    # --- Closed-form solution ---
    #   r = (S/2) * sqrt( (4(l²−e²) − S²) / (4l² − S²) )
    r_sol = (S / 2.0) * np.sqrt((4 * (l**2 - e**2) - S**2) / (4 * l**2 - S**2))

    # --- Validate against config bounds ---
    if not (r_min <= r_sol <= r_max):
        logger.debug(
            "solve_for_r: r=%.4g outside config bounds [%.4g, %.4g]",
            r_sol, r_min, r_max,
        )
        return None

    # --- Confirm full-rotation crank-slider geometry (l > r + |e|) ---
    if l <= r_sol + abs(e):
        return None

    return r_sol


def generate_valid_2d_mechanisms(  # Defines main generator for valid 2D mechanism designs.
    config: Dict[str, Any], n_attempts: int = 100000  # Accepts config and optional attempt cap.
) -> List[Dict[str, Any]]:  # Returns a list of valid design dictionaries.
    """  # Function docstring start.
    Generates a list of valid 2D mechanisms.  # Summarizes function purpose.

    Strategy:  # Starts high-level workflow summary.
      1. Sample l and e from config ranges.  # Step 1 samples independent geometry variables.
      2. Solve for r to match target ROM.  # Step 2 derives dependent variable r.
      3. Check if r is within config bounds.  # Step 3 validates solved r range.
      4. Check QRR constraints.  # Step 4 enforces quality/ratio constraint.

    Args:  # Starts argument documentation.
        config: Configuration dictionary (must contain 'geometry', 'operating', 'material' etc.)  # Describes required config.
        n_attempts: Max attempts to try.  # Documents optional attempt cap argument.

    Returns:  # Starts return documentation.
        List of dicts with valid geometry {r, l, e, ...}  # Describes returned record structure.
    """  # Function docstring end.
    valid_designs = []  # Initializes collection of accepted designs.

    # Extract settings  # Marks config extraction block.
    # We expect these keys to be present in the config.  # States assumption about config schema.
    # If not, we let it crash to signal missing config.  # Notes intended failure behavior.
    geo_ranges = config.get("geometry")  # Reads geometry range settings.
    op_settings = config.get("operating")  # Reads operating target settings.
    samp_config = config.get("sampling")  # Reads sampling method settings.

    if geo_ranges is None or op_settings is None or samp_config is None:  # Validates presence of required sections.
        logger.error(  # Logs explicit error for missing configuration.
            "Configuration is missing required sections: 'geometry', 'operating', or 'sampling'"  # Provides missing-section message.
        )
        return []  # Returns empty result when config is incomplete.

    # Ranges  # Marks geometry range extraction.
    l_range = geo_ranges["l"]  # Gets allowable l range.
    e_range = geo_ranges["e"]  # Gets allowable e range.
    r_range = geo_ranges["r"]  # Gets allowable r bounds for solver.

    # Targets  # Marks operating target extraction.
    target_rom = op_settings["ROM"]  # Reads desired ROM target.
    qrr_range = op_settings["QRR"]  # Reads acceptable QRR interval.

    # Setup sampler for l and e  # Marks sampling setup block.
    # We use random sampling for simplicity in this loop, or we could use LHS pre-generation.  # Notes sampling alternatives.
    # If config says LHS, we should ideally use that.  # States intention to honor configured method.
    # Let's support the sampling config.  # Confirms behavior is config-driven.

    # Prepare ranges for sampling ONLY l and e  # Marks sampled variable definition.
    # We do NOT sample r, we solve for it.  # Explains why r is excluded from sampler inputs.
    param_ranges_to_sample = {  # Builds sampler range map.
        "l": l_range,  # Includes l range in sampled parameters.
        "e": e_range,  # Includes e range in sampled parameters.
    }

    # Get candidate samples for l and e  # Marks sample-count selection.
    # Note: If n_samples is specified in config, we try to produce that many VALID designs?  # Documents interpretation question.
    # Or we treat n_samples as "candidates to try"?  # Documents alternative interpretation.
    # Usually "n_samples" in LHS means number of candidates generated.  # Notes common convention used.
    target_n_samples = samp_config.get("n_samples", 1000)  # Chooses candidate pool size.

    # Generate candidates  # Marks candidate generation call.
    # We are generating (l, e) pairs  # Clarifies sampled tuple content.
    candidates = sampling.get_sampler(  # Produces sampled candidate list/iterator.
        method=samp_config.get("method", "random"),  # Selects sampling method from config.
        param_ranges=param_ranges_to_sample,  # Passes parameter ranges to sampler.
        n_samples=target_n_samples,  # Requests desired number of candidates.
        seed=config.get("random_seed", 42),  # Uses deterministic default seed when absent.
    )

    for cand in candidates:  # Iterates through each sampled (l, e) candidate.
        l = cand["l"]  # Extracts sampled l.
        e = cand["e"]  # Extracts sampled e.

        # Constraints from config (strings)  # Marks deferred constraint notes.
        # "l >= 2.5*r" -> This involves r, so we can't check it yet?  # Explains dependency on unsolved r.
        # Or we can check "l >= 2.5 * (ROM/2)" as a rough check?  # Mentions possible approximation.
        # Better to solve r first, then check.  # States chosen order of operations.

        # Solve for r  # Marks r-solve step.
        r_min = r_range["min"] if isinstance(r_range, dict) else r_range[0]  # Normalizes lower bound shape.
        r_max = r_range["max"] if isinstance(r_range, dict) else r_range[1]  # Normalizes upper bound shape.

        r_sol = solve_for_r_given_rom(l, e, target_rom, r_min, r_max)  # Solves r for this candidate.

        if r_sol is None:  # Checks if solver failed or no feasible root exists.
            continue  # Skips invalid candidate.

        r = r_sol  # Stores solved radius for readability.

        # Now we have a full geometry (r, l, e).  # Notes full design tuple is available.
        # Check custom constraints if any string eval is needed (unsafe but flexible)  # Notes potential future extensibility.
        # For now, hardcode the critical ones or parse safe strings.  # Explains current conservative constraint path.
        # "l >= 2.5*r"  # Names enforced geometric ratio constraint.
        if l < 2.5 * r:  # Enforces minimum rod-to-crank ratio.
            continue  # Skips candidate that violates ratio constraint.

        # Verify kinematics (QRR, exact ROM check)  # Marks full metric validation step.
        metrics = kinematics.calculate_metrics(r, l, e)  # Computes exact metrics for solved geometry.

        if not metrics["valid"]:  # Rejects geometries failing kinematic validity checks.
            continue  # Skips invalid geometry.

        # Check QRR  # Marks quality ratio filtering.
        qrr = metrics["QRR"]  # Reads computed QRR metric.
        if not (qrr_range["min"] <= qrr <= qrr_range["max"]):  # Tests QRR against configured limits.
            continue  # Skips candidate outside QRR bounds.

        # Passed!  # Marks acceptance path.
        design = {  # Builds output record for accepted design.
            "r": r,  # Stores solved crank radius.
            "l": l,  # Stores rod length.
            "e": e,  # Stores offset.
            "ROM": metrics["ROM"],  # Stores computed ROM.
            "QRR": qrr,  # Stores computed QRR.
            "theta_min": metrics["theta_retracted"],  # Stores minimum angle endpoint.
            "theta_max": metrics["theta_extended"],  # Stores maximum angle endpoint.
        }
        valid_designs.append(design)  # Adds accepted design to results.

    return valid_designs  # Returns all valid designs found.
