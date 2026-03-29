"""
Data Generation Orchestrator.
Main entry point for generating mechanism datasets.
"""

import logging
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from mech390.datagen import stage1_kinematic, stage2_embodiment

# Try to import physics engine and mass_properties, or mock if not available
try:
    from mech390.physics import engine
    from mech390.physics import mass_properties as mp
except ImportError:
    engine = None
    mp = None
    logging.warning("mech390.physics.engine not found. Physics evaluation will be skipped/mocked.")

# Logger setup
logger = logging.getLogger(__name__)

class DatasetResult:
    """Container for generation results."""
    def __init__(self, all_cases: pd.DataFrame, pass_cases: pd.DataFrame, summary: Dict):
        self.all_cases = all_cases
        self.pass_cases = pass_cases
        self.summary = summary


def _evaluate_physics(design: Dict[str, Any], physics_engine) -> Dict[str, Any]:
    """Evaluate a single design with the physics engine or fallback mock."""
    if physics_engine:
        try:
            metrics = physics_engine.evaluate_design(design)
            if metrics is None:
                return {"valid_physics": False}
            metrics["valid_physics"] = True
            return metrics
        except Exception as exc:
            logger.error(f"Physics evaluation failed for design {design}: {exc}")
            return {"valid_physics": False, "error": str(exc)}

    # Mock physics for now if engine missing
    return {
        "valid_physics": True,
        "sigma_max": 0.0,
        "tau_max": 0.0,
        "theta_sigma_max": 0.0,
        "theta_tau_max": 0.0,
    }


def _apply_limits(case: Dict[str, Any], sigma_limit: float, tau_limit: float) -> Dict[str, Any]:
    """Compute utilization and pass/fail for a design case."""
    if case.get("valid_physics", False):
        s_max = case.get("sigma_max", 0.0)
        t_max = case.get("tau_max", 0.0)

        u_sigma = s_max / sigma_limit if sigma_limit > 0 else 0
        u_tau = t_max / tau_limit if tau_limit > 0 else 0
        utilization = max(u_sigma, u_tau)

        case["utilization"] = utilization
        case["pass_fail"] = 1 if utilization <= 1.0 else 0
    else:
        case["utilization"] = -1.0
        case["pass_fail"] = 0

    return case


def _build_summary(
    n_stage1: int,
    n_stage2: int,
    df_all: pd.DataFrame,
    df_pass: pd.DataFrame,
    start_time: float,
) -> Dict[str, Any]:
    """Build generation summary dictionary."""
    return {
        "n_stage1": n_stage1,
        "n_stage2": n_stage2,
        "n_evaluated": len(df_all),
        "n_passed": len(df_pass),
        "pass_rate": len(df_pass) / len(df_all) if len(df_all) > 0 else 0.0,
        "time_taken_sec": time.time() - start_time,
    }


def _iter_stage2_designs(valid_2d: Any, config: Dict[str, Any]):
    """Prefer streaming Stage-2 expansion when available; fallback to list API."""
    iter_fn = getattr(stage2_embodiment, "iter_expand_to_3d", None)
    if callable(iter_fn):
        return iter_fn(valid_2d, config)
    return iter(stage2_embodiment.expand_to_3d(valid_2d, config))


def generate_dataset(config: Dict[str, Any], seed: int = None) -> DatasetResult:
    """
    Main orchestration function to generate dataset.
    
    Args:
        config: Configuration dictionary.
        seed: Random seed (overrides config if provided).
        
    Returns:
        DatasetResult object.
    """
    start_time = time.time()
    
    # Setup seed
    run_seed = seed if seed is not None else config.get('random_seed', 42)
    np.random.seed(run_seed)
    
    logger.info("Starting Stage 1: Kinematic Synthesis...")
    valid_2d = stage1_kinematic.generate_valid_2d_mechanisms(config)
    
    n_stage1 = len(valid_2d)
    logger.info(f"Stage 1 complete. {n_stage1} valid 2D mechanisms generated.")
    
    if n_stage1 == 0:
        logger.warning("No valid 2D mechanisms found. Aborting.")
        return DatasetResult(pd.DataFrame(), pd.DataFrame(), {'error': 'No valid 2D'})

    logger.info("Starting Stage 2: Embodiment Expansion...")

    logger.info("Starting Physics Evaluation...")

    results = []

    limits = config.get('limits', {})
    sigma_allow = limits.get('sigma_allow', 1e20)
    tau_allow = limits.get('tau_allow', 1e20)
    safety_factor = limits.get('safety_factor', 1.0)
    mu_default = float(config.get('operating', {}).get('mu', 0.0))

    sigma_limit = sigma_allow / safety_factor
    tau_limit = tau_allow / safety_factor

    # Compute omega once — same for every design in this run.
    rpm = float(config.get('operating', {}).get('RPM', 30))
    omega = rpm * 2.0 * np.pi / 60.0

    n_stage2 = 0
    for design in _iter_stage2_designs(valid_2d, config):
        n_stage2 += 1
        design_eval = design.copy()
        design_eval['omega'] = omega
        design_eval.setdefault("mu", mu_default)

        # Merge mass properties so the physics engine has real masses/inertias.
        if mp is not None:
            try:
                mass_props = mp.compute_design_mass_properties(design_eval, config)
                design_eval.update(mass_props)
            except Exception as exc:
                logger.warning("Mass properties failed for design #%d: %s", n_stage2, exc)

        # Inject material properties and operating params required by
        # stresses.py, fatigue.py, and buckling.py.
        material_cfg = config.get('material', {})
        design_eval['E']             = float(material_cfg.get('E',             73.1e9))
        design_eval['S_ut']          = float(material_cfg.get('S_ut',          483e6))
        design_eval['S_y']           = float(material_cfg.get('S_y',           345e6))
        design_eval['S_prime_e']     = float(material_cfg.get('S_prime_e',     130e6))
        design_eval['sigma_f_prime'] = float(material_cfg.get('sigma_f_prime', 807e6))
        design_eval['n_rpm']         = rpm
        design_eval['total_cycles']  = float(
            config.get('operating', {}).get('TotalCycles', 18720000)
        )

        case = design_eval.copy()
        metrics = _evaluate_physics(design_eval, engine)
        case.update(metrics)
        case = _apply_limits(case, sigma_limit, tau_limit)
        results.append(case)

    logger.info(f"Stage 2 complete. {n_stage2} 3D candidates ready for evaluation.")

    df_all = pd.DataFrame(results)

    if not df_all.empty and 'pass_fail' in df_all.columns:
        df_pass = df_all[df_all['pass_fail'] == 1].copy()
    else:
        df_pass = pd.DataFrame()

    summary = _build_summary(n_stage1, n_stage2, df_all, df_pass, start_time)
    
    logger.info(f"Generation complete. Passed: {len(df_pass)} / {len(df_all)}")
    
    return DatasetResult(df_all, df_pass, summary)
