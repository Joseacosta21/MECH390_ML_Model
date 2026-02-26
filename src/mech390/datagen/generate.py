"""
Data Generation Orchestrator.
Main entry point for generating mechanism datasets.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

from mech390.datagen import sampling, stage1_kinematic, stage2_embodiment

# Try to import physics engine, or mock if not available
try:
    from mech390.physics import engine
except ImportError:
    engine = None
    logging.warning("mech390.physics.engine not found. Physics evaluation will be skipped/mocked.")

# Logger setup
logger = logging.getLogger(__name__)

class DatasetResult:
    """Container for generation results."""
    def __init__(self, all_cases: pd.DataFrame, pass_cases: pd.DataFrame, summary: Dict):
        self.all_cases = all_cases
        self.pass_cases = pass_cases
        self.summary = summary

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
    
    # 1. Stage 1: Kinematic Synthesis
    # -------------------------------
    logger.info("Starting Stage 1: Kinematic Synthesis...")
    
    # We pass the full config to stage1, letting it parse what it needs
    # Or we can parse here. Stage 1 function takes (config, n_attempts).
    # We should ensure config has 'sampling' settings.
    
    valid_2d = stage1_kinematic.generate_valid_2d_mechanisms(config)
    
    n_stage1 = len(valid_2d)
    logger.info(f"Stage 1 complete. {n_stage1} valid 2D mechanisms generated.")
    
    if n_stage1 == 0:
        logger.warning("No valid 2D mechanisms found. Aborting.")
        return DatasetResult(pd.DataFrame(), pd.DataFrame(), {'error': 'No valid 2D'})

    # 2. Stage 2: Embodiment
    # ----------------------
    logger.info("Starting Stage 2: Embodiment Expansion...")
    
    candidates_3d = stage2_embodiment.expand_to_3d(valid_2d, config)
    
    n_stage2 = len(candidates_3d)
    logger.info(f"Stage 2 complete. {n_stage2} 3D candidates ready for evaluation.")

    # 3. Physics Evaluation
    # ---------------------
    logger.info("Starting Physics Evaluation...")
    
    results = []
    
    # Limits
    limits = config.get('limits', {})
    sigma_allow = limits.get('sigma_allow', 1e20) # Default huge
    tau_allow = limits.get('tau_allow', 1e20)
    safety_factor = limits.get('safety_factor', 1.0)
    
    sigma_limit = sigma_allow / safety_factor
    tau_limit = tau_allow / safety_factor # Assuming shear limit also uses SF
    
    for design in candidates_3d:
        # Construct a full 'case' dict
        case = design.copy()
        
        # Call physics engine
        # We need to map our design dict to what engine expects.
        # Assuming engine.evaluate(design_dict) -> metrics_dict
        
        metrics = {}
        if engine:
            try:
                # This is the integration point.
                # Engine should return sigma_max, tau_max, etc.
                metrics = engine.evaluate_design(design) 
                
                # If engine returns None or indicates failure:
                if metrics is None:
                     metrics = {'valid_physics': False}
                else:
                     metrics['valid_physics'] = True
            except Exception as e:
                logger.error(f"Physics evaluation failed for design {design}: {e}")
                metrics = {'valid_physics': False, 'error': str(e)}
        else:
            # Mock physics for now if engine missing
            metrics = {
                'valid_physics': True,
                'sigma_max': 0.0,
                'tau_max': 0.0,
                'theta_sigma_max': 0.0,
                'theta_tau_max': 0.0
            }
            
        # Merge metrics
        case.update(metrics)
        
        # 4. Pass/Fail Check
        # ------------------
        if case.get('valid_physics', False):
            s_max = case.get('sigma_max', 0.0)
            t_max = case.get('tau_max', 0.0)
            
            # Utilization
            # Avoid div by zero
            u_sigma = s_max / sigma_limit if sigma_limit > 0 else 0
            u_tau = t_max / tau_limit if tau_limit > 0 else 0
            
            utilization = max(u_sigma, u_tau)
            case['utilization'] = utilization
            
            # Pass if util <= 1.0
            passed = 1 if utilization <= 1.0 else 0
            case['pass_fail'] = passed
        else:
            case['utilization'] = -1.0
            case['pass_fail'] = 0
            
        results.append(case)
        
    # Convert to DataFrames
    df_all = pd.DataFrame(results)
    
    if not df_all.empty and 'pass_fail' in df_all.columns:
        df_pass = df_all[df_all['pass_fail'] == 1].copy()
    else:
        df_pass = pd.DataFrame()
        
    # Stats
    summary = {
        'n_stage1': n_stage1,
        'n_stage2': n_stage2,
        'n_evaluated': len(df_all),
        'n_passed': len(df_pass),
        'pass_rate': len(df_pass) / len(df_all) if len(df_all) > 0 else 0.0,
        'time_taken_sec': time.time() - start_time
    }
    
    logger.info(f"Generation complete. Passed: {len(df_pass)} / {len(df_all)}")
    
    return DatasetResult(df_all, df_pass, summary)