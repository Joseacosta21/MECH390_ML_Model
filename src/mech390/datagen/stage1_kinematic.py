"""
Stage 1: 2D Kinematic Synthesis and Filtering.
Implements the "Sample 2, Solve 1" strategy to enforce ROM constraints.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, List, Optional, Tuple, Any
import logging

from mech390.physics import kinematics
from mech390.datagen import sampling

from mech390.config import get_baseline_config

# Set up logger
logger = logging.getLogger(__name__)

# Load baseline config for defaults
BASELINE_CONFIG = get_baseline_config()
DEFAULT_ROM = BASELINE_CONFIG['operating']['ROM']
R_MIN = BASELINE_CONFIG['geometry']['r']['min']
R_MAX = BASELINE_CONFIG['geometry']['r']['max']

def solve_for_r_given_rom(l: float, e: float, target_rom: float = DEFAULT_ROM, 
                          r_min: float = R_MIN, r_max: float = R_MAX, 
                          tol: float = 1e-4) -> Optional[float]:
    """
    Numerically solves for crank radius r that gives the target ROM,
    given rod length l and offset e.
    
    Args:
        l: Rod length.
        e: Offset (D).
        target_rom: Desired Range of Motion.
        r_min, r_max: Bounds for r.
        tol: Tolerance for ROM error.
        
    Returns:
        r_solution: The found radius r, or None if no solution in bounds.
    """
    
    def objective(r_try):
        # Kinematics check first
        # We need to ensure valid geometry before computing ROM
        # Basic check: l > r + |e| is a safe heuristic for full rotation, 
        # but let kinematics module handle exact checks.
        
        # We wrap kinematics call in try-except to handle invalid geometries gracefully
        # during the solver's exploration
        try:
             metrics = kinematics.calculate_metrics(r_try, l, e)
             if not metrics['valid']:
                 # Penalty or indicator of invalidity. 
                 # If invalid, it usually means locking, so ROM is undefined or 0?
                 # Returning a large error might confuse solver if not monotonic.
                 return -1.0 # arbitrary indicator?
                 
             return metrics['ROM'] - target_rom
        except ValueError:
             return -1.0 # invalid

    # Clamp r_max to ensure valid geometry: l > r + |e| => r < l - |e|
    # Heuristic: subtract small epsilon
    r_geo_max = l - abs(e) - 0.001
    if r_max > r_geo_max:
        r_max = r_geo_max
        
    if r_max < r_min:
        return None # Infeasible range

    # Check bounds first to see if they bracket the zero
    try:
        y_min = objective(r_min)
        y_max = objective(r_max)
    except Exception:
        return None

    # We expect ROM to increase with r roughly monotonically (ROM ~ 2r).
    # If objective(r_min) > 0, then even smallest r is too big -> Fail.
    # If objective(r_max) < 0, then even largest r is too small -> Fail.
    
    
    if y_min > 0:
        return None # r_min gives too much ROM
    if y_max < 0:
        return None # r_max gives too little ROM
        
    # If valid brackets, solve
    try:
        r_sol = brentq(objective, r_min, r_max, xtol=tol)
        return r_sol
    except Exception:
        return None


def generate_valid_2d_mechanisms(config: Dict[str, Any], n_attempts: int = 100000) -> List[Dict[str, Any]]:
    """
    Generates a list of valid 2D mechanisms.
    
    Strategy:
      1. Sample l and e from config ranges.
      2. Solve for r to match target ROM.
      3. Check if r is within config bounds.
      4. Check QRR constraints.
      
    Args:
        config: Configuration dictionary (must contain 'geometry', 'operating', 'material' etc.)
        n_attempts: Max attempts to try.
        
    Returns:
        List of dicts with valid geometry {r, l, e, ...}
    """
    valid_designs = []
    
    # Extract settings
    geo_ranges = config.get('geometry', {})
    op_settings = config.get('operating', {})
    
    # Ranges
    l_range = geo_ranges.get('l', {'min': 0.1, 'max': 1.0})
    e_range = geo_ranges.get('e', {'min': 0.0, 'max': 0.5})
    r_range = geo_ranges.get('r', {'min': 0.01, 'max': 1.0}) # Bounds for solver
    
    # Targets
    target_rom = op_settings.get('ROM', 0.25)
    qrr_range = op_settings.get('QRR', {'min': 1.0, 'max': 100.0})
    
    # Setup sampler for l and e
    # We use random sampling for simplicity in this loop, or we could use LHS pre-generation.
    # If config says LHS, we should ideally use that.
    # Let's support the sampling config.
    
    samp_config = config.get('sampling', {})
    
    # Prepare ranges for sampling ONLY l and e
    # We do NOT sample r, we solve for it.
    param_ranges_to_sample = {
        'l': l_range,
        'e': e_range
    }
    
    # Get candidate samples for l and e
    # Note: If n_samples is specified in config, we try to produce that many VALID designs?
    # Or we treat n_samples as "candidates to try"?
    # Usually "n_samples" in LHS means number of candidates generated.
    target_n_samples = samp_config.get('n_samples', 1000)
    
    # Generate candidates
    # We are generating (l, e) pairs
    candidates = sampling.get_sampler(
        method=samp_config.get('method', 'random'),
        param_ranges=param_ranges_to_sample,
        n_samples=target_n_samples,
        seed=config.get('random_seed', 42)
    )
    
    for cand in candidates:
        l = cand['l']
        e = cand['e']
        
        # Constraints from config (strings)
        # "l >= 2.5*r" -> This involves r, so we can't check it yet? 
        # Or we can check "l >= 2.5 * (ROM/2)" as a rough check?
        # Better to solve r first, then check.
        
        # Solve for r
        r_min = r_range['min'] if isinstance(r_range, dict) else 0.01
        r_max = r_range['max'] if isinstance(r_range, dict) else 1.0
        
        r_sol = solve_for_r_given_rom(l, e, target_rom, r_min, r_max)
        
        if r_sol is None:
            continue
            
        r = r_sol
        
        # Now we have a full geometry (r, l, e).
        # Check custom constraints if any string eval is needed (unsafe but flexible)
        # For now, hardcode the critical ones or parse safe strings.
        # "l >= 2.5*r"
        if l < 2.5 * r:
            continue
            
        # Verify kinematics (QRR, exact ROM check)
        metrics = kinematics.calculate_metrics(r, l, e)
        
        if not metrics['valid']:
            continue
            
        # Check QRR
        qrr = metrics['QRR']
        if not (qrr_range['min'] <= qrr <= qrr_range['max']):
            continue
            
        # Passed!
        design = {
            'r': r,
            'l': l,
            'e': e,
            'ROM': metrics['ROM'],
            'QRR': qrr,
            'theta_min': metrics['theta_retracted'],
            'theta_max': metrics['theta_extended']
        }
        valid_designs.append(design)
        
    return valid_designs
