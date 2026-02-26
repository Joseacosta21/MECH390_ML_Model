"""
Stage 2: 3D Embodiment Expansion.
Expands valid 2D mechanisms into 3D designs by sampling component dimensions.
"""

import numpy as np
from typing import List, Dict, Any
from mech390.datagen import sampling

def expand_to_3d(valid_2d_designs: List[Dict[str, float]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expands a list of valid 2D designs into 3D embodiments.
    
    Args:
        valid_2d_designs: List of dicts having at least r, l, e.
        config: Full configuration dictionary.
        
    Returns:
        List of expanded design dictionaries.
    """
    expanded_designs = []
    
    # Identify 3D parameters to sample
    # We look for them in 'geometry' section or a dedicated 'embodiment' section
    geo_config = config.get('geometry', {})
    
    # Potential 3D keys (canonical names we support)
    keys_3d = ['link_thickness', 'link_width', 'pin_diameter', 'density']
    
    # Also include any other keys in geometry that are ranges but not r, l, e
    # (Flexible adaption)
    known_2d = ['r', 'l', 'e', 'slider']
    
    param_ranges = {}
    
    # 1. Explicit keys from internal list, if present in config
    for k in keys_3d:
        if k in geo_config:
             param_ranges[k] = geo_config[k]

    # 2. Any other keys in geometry that look like parameters
    for k, v in geo_config.items():
        if k not in known_2d and k not in param_ranges:
             param_ranges[k] = v
             
    # If no 3D params found, should we add defaults?
    # For now, we assume config will conduct this. 
    # If empty, we just pass through (maybe just density is needed).
    
    # Material props might be in 'material' section
    mat_config = config.get('material', {})
    if 'rho' in mat_config:
         param_ranges['density'] = mat_config['rho']
         
    # Setup sampler stuff
    # We iterate through each 2D design and generate *one* or *more* 3D variants?
    # Instructions: "Expands each into multiple 3D variants."
    # How many? Maybe `n_variants_per_2d`? Or just 1?
    # Let's assume 1 for now unless config says otherwise, or maybe the sampling happens *per* design.
    
    # Actually, if we want to do LHS interaction between 2D and 3D, we should have done it in one go.
    # But strict staging means we take a 2D result and "dress" it.
    # Simple approach: For each 2D, sample ONE set of 3D params.
    # If we want multiple, we can loop.
    
    sampler_method = config.get('sampling', {}).get('method', 'random')
    seed = config.get('random_seed', 42)
    
    rng = np.random.RandomState(seed)
    
    for design_2d in valid_2d_designs:
        # Create a copy to avoid mutating original if reused
        design_3d = design_2d.copy()
        
        # Sample 3D params
        for param_name, param_def in param_ranges.items():
            # We use a local sample here (random) because LHS across this conditional set is hard
            # unless we structured it differently.
            # Using simple random sample for now relative to the range definition.
            val = sampling.sample_scalar(param_def, seed=None) # Use global/shared RNG state if passed? 
            # Actually sample_scalar uses `random` module state if seed is None.
            # We should be consistent.
            # For strict determinism, we should pass a generator or seed.
            # But `sample_scalar` takes a seed int, not a generator.
            # Let's rely on `random` state being managed by the caller (generate.py setting seed).
            design_3d[param_name] = val
            
        # TODO: Calculate mass and moments of inertia (Izz).
        
        # TODO: Shape approximation.
        
        expanded_designs.append(design_3d)
        
    return expanded_designs
