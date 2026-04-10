"""
Sampling module for generating mechanism candidates.
No physics imports allowed here.
"""

import random
import numpy as np
from typing import Union, List, Dict, Any, Optional

def sample_scalar(range_def: Union[Dict, List, float, int], seed: Optional[int] = None) -> float:
    """
    Samples a scalar value based on the definition.
    
    Args:
        range_def: 
            - {'min': float, 'max': float}: Uniform sample between min and max.
            - [v1, v2, ...]: Discrete choice.
            - s (float/int): Constant value.
        seed: Random seed (only used if this is the only call; prefer setting global seed outside).
        
    Returns:
        float: Sampled value.
    """
    if isinstance(range_def, dict):
        if 'min' in range_def and 'max' in range_def:
            return random.uniform(range_def['min'], range_def['max'])
        else:
            # Fallback or other types (log-uniform could be added here)
            raise ValueError(f"Unknown range definition dict: {range_def}")
            
    elif isinstance(range_def, list):
        return random.choice(range_def)
        
    elif isinstance(range_def, (float, int)):
        return float(range_def)
        
    else:
        raise TypeError(f"Unsupported range definition type: {type(range_def)}")


class LatinHypercubeSampler:
    """
    Latin Hypercube Sampler for parameter sets.
    """
    def __init__(self, param_ranges: Dict[str, Dict[str, float]], n_samples: int, seed: Optional[int] = None):
        """
        Args:
            param_ranges: Dict of canonical names to {'min': ..., 'max': ...}.
            n_samples: Number of samples to generate.
            seed: Random seed.
        """
        self.param_ranges = param_ranges
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> List[Dict[str, float]]:
        """
        Generates n_samples using geometric LHS (stratified sampling).
        """
        from scipy.stats import qmc
        
        dims = len(self.param_ranges)
        names = sorted(list(self.param_ranges.keys())) # Sort for determinism
        
        # Create sampler
        sampler = qmc.LatinHypercube(d=dims, seed=self.seed)
        sample_unit_hypercube = sampler.random(n=self.n_samples)
        
        # Scale to bounds
        l_bounds = []
        u_bounds = []
        
        for name in names:
            r = self.param_ranges[name]
            # Handle fixed values if passed in ranges (though LHS usually implies ranges)
            if isinstance(r, (float, int)):
                l_bounds.append(float(r))
                u_bounds.append(float(r))
            elif isinstance(r, dict) and 'min' in r and 'max' in r:
                l_bounds.append(r['min'])
                u_bounds.append(r['max'])
            else:
                 raise ValueError(f"LHS only supports min/max dicts or constants. Got: {r}")
        
        samples_scaled = qmc.scale(sample_unit_hypercube, l_bounds, u_bounds)
        
        results = []
        for row in samples_scaled:
            res_dict = {}
            for i, name in enumerate(names):
                res_dict[name] = float(row[i])
            results.append(res_dict)
            
        return results

def get_sampler(method: str, param_ranges: Dict, n_samples: int, seed: int):
    """Factory to get a sampler/generator iterable."""
    if method == "latin_hypercube":
        lhs = LatinHypercubeSampler(param_ranges, n_samples, seed)
        return lhs.generate()
    elif method == "random":
        # Generator for random sampling
        rng = random.Random(seed)
        def random_gen():
           for _ in range(n_samples):
               sample = {}
               for k, v in param_ranges.items():
                   # We re-use logic but need careful seeding/state if done this way
                   # Better to just use local random state
                   if isinstance(v, dict) and 'min' in v:
                       sample[k] = rng.uniform(v['min'], v['max'])
                   elif isinstance(v, (float, int)):
                       sample[k] = float(v)
                   # ... handle list
               yield sample
        return list(random_gen()) # Return list to match LHS
    else:
        raise ValueError(f"Unknown sampling method: {method}")