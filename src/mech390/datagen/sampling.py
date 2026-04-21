"""
Sampling module for generating mechanism candidates.
No physics imports allowed here.
"""

import random
import numpy as np
from typing import Union, List, Dict, Any, Optional


# picks a single value from a range dict, discrete list, or constant
def sample_scalar(range_def: Union[Dict, List, float, int], seed: Optional[int] = None) -> float:
    if isinstance(range_def, dict):
        if 'min' in range_def and 'max' in range_def:
            return random.uniform(range_def['min'], range_def['max'])
        else:
            raise ValueError(f"Unknown range definition dict: {range_def}")

    elif isinstance(range_def, list):
        return random.choice(range_def)

    elif isinstance(range_def, (float, int)):
        return float(range_def)

    else:
        raise TypeError(f"Unsupported range definition type: {type(range_def)}")


class LatinHypercubeSampler:
    """Sampler that spreads samples evenly across each parameter's range instead of clustering randomly."""

    # sets up the sampler with parameter bounds and sample count
    def __init__(self, param_ranges: Dict[str, Dict[str, float]], n_samples: int, seed: Optional[int] = None):
        self.param_ranges = param_ranges
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # generates n_samples spread evenly across all parameter ranges
    def generate(self) -> List[Dict[str, float]]:
        from scipy.stats import qmc

        dims = len(self.param_ranges)
        names = sorted(list(self.param_ranges.keys()))  # sort so output order is always the same

        sampler = qmc.LatinHypercube(d=dims, seed=self.seed)
        sample_unit_hypercube = sampler.random(n=self.n_samples)

        l_bounds = []
        u_bounds = []

        for name in names:
            r = self.param_ranges[name]
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


# returns a list of samples using the requested method (latin_hypercube or random)
def get_sampler(method: str, param_ranges: Dict, n_samples: int, seed: int):
    if method == "latin_hypercube":
        lhs = LatinHypercubeSampler(param_ranges, n_samples, seed)
        return lhs.generate()
    elif method == "random":
        rng = random.Random(seed)
        def random_gen():
            for _ in range(n_samples):
                sample = {}
                for k, v in param_ranges.items():
                    if isinstance(v, dict) and 'min' in v:
                        sample[k] = rng.uniform(v['min'], v['max'])
                    elif isinstance(v, (float, int)):
                        sample[k] = float(v)
                yield sample
        return list(random_gen())
    else:
        raise ValueError(f"Unknown sampling method: {method}")
