
import logging
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath('src'))

from mech390.datagen import generate

# Configure logging
logging.basicConfig(level=logging.INFO)

config = {
    'geometry': {
        'r': {'min': 0.1, 'max': 0.2},
        'l': {'min': 0.3, 'max': 0.5},
        'e': {'min': 0.0, 'max': 0.1},
        'widths': {
            'width_r': {'min': 0.012, 'max': 0.02},
            'width_l': {'min': 0.012, 'max': 0.02},
        },
        'thicknesses': {
            'thickness_r': {'min': 0.008, 'max': 0.015},
            'thickness_l': {'min': 0.008, 'max': 0.015},
        },
        'pin_diameters': {
            'pin_diameter_A': {'min': 0.006, 'max': 0.011},
            'pin_diameter_B': {'min': 0.006, 'max': 0.011},
            'pin_diameter_C': {'min': 0.006, 'max': 0.011},
        },
    },
    'operating': {
        'ROM': 0.25, # Target approx 2*r
        'QRR': {'min': 1.0, 'max': 5.0} # Loose constraint
    },
    'sampling': {
        'method': 'random',
        'n_samples': 50,
        'n_variants_per_2d': 3,
    },
    'limits': {
        'sigma_allow': 200e6,
        'tau_allow': 100e6,
        'safety_factor': 1.5
    },
    'random_seed': 123
}

print("Running generation...")
result = generate.generate_dataset(config)

print("Summary:", result.summary)
print("All cases rows:", len(result.all_cases))
print("Pass cases rows:", len(result.pass_cases))

if len(result.all_cases) > 0:
    print("Sample row:", result.all_cases.iloc[0].to_dict())
else:
    print("No cases generated.")
