import sys
import os
import json
import logging

# Ensure src is in path
sys.path.append(os.path.abspath('src'))

from mech390.datagen import stage1_kinematic
from mech390.config import get_baseline_config

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_debug():
    print("Loading baseline configuration...")
    config = get_baseline_config()
    
    # Optional: Override n_samples to be smaller for quick debugging if needed
    # But let's respect the config for now unless it's huge
    # config['sampling']['n_samples'] = 100 
    
    print("\n--- Configuration Summary ---")
    print(f"Target ROM: {config['operating']['ROM']}")
    print(f"Geometry Ranges: {config['geometry']}")
    print(f"Sampling Method: {config['sampling']['method']}")
    print(f"Target Samples: {config['sampling']['n_samples']}")
    print("-----------------------------\n")

    print("Running Stage 1: Kinematic Synthesis...")
    try:
        valid_designs = stage1_kinematic.generate_valid_2d_mechanisms(config)
    except Exception as e:
        print(f"Error running Stage 1: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nStage 1 Complete.")
    print(f"Total Valid Designs Found: {len(valid_designs)}")
    
    if valid_designs:
        print("\n--- First 5 Designs ---")
        for i, design in enumerate(valid_designs[:5]):
            print(f"Design {i+1}:")
            print(json.dumps(design, indent=2))
            print("-" * 20)
            
        # Basic stats
        rs = [d['r'] for d in valid_designs]
        ls = [d['l'] for d in valid_designs]
        es = [d['e'] for d in valid_designs]
        
        print("\n--- Statistics ---")
        print(f"r: min={min(rs):.4f}, max={max(rs):.4f}, avg={sum(rs)/len(rs):.4f}")
        print(f"l: min={min(ls):.4f}, max={max(ls):.4f}, avg={sum(ls)/len(ls):.4f}")
        print(f"e: min={min(es):.4f}, max={max(es):.4f}, avg={sum(es)/len(es):.4f}")

if __name__ == "__main__":
    run_debug()
