import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        Dictionary containing configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_baseline_config() -> Dict[str, Any]:
    """
    Load the baseline configuration from configs/generate/baseline.yaml.
    
    Returns:
        Dictionary containing baseline configuration.
    """
    # Assuming code is run from project root, or we can find it relative to this file
    # This file is in src/mech390/config.py
    # Config is in configs/generate/baseline.yaml
    # relative path: ../../../configs/generate/baseline.yaml
    
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.parent # src -> <root>
    config_path = project_root / 'configs' / 'generate' / 'baseline.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Baseline config not found at {config_path}")
        
    return load_config(str(config_path))
