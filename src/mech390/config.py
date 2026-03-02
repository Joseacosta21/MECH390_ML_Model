import re
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

_NUMERIC_RE = re.compile(
    r"^[\+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][\+\-]?\d+)?$"
)

_INT_RE = re.compile(r"^[\+\-]?\d+$")


def _coerce_numeric(value: Any) -> Any:
    """Convert numeric-looking strings to int/float; leave other values unchanged."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or not _NUMERIC_RE.match(stripped):
        return value

    try:
        if _INT_RE.match(stripped):
            return int(stripped)
        return float(stripped)
    except ValueError:
        return value


def _normalize_config_values(value: Any) -> Any:
    """Recursively normalize config values (dict/list/scalar)."""
    if isinstance(value, dict):
        return {k: _normalize_config_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_config_values(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize_config_values(v) for v in value)
    return _coerce_numeric(value)


def normalize_range_def(range_def: Any, name: str) -> Dict[str, float]:
    """
    Normalize a range definition to canonical {'min': float, 'max': float}.
    Supports {min,max}, [min,max], (min,max), or scalar.
    """
    if isinstance(range_def, dict) and "min" in range_def and "max" in range_def:
        min_val = float(range_def["min"])
        max_val = float(range_def["max"])
    elif isinstance(range_def, (list, tuple)) and len(range_def) == 2:
        min_val = float(range_def[0])
        max_val = float(range_def[1])
    elif isinstance(range_def, (int, float)):
        min_val = float(range_def)
        max_val = float(range_def)
    else:
        raise ValueError(
            f"Range for '{name}' must be {{min,max}}, [min,max], or scalar. "
            f"Got: {range_def!r}"
        )

    if min_val > max_val:
        raise ValueError(
            f"Invalid range for '{name}': min ({min_val}) > max ({max_val})."
        )

    return {"min": min_val, "max": max_val}


def _validate_min_max_ranges(value: Any, path: str = "config") -> None:
    """Recursively validate only explicit {min,max} dict ranges."""
    if isinstance(value, dict):
        if "min" in value and "max" in value:
            normalize_range_def(value, path)
        for key, child in value.items():
            _validate_min_max_ranges(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _validate_min_max_ranges(child, f"{path}[{idx}]")


def _first_present_range(
    primary: Mapping[str, Any],
    primary_key: str,
    fallback: Mapping[str, Any],
    fallback_key: str,
    path_hint: str,
) -> Dict[str, float]:
    """Pick range from primary mapping, otherwise fallback mapping, then normalize."""
    if primary_key in primary:
        return normalize_range_def(primary[primary_key], f"{path_hint}.{primary_key}")
    if fallback_key in fallback:
        return normalize_range_def(fallback[fallback_key], f"geometry.{fallback_key}")
    raise ValueError(
        f"Missing Stage-2 parameter range '{fallback_key}'. "
        f"Expected '{path_hint}.{primary_key}' or 'geometry.{fallback_key}'."
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        Dictionary containing configuration.
    """
    with open(config_path, "r") as f:
        parsed = yaml.safe_load(f)

    normalized = _normalize_config_values(parsed)
    _validate_min_max_ranges(normalized)
    return normalized


def get_stage2_param_ranges(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Extract canonical Stage-2 embodiment parameter ranges.
    Precedence: nested groups in geometry -> legacy flat geometry keys.
    """
    geometry = config.get("geometry", {})
    if not isinstance(geometry, dict):
        raise ValueError("Config must contain a 'geometry' mapping for Stage 2.")

    widths = geometry.get("widths", {})
    thicknesses = geometry.get("thicknesses", {})
    pin_diameters = geometry.get("pin_diameters", {})

    if widths is None:
        widths = {}
    if thicknesses is None:
        thicknesses = {}
    if pin_diameters is None:
        pin_diameters = {}

    if not isinstance(widths, dict) or not isinstance(thicknesses, dict) or not isinstance(pin_diameters, dict):
        raise ValueError(
            "Stage-2 grouped keys must be mappings: geometry.widths, "
            "geometry.thicknesses, geometry.pin_diameters."
        )

    return {
        "width_r": _first_present_range(
            primary=widths,
            primary_key="width_r",
            fallback=geometry,
            fallback_key="width_r",
            path_hint="geometry.widths",
        ),
        "width_l": _first_present_range(
            primary=widths,
            primary_key="width_l",
            fallback=geometry,
            fallback_key="width_l",
            path_hint="geometry.widths",
        ),
        "thickness_r": _first_present_range(
            primary=thicknesses,
            primary_key="thickness_r",
            fallback=geometry,
            fallback_key="thickness_r",
            path_hint="geometry.thicknesses",
        ),
        "thickness_l": _first_present_range(
            primary=thicknesses,
            primary_key="thickness_l",
            fallback=geometry,
            fallback_key="thickness_l",
            path_hint="geometry.thicknesses",
        ),
        "pin_diameter_A": _first_present_range(
            primary=pin_diameters,
            primary_key="pin_diameter_A",
            fallback=geometry,
            fallback_key="pin_diameter_A",
            path_hint="geometry.pin_diameters",
        ),
        "pin_diameter_B": _first_present_range(
            primary=pin_diameters,
            primary_key="pin_diameter_B",
            fallback=geometry,
            fallback_key="pin_diameter_B",
            path_hint="geometry.pin_diameters",
        ),
        "pin_diameter_C": _first_present_range(
            primary=pin_diameters,
            primary_key="pin_diameter_C",
            fallback=geometry,
            fallback_key="pin_diameter_C",
            path_hint="geometry.pin_diameters",
        ),
    }


def get_stage2_sampling_settings(config: Dict[str, Any]) -> Dict[str, int]:
    """Extract Stage-2 sampling settings with defaults and validation."""
    sampling = config.get("sampling", {})
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, dict):
        raise ValueError("Config 'sampling' section must be a mapping.")

    n_variants = int(sampling.get("n_variants_per_2d", 1))
    if n_variants <= 0:
        raise ValueError("sampling.n_variants_per_2d must be > 0.")

    max_attempts_raw = sampling.get("stage2_max_attempts_per_2d")
    if max_attempts_raw is None:
        max_attempts = max(100, 50 * n_variants)
    else:
        max_attempts = int(max_attempts_raw)

    if max_attempts <= 0:
        raise ValueError("sampling.stage2_max_attempts_per_2d must be > 0.")
    if max_attempts < n_variants:
        raise ValueError(
            "sampling.stage2_max_attempts_per_2d must be >= sampling.n_variants_per_2d."
        )

    return {
        "n_variants_per_2d": n_variants,
        "stage2_max_attempts_per_2d": max_attempts,
    }

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
    project_root = current_dir.parent.parent  # src -> <root>
    config_path = project_root / "configs" / "generate" / "baseline.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Baseline config not found at {config_path}")
        
    return load_config(str(config_path))
