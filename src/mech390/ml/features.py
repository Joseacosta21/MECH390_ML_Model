"""
Feature engineering for the crank-slider surrogate model.

Responsibilities
----------------
- Load and merge passed/failed CSVs into a single labeled DataFrame
- Derive the `min_n_static` column (min of the three per-link static FOS values)
- Split into input features X and targets y
- Fit a StandardScaler on the training split; transform all splits
- Provide helpers to persist and reload the scaler alongside the model checkpoint

Column contracts
----------------
INPUT_FEATURES (10)  : the 10 independent design variables (d_shaft_A replaces pin_diameter_A)
REGRESSION_TARGETS   : continuous outputs predicted by regression heads (8 targets incl. n_shaft)
CLASSIFICATION_TARGET: binary pass/fail label
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

INPUT_FEATURES: List[str] = [
    'r', 'l', 'e',
    'width_r', 'thickness_r',
    'width_l', 'thickness_l',
    'd_shaft_A', 'pin_diameter_B', 'pin_diameter_C',
]

CLASSIFICATION_TARGET: str = 'pass_fail'

REGRESSION_TARGETS: List[str] = [
    'total_mass',
    'volume_envelope',
    'tau_A_max',
    'E_rev',
    'min_n_static',   # derived: min(n_static_rod, n_static_crank, n_static_pin)
    'utilization',
    'n_buck',
    'n_shaft',        # Mott 12-24 ASME-Elliptic shaft safety factor
]

ALL_TARGETS: List[str] = [CLASSIFICATION_TARGET] + REGRESSION_TARGETS


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(csv_pass: str, csv_fail: str) -> pd.DataFrame:
    """
    Load and merge the passed and failed design CSVs.

    Adds the derived column `min_n_static` = min of the three per-link
    static FOS values.  Rows missing any input feature or target are dropped.

    Args:
        csv_pass: Path to passed_configs.csv
        csv_fail: Path to failed_configs.csv

    Returns:
        Combined DataFrame with all required columns present.
    """
    dfs = []
    for path in (csv_pass, csv_fail):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)

    # Derive min_n_static — minimum static FOS across all three links
    fos_cols = ['n_static_rod', 'n_static_crank', 'n_static_pin']
    if all(c in df.columns for c in fos_cols):
        df['min_n_static'] = df[fos_cols].min(axis=1)
    else:
        raise KeyError(
            f"Missing static FOS columns. Expected: {fos_cols}. "
            "Re-run the data generation pipeline to produce these columns."
        )

    # Drop rows missing any required column
    required = INPUT_FEATURES + ALL_TARGETS
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped > 0:
        import logging
        logging.getLogger(__name__).warning(
            "Dropped %d rows with NaN in required columns.", dropped
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    random_seed: int  = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split on `pass_fail`.

    Returns:
        (train_df, val_df, test_df)
    """
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    # First split off test
    train_val, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df[CLASSIFICATION_TARGET],
        random_state=random_seed,
    )
    # Then split train from val
    relative_val = val_frac / (train_frac + val_frac)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[CLASSIFICATION_TARGET],
        random_state=random_seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the training input features."""
    scaler = StandardScaler()
    scaler.fit(train_df[INPUT_FEATURES].values)
    return scaler


def get_arrays(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X, y_clf, y_reg) numpy arrays for one split.

    X       : (N, 10)  float32 — normalised input features
    y_clf   : (N, 1)   float32 — pass_fail label
    y_reg   : (N, 8)   float32 — regression targets in REGRESSION_TARGETS order
    """
    X     = scaler.transform(df[INPUT_FEATURES].values).astype(np.float32)
    y_clf = df[[CLASSIFICATION_TARGET]].values.astype(np.float32)
    y_reg = df[REGRESSION_TARGETS].values.astype(np.float32)
    return X, y_clf, y_reg


# ---------------------------------------------------------------------------
# Target normalisation stats (used by optimizer score function)
# ---------------------------------------------------------------------------

def compute_target_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute min/max of each regression target from the training set.
    Used by the optimizer to normalise objective values to [0, 1].

    Returns:
        {target_name: {'min': float, 'max': float}}
    """
    stats = {}
    for col in REGRESSION_TARGETS:
        stats[col] = {
            'min': float(train_df[col].min()),
            'max': float(train_df[col].max()),
        }
    return stats


# ---------------------------------------------------------------------------
# Target normalisation helpers (train on [0,1]; infer in physical units)
# ---------------------------------------------------------------------------

def normalize_targets(y_reg: np.ndarray, target_stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Normalise regression targets to [0, 1] using training min/max.

    Needed so MSE loss is not dominated by targets with large physical scales
    (e.g. volume_envelope in m³ vs utilization in [0,1]).
    """
    y_norm = y_reg.copy().astype(np.float32)
    for i, col in enumerate(REGRESSION_TARGETS):
        mn  = float(target_stats[col]['min'])
        mx  = float(target_stats[col]['max'])
        rng = mx - mn
        if rng > 0:
            y_norm[:, i] = (y_reg[:, i] - mn) / rng
        else:
            y_norm[:, i] = 0.0
    return y_norm


def denormalize_targets(y_norm: np.ndarray, target_stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Denormalise regression predictions from [0, 1] back to physical units.

    Apply this to model output before reporting or using in the optimizer
    score function, which expects physical-unit values.
    """
    y_phys = y_norm.copy().astype(np.float32)
    for i, col in enumerate(REGRESSION_TARGETS):
        mn  = float(target_stats[col]['min'])
        mx  = float(target_stats[col]['max'])
        y_phys[:, i] = y_norm[:, i] * (mx - mn) + mn
    return y_phys


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_scaler(scaler: StandardScaler, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> StandardScaler:
    with open(path, 'rb') as f:
        return pickle.load(f)
