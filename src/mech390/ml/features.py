"""
Feature engineering for the crank-slider surrogate model.

Responsibilities
----------------
- Load and merge passed/failed CSVs into a single labeled DataFrame
- Derive the `min_n_static` column (min of the three per-link static FOS values)
- Derive slenderness features (slenderness_r, slenderness_l)
- Split into input features X and targets y
- Fit a StandardScaler on the training split; transform all splits
- Provide helpers to persist and reload the scaler alongside the model checkpoint

Column contracts
----------------
RAW_GEO_KEYS (10)    : the 10 independent design variables the optimizer searches over
INPUT_FEATURES (12)  : RAW_GEO_KEYS + 2 derived slenderness features fed to the NN
REGRESSION_TARGETS   : continuous outputs predicted by regression heads (8 targets)
CLASSIFICATION_TARGET: binary pass/fail label
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

# Raw geometry variables — the 10 independent design parameters.
# This is the search space the optimizer uses; bounds come from baseline.yaml.
RAW_GEO_KEYS: List[str] = [
    'r', 'l', 'e',
    'width_r', 'thickness_r',
    'width_l', 'thickness_l',
    'd_shaft_A', 'pin_diameter_B', 'pin_diameter_C',
]

# Model input features — raw geometry + 2 derived slenderness ratios.
# slenderness_r = r / thickness_r  (crank slenderness)
# slenderness_l = l / thickness_l  (rod slenderness)
# Computed by derive_input_features(); never appear in the raw CSVs.
INPUT_FEATURES: List[str] = RAW_GEO_KEYS + ['slenderness_r', 'slenderness_l']

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
    'min_n_fatigue',  # derived: min(n_rod, n_crank, n_pin) — Goodman fatigue FoS
]

ALL_TARGETS: List[str] = [CLASSIFICATION_TARGET] + REGRESSION_TARGETS


# ---------------------------------------------------------------------------
# Derived feature helpers
# ---------------------------------------------------------------------------

def derive_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived input features to a DataFrame in-place (returns copy).

    Computes:
        slenderness_r = r / thickness_r
        slenderness_l = l / thickness_l

    Call this after loading raw CSV data and before calling get_arrays() or
    scaler.transform().  SurrogatePredictor.predict() calls this automatically.
    """
    df = df.copy()
    df['slenderness_r'] = df['r'] / df['thickness_r']
    df['slenderness_l'] = df['l'] / df['thickness_l']
    return df


def raw_to_model_input(x: np.ndarray) -> np.ndarray:
    """
    Convert a 10-dim raw geometry vector (RAW_GEO_KEYS order) to a 12-dim
    model input vector (INPUT_FEATURES order) by appending slenderness ratios.

    Used by the optimizer score function, which searches in the 10-dim raw
    geometry space but must feed 12-dim vectors to the surrogate.

    Args:
        x: 1-D numpy array of length 10 in RAW_GEO_KEYS order.

    Returns:
        1-D numpy array of length 12 (float32) in INPUT_FEATURES order.
    """
    r_idx   = RAW_GEO_KEYS.index('r')
    tr_idx  = RAW_GEO_KEYS.index('thickness_r')
    l_idx   = RAW_GEO_KEYS.index('l')
    tl_idx  = RAW_GEO_KEYS.index('thickness_l')
    slend_r = float(x[r_idx])  / float(x[tr_idx])
    slend_l = float(x[l_idx])  / float(x[tl_idx])
    return np.append(x.astype(np.float32), [slend_r, slend_l])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(csv_pass: str, csv_fail: str) -> pd.DataFrame:
    """
    Load and merge the passed and failed design CSVs.

    Adds the derived column `min_n_static` = min of the three per-link
    static FOS values.  Also adds slenderness_r and slenderness_l derived
    features.  Rows missing any input feature or target are dropped.

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

    # Derive min_n_fatigue — minimum Goodman fatigue FoS across all three links
    fatigue_cols = ['n_rod', 'n_crank', 'n_pin']
    if all(c in df.columns for c in fatigue_cols):
        df['min_n_fatigue'] = df[fatigue_cols].min(axis=1)
    else:
        raise KeyError(
            f"Missing fatigue FOS columns. Expected: {fatigue_cols}. "
            "Re-run the data generation pipeline to produce these columns."
        )

    # Derive slenderness features
    df = derive_input_features(df)

    # Drop rows missing any required column
    required = INPUT_FEATURES + ALL_TARGETS
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN in required columns.", dropped)

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

    Logs a warning if any regression target mean differs between train and val
    by more than 20% (relative), indicating a skewed split.

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
    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    # 8.2 — warn if regression target distributions are severely skewed across splits
    for col in REGRESSION_TARGETS:
        if col not in train.columns:
            continue
        train_mean = float(train[col].mean())
        val_mean   = float(val[col].mean())
        if abs(train_mean) > 1e-8:
            rel_diff = abs(train_mean - val_mean) / abs(train_mean)
            if rel_diff > 0.20:
                logger.warning(
                    "split_dataset: '%s' distribution differs between train "
                    "(mean=%.4g) and val (mean=%.4g) by %.1f%% — consider "
                    "re-seeding or increasing dataset size.",
                    col, train_mean, val_mean, rel_diff * 100,
                )

    return train, val, test


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

    X       : (N, 12)  float32 — normalised input features (incl. slenderness)
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
    Compute min/max of each regression target from PASSING rows of the training set.

    Restricted to pass_fail == 1 to exclude pathological failing designs
    (near-zero cross-sections, extreme stresses) that inflate the ranges and
    make objective normalisation useless in the optimizer score function.
    Example: tau_A_max on all rows spans [0.17, 70,541 N·m]; on passing rows
    only it spans [0.17, ~5 N·m], giving the optimizer meaningful gradient.

    Returns:
        {target_name: {'min': float, 'max': float}}
    """
    pass_df = train_df[train_df[CLASSIFICATION_TARGET] == 1]
    if len(pass_df) == 0:
        raise ValueError("compute_target_stats: no passing rows in training set.")
    logger.info(
        "compute_target_stats: using %d passing rows (of %d total) for normalisation ranges.",
        len(pass_df), len(train_df),
    )
    stats = {}
    for col in REGRESSION_TARGETS:
        stats[col] = {
            'min': float(pass_df[col].min()),
            'max': float(pass_df[col].max()),
        }
    return stats


# ---------------------------------------------------------------------------
# Target normalisation helpers (train on [0,1]; infer in physical units)
# ---------------------------------------------------------------------------

def normalize_targets(
    y_reg: np.ndarray,
    target_stats: Dict[str, Dict[str, float]],
    warn: bool = False,
) -> np.ndarray:
    """
    Normalise regression targets to [0, 1] using training min/max.

    Needed so MSE loss is not dominated by targets with large physical scales
    (e.g. volume_envelope in m³ vs utilization in [0,1]).

    8.1 — When warn=True, logs a warning per column when values fall outside
    the training range [min, max], indicating distribution mismatch.
    Pass warn=True only for val/test sets — on the training set, fail rows
    below the pass-only normalisation range are expected by design and the
    warnings are not actionable.
    """
    y_norm = y_reg.copy().astype(np.float32)
    for i, col in enumerate(REGRESSION_TARGETS):
        mn  = float(target_stats[col]['min'])
        mx  = float(target_stats[col]['max'])
        rng = mx - mn
        if rng > 0:
            if warn:
                n_below = int((y_reg[:, i] < mn).sum())
                n_above = int((y_reg[:, i] > mx).sum())
                if n_below + n_above > 0:
                    logger.warning(
                        "normalize_targets: '%s' has %d row(s) outside training "
                        "range [%.4g, %.4g] (%d below, %d above). "
                        "Train/val distributions may not fully overlap.",
                        col, n_below + n_above, mn, mx, n_below, n_above,
                    )
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
