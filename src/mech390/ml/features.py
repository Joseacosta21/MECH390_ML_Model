"""
Feature engineering for the crank-slider surrogate model.

Loads and merges pass/fail CSVs, derives extra columns, splits into train/val/test,
fits a StandardScaler, and provides helpers to save/load the scaler.

Column contracts:
  RAW_GEO_KEYS (10)    - the 10 independent design variables the optimizer searches over
  INPUT_FEATURES (12)  - RAW_GEO_KEYS + 2 derived slenderness features fed to the NN
  REGRESSION_TARGETS   - continuous outputs predicted by regression heads (8 targets)
  CLASSIFICATION_TARGET- binary pass/fail label
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

### Column contracts

# raw geometry variables - the 10 independent design parameters
# this is the search space the optimizer uses; bounds come from baseline.yaml
RAW_GEO_KEYS: List[str] = [
    'r', 'l', 'e',
    'width_r', 'thickness_r',
    'width_l', 'thickness_l',
    'd_shaft_A', 'pin_diameter_B', 'pin_diameter_C',
]

# model input features - raw geometry + 2 derived slenderness ratios
# slenderness_r = r / thickness_r  (crank slenderness)
# slenderness_l = l / thickness_l  (rod slenderness)
# computed by derive_input_features(); never appear in the raw CSVs
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
    'n_shaft',
    'min_n_fatigue',  # derived: min(n_rod, n_crank, n_pin) - Goodman fatigue FoS
]

ALL_TARGETS: List[str] = [CLASSIFICATION_TARGET] + REGRESSION_TARGETS


### Derived feature helpers

def derive_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds slenderness_r and slenderness_l columns to a copy of the DataFrame."""
    df = df.copy()
    df['slenderness_r'] = df['r'] / df['thickness_r']
    df['slenderness_l'] = df['l'] / df['thickness_l']
    return df


def raw_to_model_input(x: np.ndarray) -> np.ndarray:
    """Converts a 10-dim raw geometry vector to a 12-dim model input by appending slenderness ratios."""
    r_idx   = RAW_GEO_KEYS.index('r')
    tr_idx  = RAW_GEO_KEYS.index('thickness_r')
    l_idx   = RAW_GEO_KEYS.index('l')
    tl_idx  = RAW_GEO_KEYS.index('thickness_l')
    slend_r = float(x[r_idx])  / float(x[tr_idx])
    slend_l = float(x[l_idx])  / float(x[tl_idx])
    return np.append(x.astype(np.float32), [slend_r, slend_l])


### Dataset loading

def load_dataset(csv_pass: str, csv_fail: str) -> pd.DataFrame:
    """Loads and merges the passed and failed CSVs, derives min_n_static and min_n_fatigue, drops incomplete rows."""
    dfs = []
    for path in (csv_pass, csv_fail):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")
        dfs.append(pd.read_csv(p))

    df = pd.concat(dfs, ignore_index=True)

    # derive min_n_static - minimum static FoS across all three links
    fos_cols = ['n_static_rod', 'n_static_crank', 'n_static_pin']
    if all(c in df.columns for c in fos_cols):
        df['min_n_static'] = df[fos_cols].min(axis=1)
    else:
        raise KeyError(
            f"Missing static FOS columns. Expected: {fos_cols}. "
            "Re-run the data generation pipeline to produce these columns."
        )

    # derive min_n_fatigue - minimum Goodman fatigue FoS across all three links
    fatigue_cols = ['n_rod', 'n_crank', 'n_pin']
    if all(c in df.columns for c in fatigue_cols):
        df['min_n_fatigue'] = df[fatigue_cols].min(axis=1)
    else:
        raise KeyError(
            f"Missing fatigue FOS columns. Expected: {fatigue_cols}. "
            "Re-run the data generation pipeline to produce these columns."
        )

    # add slenderness features
    df = derive_input_features(df)

    # drop rows missing any required column
    required = INPUT_FEATURES + ALL_TARGETS
    before = len(df)
    df = df.dropna(subset=required)
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN in required columns.", dropped)

    return df.reset_index(drop=True)


### Train / val / test split

def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    random_seed: int  = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits into train/val/test keeping the same pass/fail ratio in each split."""
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train_frac + val_frac must be < 1.0"

    # first split off test
    train_val, test = train_test_split(
        df,
        test_size=test_frac,
        stratify=df[CLASSIFICATION_TARGET],
        random_state=random_seed,
    )
    # then split train from val
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

    # warn if regression target distributions look very different between train and val
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
                    "(mean=%.4g) and val (mean=%.4g) by %.1f%% - consider "
                    "re-seeding or increasing dataset size.",
                    col, train_mean, val_mean, rel_diff * 100,
                )

    return train, val, test


### Normalisation

# fits a StandardScaler on the training input features
def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[INPUT_FEATURES].values)
    return scaler


def get_arrays(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y_clf, y_reg) numpy arrays for one split, with X already scaled."""
    X     = scaler.transform(df[INPUT_FEATURES].values).astype(np.float32)
    y_clf = df[[CLASSIFICATION_TARGET]].values.astype(np.float32)
    y_reg = df[REGRESSION_TARGETS].values.astype(np.float32)
    return X, y_clf, y_reg


### Target normalisation stats

def compute_target_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Computes min/max of each regression target using only passing rows from the training set.

    Restricted to passing rows to avoid failing designs (extreme stresses, near-zero
    cross-sections) inflating the ranges and making optimizer scoring meaningless.
    For example, tau_A_max on all rows spans [0.17, 70541 N*m]; on passing rows only
    it spans [0.17, ~5 N*m], giving the optimizer a useful gradient.
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


### Target normalisation helpers

def normalize_targets(
    y_reg: np.ndarray,
    target_stats: Dict[str, Dict[str, float]],
    warn: bool = False,
) -> np.ndarray:
    """Scales regression targets to [0, 1] using training min/max.

    Keeps MSE loss balanced across targets with very different physical scales
    (e.g. volume_envelope in m^3 vs utilization in [0, 1]).
    Pass warn=True for val/test sets to flag values outside the training range.
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
    """Converts regression predictions from [0, 1] back to physical units."""
    y_phys = y_norm.copy().astype(np.float32)
    for i, col in enumerate(REGRESSION_TARGETS):
        mn  = float(target_stats[col]['min'])
        mx  = float(target_stats[col]['max'])
        y_phys[:, i] = y_norm[:, i] * (mx - mn) + mn
    return y_phys


### Persistence helpers

# saves the fitted scaler to disk as a pickle file
def save_scaler(scaler: StandardScaler, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(scaler, f)


# loads a previously saved scaler from disk
def load_scaler(path: str) -> StandardScaler:
    with open(path, 'rb') as f:
        return pickle.load(f)
