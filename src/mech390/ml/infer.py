"""
Inference wrapper for the trained CrankSliderSurrogate.

Loads a checkpoint + scaler and exposes a simple predict() function that
accepts either a single design dict or a DataFrame and returns predictions
for all targets.

Usage
-----
    from mech390.ml.infer import SurrogatePredictor

    predictor = SurrogatePredictor(
        checkpoint  = 'data/models/surrogate_best.pt',
        scaler_path = 'data/models/scaler.pkl',
    )

    result = predictor.predict({
        'r': 0.08, 'l': 0.22, 'e': 0.05,
        'width_r': 0.012, 'thickness_r': 0.008,
        'width_l': 0.010, 'thickness_l': 0.007,
        'd_shaft_A': 0.003, 'pin_diameter_B': 0.003, 'pin_diameter_C': 0.003,
    })
    print(result['pass_prob'], result['total_mass'])
"""

import json
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch

from mech390.ml import features as F
from mech390.ml.models import (
    build_model_from_hparams,
    load_checkpoint,
    validate_checkpoint_version,
)


class SurrogatePredictor:
    """
    Thin wrapper around a loaded CrankSliderSurrogate checkpoint.

    Args:
        checkpoint:  Path to surrogate_best.pt
        scaler_path: Path to scaler.pkl
        device:      'cpu' or 'cuda' (default 'cpu')
    """

    def __init__(
        self,
        checkpoint:  str,
        scaler_path: str,
        stats_path:  str = 'data/models/target_stats.json',
        device:      str = 'cpu',
    ):
        self.device = torch.device(device)
        ckpt        = load_checkpoint(checkpoint, device=device)
        validate_checkpoint_version(ckpt)

        ckpt_n_reg = ckpt['hparams'].get('n_reg_targets')
        if ckpt_n_reg != len(F.REGRESSION_TARGETS):
            raise ValueError(
                f"Checkpoint has n_reg_targets={ckpt_n_reg} but "
                f"features.py defines {len(F.REGRESSION_TARGETS)} REGRESSION_TARGETS. "
                "Re-train the model or restore a matching checkpoint."
            )

        self.model  = build_model_from_hparams(ckpt['hparams'])
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.scaler = F.load_scaler(scaler_path)
        with open(stats_path) as fh:
            self.target_stats = json.load(fh)

    def predict(
        self,
        design: Union[Dict[str, float], pd.DataFrame, List[Dict[str, float]]],
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Predict pass probability and all regression targets.

        Args:
            design: A single design dict, a list of dicts, or a DataFrame.
                    Must contain all 10 INPUT_FEATURES keys.

        Returns:
            - If input is a single dict: returns a dict with keys
              'pass_prob', 'pass_fail_pred', and all REGRESSION_TARGETS.
            - If input is a list or DataFrame: returns a DataFrame with
              one row per design.
        """
        single = isinstance(design, dict)
        if single:
            design = [design]

        if isinstance(design, list):
            df = pd.DataFrame(design)
        else:
            df = design.copy()

        # Derive slenderness features before checking for INPUT_FEATURES presence
        df = F.derive_input_features(df)

        missing = [c for c in F.INPUT_FEATURES if c not in df.columns]
        if missing:
            raise KeyError(f"Input missing columns: {missing}")

        X = self.scaler.transform(df[F.INPUT_FEATURES].values).astype(np.float32)
        x_t = torch.from_numpy(X).to(self.device)

        with torch.no_grad():
            logit, pred_reg = self.model(x_t)
            pass_probs = torch.sigmoid(logit).cpu().numpy().ravel()

        # Model outputs are normalised [0,1] — denormalise to physical units
        reg_vals = F.denormalize_targets(
            pred_reg.cpu().numpy(), self.target_stats
        )  # (N, len(REGRESSION_TARGETS)) in physical units

        results = []
        for i in range(len(df)):
            row: Dict[str, Any] = {'pass_prob': float(pass_probs[i]),
                                   'pass_fail_pred': int(pass_probs[i] >= 0.5)}
            for j, name in enumerate(F.REGRESSION_TARGETS):
                row[name] = float(reg_vals[i, j])
            results.append(row)

        if single:
            return results[0]

        return pd.DataFrame(results)
