"""
Training loop and Optuna hyperparameter sweep for CrankSliderSurrogate.

Usage (via CLI)
---------------
    python scripts/train_model.py --config configs/train/surrogate.yaml

What this module does
---------------------
1. Loads and splits the dataset (features.py)
2. Runs an Optuna study over the search space defined in surrogate.yaml
3. For each trial: builds a model, trains with early stopping, returns val_f1
4. Saves the best checkpoint + scaler + target_stats to data/models/
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mech390.ml import features as F
from mech390.ml.models import (
    CrankSliderSurrogate,
    build_model_from_hparams,
    save_checkpoint,
)

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_loader(X, y_clf, y_reg, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_clf),
        torch.from_numpy(y_reg),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def _train_one_trial(
    hparams:     Dict[str, Any],
    loaders:     Dict[str, DataLoader],
    cfg:         Dict[str, Any],
    device:      torch.device,
) -> float:
    """
    Train a single model with given hparams; return best val_f1.
    """
    model = CrankSliderSurrogate(
        input_dim      = len(F.INPUT_FEATURES),
        hidden_sizes   = hparams['hidden_sizes'],
        n_reg_targets  = len(F.REGRESSION_TARGETS),
        dropout_rate   = hparams['dropout_rate'],
        use_batch_norm = cfg['model'].get('use_batch_norm', True),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = hparams['lr'],
        weight_decay = hparams['weight_decay'],
    )

    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    w_bce    = float(cfg['loss_weights']['bce'])
    w_mse    = float(cfg['loss_weights']['mse'])

    max_epochs = int(cfg['training']['max_epochs'])
    patience   = int(cfg['training']['patience'])

    best_val_f1   = 0.0
    best_state    = None
    no_improve    = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        for X_b, y_clf_b, y_reg_b in loaders['train']:
            X_b, y_clf_b, y_reg_b = X_b.to(device), y_clf_b.to(device), y_reg_b.to(device)
            logit, pred_reg = model(X_b)
            loss = w_bce * bce_loss(logit, y_clf_b) + w_mse * mse_loss(pred_reg, y_reg_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_b, y_clf_b, y_reg_b in loaders['val']:
                X_b, y_clf_b, y_reg_b = X_b.to(device), y_clf_b.to(device), y_reg_b.to(device)
                logit, pred_reg = model(X_b)
                val_loss += (
                    w_bce * bce_loss(logit, y_clf_b) + w_mse * mse_loss(pred_reg, y_reg_b)
                ).item()
                all_probs.append(torch.sigmoid(logit).cpu().numpy())
                all_labels.append(y_clf_b.cpu().numpy())

        val_loss /= len(loaders['val'])
        probs  = np.vstack(all_probs).ravel()
        labels = np.vstack(all_labels).ravel().astype(int)
        preds  = (probs >= 0.5).astype(int)

        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        val_f1    = 2 * precision * recall / (precision + recall + 1e-8)

        # Early stopping on val_f1 (the Optuna objective), not val_loss.
        # Saving on val_loss can discard the best-classified epoch when
        # regression MSE continues to fall after F1 has already peaked.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best weights into model (for checkpoint saving by caller)
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_f1, model


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _make_objective(loaders, cfg, device, best_tracker):
    hidden_options = cfg['model']['hidden_sizes_options']
    dr_lo, dr_hi   = cfg['model']['dropout_range']
    lr_lo, lr_hi   = cfg['training']['lr_range']
    wd_lo, wd_hi   = cfg['training']['weight_decay_range']
    bs_options     = cfg['training']['batch_size_options']

    def objective(trial: optuna.Trial) -> float:
        hparams = {
            'hidden_sizes': hidden_options[
                trial.suggest_int('arch_idx', 0, len(hidden_options) - 1)
            ],
            'dropout_rate':  trial.suggest_float('dropout_rate', dr_lo, dr_hi),
            'lr':            trial.suggest_float('lr', lr_lo, lr_hi, log=True),
            'weight_decay':  trial.suggest_float('weight_decay', wd_lo, wd_hi, log=True),
            'batch_size':    bs_options[
                trial.suggest_int('bs_idx', 0, len(bs_options) - 1)
            ],
        }

        # Rebuild loaders with this trial's batch size
        trial_loaders = {
            'train': DataLoader(
                loaders['train'].dataset,
                batch_size=hparams['batch_size'],
                shuffle=True,
            ),
            'val': loaders['val'],
        }

        val_f1, model = _train_one_trial(hparams, trial_loaders, cfg, device)

        # Track the globally best model
        if val_f1 > best_tracker['val_f1']:
            best_tracker['val_f1']   = val_f1
            best_tracker['model']    = model
            best_tracker['hparams']  = hparams

        return val_f1

    return objective


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_training(cfg: Dict[str, Any]) -> None:
    """
    Full training pipeline: load data → Optuna sweep → save best checkpoint.

    Args:
        cfg: Parsed surrogate.yaml as a dict.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Training on device: %s", device)

    # --- Data ---
    data_cfg = cfg['data']
    df = F.load_dataset(data_cfg['csv_pass'], data_cfg['csv_fail'])
    logger.info("Dataset: %d rows (%d pass, %d fail)",
                len(df), int(df['pass_fail'].sum()), int((df['pass_fail'] == 0).sum()))

    train_df, val_df, test_df = F.split_dataset(
        df,
        train_frac  = data_cfg['train_frac'],
        val_frac    = data_cfg['val_frac'],
        random_seed = data_cfg['random_seed'],
    )
    logger.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    scaler       = F.fit_scaler(train_df)
    target_stats = F.compute_target_stats(train_df)

    X_tr,  y_clf_tr,  y_reg_tr_raw  = F.get_arrays(train_df, scaler)
    X_val, y_clf_val, y_reg_val_raw = F.get_arrays(val_df,   scaler)

    # Normalise regression targets to [0, 1] so MSE loss is not dominated
    # by targets with large physical scales (e.g. volume_envelope in m³).
    y_reg_tr  = F.normalize_targets(y_reg_tr_raw,  target_stats)
    y_reg_val = F.normalize_targets(y_reg_val_raw, target_stats)

    default_bs = int(cfg['training']['batch_size_options'][1])  # middle option as default
    base_loaders = {
        'train': _make_loader(X_tr, y_clf_tr, y_reg_tr, default_bs, shuffle=True),
        'val':   _make_loader(X_val, y_clf_val, y_reg_val, 256, shuffle=False),
    }

    # --- Optuna sweep ---
    best_tracker = {'val_f1': 0.0, 'model': None, 'hparams': None}
    study = optuna.create_study(
        direction  = cfg['optuna']['direction'],
        study_name = 'crank_slider_surrogate',
    )
    n_trials = int(cfg['optuna']['n_trials'])
    logger.info("Starting Optuna sweep: %d trials …", n_trials)
    study.optimize(
        _make_objective(base_loaders, cfg, device, best_tracker),
        n_trials = n_trials,
        show_progress_bar = True,
    )

    best_val_f1 = best_tracker['val_f1']
    best_model  = best_tracker['model']
    best_hparms = best_tracker['hparams']
    logger.info("Best trial val_f1 = %.4f | hparams = %s", best_val_f1, best_hparms)

    # --- Save artefacts ---
    out_cfg = cfg.get('output', {})
    model_dir = Path(out_cfg.get('model_dir', 'data/models'))
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path  = str(model_dir / out_cfg.get('checkpoint', 'surrogate_best.pt').split('/')[-1])
    scaler_path = str(model_dir / out_cfg.get('scaler', 'scaler.pkl').split('/')[-1])
    stats_path  = str(model_dir / out_cfg.get('target_stats', 'target_stats.json').split('/')[-1])

    save_checkpoint(
        model          = best_model,
        optimizer_state= None,
        epoch          = -1,
        val_f1         = best_val_f1,
        hparams        = {**best_hparms,
                          'input_dim':      len(F.INPUT_FEATURES),
                          'n_reg_targets':  len(F.REGRESSION_TARGETS),
                          'use_batch_norm': cfg['model'].get('use_batch_norm', True)},
        path           = ckpt_path,
    )
    F.save_scaler(scaler, scaler_path)
    with open(stats_path, 'w') as fh:
        json.dump(target_stats, fh, indent=2)

    logger.info("Saved checkpoint → %s", ckpt_path)
    logger.info("Saved scaler     → %s", scaler_path)
    logger.info("Saved stats      → %s", stats_path)
    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Best architecture: {best_hparms}")
