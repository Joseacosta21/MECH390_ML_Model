"""
CrankSliderSurrogate — PyTorch multi-output neural network.

Architecture
------------
    Input (10 normalised design vars)
        │
    [Linear → BatchNorm → ReLU → Dropout] × N_layers   ← shared trunk
        │                   │
    [Linear → Sigmoid]   [Linear (×7)]
        │                   │
    pass_fail (1)       regression targets (7):
                            total_mass, volume_envelope, tau_A_max,
                            E_rev, min_n_static, utilization, n_buck

The hidden layer widths, depth, dropout rate, and batch-norm flag are all
configurable so that Optuna can search over them.
"""

from typing import List

import torch
import torch.nn as nn


class CrankSliderSurrogate(nn.Module):
    """
    Multi-output surrogate for the crank-slider mechanism.

    Args:
        input_dim:       Number of input features (default 10).
        hidden_sizes:    List of hidden layer widths, e.g. [256, 128, 64].
        n_reg_targets:   Number of regression output heads (default 7).
        dropout_rate:    Dropout probability applied after each hidden layer.
        use_batch_norm:  Whether to insert BatchNorm1d after each linear layer.
    """

    def __init__(
        self,
        input_dim:      int        = 10,
        hidden_sizes:   List[int]  = (256, 128, 64),
        n_reg_targets:  int        = 7,
        dropout_rate:   float      = 0.1,
        use_batch_norm: bool       = True,
    ):
        super().__init__()

        # --- Shared trunk ---
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h

        self.trunk = nn.Sequential(*layers)
        trunk_out  = hidden_sizes[-1]

        # --- Output heads ---
        self.clf_head = nn.Linear(trunk_out, 1)          # → pass_fail (sigmoid applied in loss)
        self.reg_head = nn.Linear(trunk_out, n_reg_targets)  # → regression targets

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, input_dim) float32 — normalised input features

        Returns:
            logit_clf : (batch, 1)            — raw logit for pass_fail (apply sigmoid for prob)
            pred_reg  : (batch, n_reg_targets) — raw regression predictions (un-normalised)
        """
        h = self.trunk(x)
        return self.clf_head(h), self.reg_head(h)

    def predict_pass_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns pass probability in [0, 1]."""
        logit, _ = self.forward(x)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:        CrankSliderSurrogate,
    optimizer_state,
    epoch:        int,
    val_f1:       float,
    hparams:      dict,
    path:         str,
) -> None:
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'epoch':                epoch,
        'val_f1':               val_f1,
        'hparams':              hparams,
    }, path)


def load_checkpoint(path: str, device: str = 'cpu') -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def build_model_from_hparams(hparams: dict) -> CrankSliderSurrogate:
    """
    Reconstruct a model from the hparams dict saved in a checkpoint.

    Raises KeyError if any required key is missing, so that a stale or
    incompatible checkpoint is caught at load time rather than silently
    producing a model with wrong architecture.
    """
    _REQUIRED = ('hidden_sizes', 'dropout_rate', 'input_dim', 'n_reg_targets', 'use_batch_norm')
    missing = [k for k in _REQUIRED if k not in hparams]
    if missing:
        raise KeyError(
            f"build_model_from_hparams: checkpoint hparams missing required keys: {missing}. "
            f"Present keys: {list(hparams.keys())}"
        )
    return CrankSliderSurrogate(
        input_dim      = hparams['input_dim'],
        hidden_sizes   = hparams['hidden_sizes'],
        n_reg_targets  = hparams['n_reg_targets'],
        dropout_rate   = hparams['dropout_rate'],
        use_batch_norm = hparams['use_batch_norm'],
    )
