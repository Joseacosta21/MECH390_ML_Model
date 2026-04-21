"""
Neural network surrogate for the crank-slider mechanism.

Shared trunk feeds into two heads: a classifier (pass/fail) and a regression
head for mass, torque, safety factors, and other outputs. Layer widths,
depth, dropout, and batch-norm are all tunable by Optuna.
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# bump this when the checkpoint dict schema changes (new/renamed keys)
CHECKPOINT_VERSION: int = 1


class CrankSliderSurrogate(nn.Module):
    """Multi-output surrogate: one classification head (pass/fail) and one regression head."""

    # builds the shared trunk and the two output heads
    def __init__(
        self,
        input_dim:      int        = 10,
        hidden_sizes:   List[int]  = (256, 128, 64),
        n_reg_targets:  int        = 8,
        dropout_rate:   float      = 0.1,
        use_batch_norm: bool       = True,
    ):
        super().__init__()

        # shared trunk
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

        # output heads
        self.clf_head = nn.Linear(trunk_out, 1)               # pass_fail (sigmoid applied in loss)
        self.reg_head = nn.Linear(trunk_out, n_reg_targets)   # regression targets

    # runs inputs through the trunk and both heads, returns raw logit and raw regression values
    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.clf_head(h), self.reg_head(h)

    # returns pass probability in [0, 1] without needing to unpack both heads
    def predict_pass_prob(self, x: torch.Tensor) -> torch.Tensor:
        logit, _ = self.forward(x)
        return torch.sigmoid(logit)


### Checkpoint helpers

# saves model weights, optimizer state, epoch, val_f1, and hparams to a .pt file
def save_checkpoint(
    model:        CrankSliderSurrogate,
    optimizer_state,
    epoch:        int,
    val_f1:       float,
    hparams:      dict,
    path:         str,
) -> None:
    torch.save({
        'version':              CHECKPOINT_VERSION,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer_state,
        'epoch':                epoch,
        'val_f1':               val_f1,
        'hparams':              hparams,
    }, path)


# loads a checkpoint dict from disk
def load_checkpoint(path: str, device: str = 'cpu') -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def validate_checkpoint_version(ckpt: dict) -> None:
    """Checks the checkpoint schema version; warns if missing, raises if mismatched."""
    ver = ckpt.get('version')
    if ver is None:
        logger.warning(
            "Checkpoint has no 'version' key (pre-versioning checkpoint). "
            "Proceeding - re-train to get a versioned checkpoint."
        )
    elif ver != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {ver} does not match expected "
            f"{CHECKPOINT_VERSION}. Re-train the model."
        )


def build_model_from_hparams(hparams: dict) -> CrankSliderSurrogate:
    """Rebuilds a model from the hparams dict saved inside a checkpoint."""
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
