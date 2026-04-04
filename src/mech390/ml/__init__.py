"""
mech390.ml — surrogate model and optimizer for the offset crank-slider mechanism.

Modules
-------
features  : dataset loading, feature/target split, normalization
model     : CrankSliderSurrogate (PyTorch multi-output NN)
train     : training loop + Optuna hyperparameter sweep
optimize  : surrogate-based weighted design optimizer
infer     : inference wrapper (load checkpoint, predict)
"""
