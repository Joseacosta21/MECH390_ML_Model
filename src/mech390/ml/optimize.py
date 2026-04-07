"""
Surrogate-based weighted optimizer for the crank-slider mechanism.

Workflow
--------
1. Load the trained surrogate checkpoint + scaler + target_stats.
2. Read geometry bounds from the generate config (baseline.yaml).
3. Read the weight table and constraint from the optimize config (search.yaml).
4. Run scipy.optimize.differential_evolution over the 10-dimensional geometry
   space, evaluating each candidate through the frozen surrogate.
5. Return and print the top-N geometries ranked by weighted score.

Score function
--------------
    score(x) = Σ wᵢ * normalize(objᵢ(x))
               - penalty * max(0, threshold - pass_prob(x))

    normalize maps each objective to [0,1] using training-set min/max:
      - For 'minimize' objectives: score contribution = 1 - normalized_value
        (lower raw value → higher score contribution)
      - For 'maximize' objectives: score contribution = normalized_value
        (higher raw value → higher score contribution)

    The optimizer MINIMISES -score (i.e., maximises score).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import differential_evolution

from mech390.ml import features as F
from mech390.ml.models import build_model_from_hparams, load_checkpoint

logger = logging.getLogger(__name__)

# Fixed order of the 10 geometry variables — must match INPUT_FEATURES
_GEO_KEYS = F.INPUT_FEATURES  # ['r', 'l', 'e', 'width_r', ...]

# Map from objective name → index in REGRESSION_TARGETS
_OBJ_IDX = {name: i for i, name in enumerate(F.REGRESSION_TARGETS)}


# ---------------------------------------------------------------------------
# Bounds extraction from generate config
# ---------------------------------------------------------------------------

def _extract_bounds(gen_cfg: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Read geometry bounds from the generate config and return them in
    _GEO_KEYS order as a list of (min, max) tuples.

    Raises KeyError if any expected key is missing.
    """
    geo = gen_cfg['geometry']
    raw = {
        'r':              geo['r'],
        'l':              geo['l'],
        'e':              geo['e'],
        'width_r':        geo['widths']['width_r'],
        'thickness_r':    geo['thicknesses']['thickness_r'],
        'width_l':        geo['widths']['width_l'],
        'thickness_l':    geo['thicknesses']['thickness_l'],
        'pin_diameter_A': geo['pin_diameters']['pin_diameter_A'],
        'pin_diameter_B': geo['pin_diameters']['pin_diameter_B'],
        'pin_diameter_C': geo['pin_diameters']['pin_diameter_C'],
    }
    return [(float(raw[k]['min']), float(raw[k]['max'])) for k in _GEO_KEYS]


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def _build_score_fn(
    model:           torch.nn.Module,
    scaler,
    target_stats:    Dict[str, Dict[str, float]],
    objectives:      Dict[str, Dict[str, Any]],
    pass_threshold:  float,
    min_net_section: float,
    penalty:         float = 10.0,
    device:          torch.device = torch.device('cpu'),
):
    """
    Returns a callable f(x) → -score  (negative because scipy minimises).

    x : 1-D numpy array of length 10 (raw geometry values, un-normalised)

    Two hard constraints applied as penalties:
    1. pass_prob >= pass_threshold  (surrogate classification gate)
    2. width - D_pin > min_net_section for every pin (prevents degenerate
       near-zero net sections that produce ~TPa stresses in the physics engine)
    """
    # Indices of width/pin pairs in _GEO_KEYS that must satisfy the net-section constraint:
    # (width_idx, pin_idx) matching the four physical constraints
    _NET_PAIRS = [
        (_GEO_KEYS.index('width_r'), _GEO_KEYS.index('pin_diameter_A')),
        (_GEO_KEYS.index('width_r'), _GEO_KEYS.index('pin_diameter_B')),
        (_GEO_KEYS.index('width_l'), _GEO_KEYS.index('pin_diameter_B')),
        (_GEO_KEYS.index('width_l'), _GEO_KEYS.index('pin_diameter_C')),
    ]
    model.eval()

    def _score(x: np.ndarray) -> float:
        x_norm = scaler.transform(x.reshape(1, -1)).astype(np.float32)
        x_t    = torch.from_numpy(x_norm).to(device)

        with torch.no_grad():
            logit, pred_reg = model(x_t)
            pass_prob = float(torch.sigmoid(logit).item())

        # Model outputs are normalised [0,1] — denormalise to physical units
        # before the score function applies its own normalisation step.
        reg_vals = F.denormalize_targets(
            pred_reg.cpu().numpy(), target_stats
        ).ravel()  # shape (7,) in physical units

        score = 0.0
        for obj_name, obj_cfg in objectives.items():
            if obj_name not in _OBJ_IDX:
                logger.warning("Unknown objective '%s' — skipping.", obj_name)
                continue
            idx    = _OBJ_IDX[obj_name]
            raw    = float(reg_vals[idx])
            stats  = target_stats[obj_name]
            rng    = stats['max'] - stats['min']
            norm   = (raw - stats['min']) / rng if rng > 0 else 0.5
            norm   = float(np.clip(norm, 0.0, 1.0))
            weight = float(obj_cfg['weight'])

            if obj_cfg['direction'] == 'minimize':
                score += weight * (1.0 - norm)   # lower raw → higher contribution
            else:
                score += weight * norm            # higher raw → higher contribution

        # Hard constraint 1: pass probability gate
        if pass_prob < pass_threshold:
            score -= penalty * (pass_threshold - pass_prob)

        # Hard constraint 2: net-section feasibility
        # Penalise any geometry where width - D_pin <= min_net_section
        for w_idx, p_idx in _NET_PAIRS:
            violation = min_net_section - (x[w_idx] - x[p_idx])
            if violation > 0:
                score -= penalty * violation * 1e3   # scale to metres → significant penalty

        return -score   # scipy minimises, we want to maximise score

    return _score


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_optimization(
    gen_cfg:      Dict[str, Any],
    opt_cfg:      Dict[str, Any],
    checkpoint:   str,
    scaler_path:  str,
    stats_path:   str,
) -> List[Dict[str, Any]]:
    """
    Run the surrogate optimizer and return top-N results.

    Args:
        gen_cfg:     Parsed baseline.yaml (source of geometry bounds).
        opt_cfg:     Parsed search.yaml (weight table + optimizer settings).
        checkpoint:  Path to surrogate_best.pt
        scaler_path: Path to scaler.pkl
        stats_path:  Path to target_stats.json

    Returns:
        List of dicts, one per top-N result, each containing:
            - the 10 geometry values
            - predicted pass_prob
            - predicted regression targets
            - weighted_score
    """
    device = torch.device('cpu')

    # --- Load model ---
    ckpt   = load_checkpoint(checkpoint, device=str(device))
    model  = build_model_from_hparams(ckpt['hparams'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    logger.info("Loaded surrogate (val_f1=%.4f) from %s", ckpt.get('val_f1', -1), checkpoint)

    scaler       = F.load_scaler(scaler_path)
    with open(stats_path) as fh:
        target_stats = json.load(fh)

    # --- Bounds and objectives ---
    bounds         = _extract_bounds(gen_cfg)
    objectives     = opt_cfg['objectives']
    pass_threshold = float(opt_cfg['constraints']['pass_fail_prob_min'])
    n_top          = int(opt_cfg['optimizer']['n_top_results'])
    seed           = int(opt_cfg['optimizer']['seed'])
    maxiter        = int(opt_cfg['optimizer']['maxiter'])
    popsize        = int(opt_cfg['optimizer']['popsize'])

    # Net-section constraint: width - D_pin > delta + 2 * min_wall
    stress_cfg      = gen_cfg.get('stress_analysis', {})
    delta_m         = float(stress_cfg.get('delta',       1e-4))
    min_wall_m      = float(stress_cfg.get('min_wall_mm', 0.5e-3))
    min_net_section = delta_m + 2.0 * min_wall_m

    score_fn = _build_score_fn(
        model, scaler, target_stats, objectives, pass_threshold,
        min_net_section=min_net_section, device=device,
    )

    logger.info("Running differential_evolution over %d-D space …", len(bounds))
    result = differential_evolution(
        score_fn,
        bounds,
        seed       = seed,
        maxiter    = maxiter,
        popsize    = popsize,
        tol        = 1e-6,
        mutation   = (0.5, 1.0),
        recombination = 0.7,
        workers    = 1,
    )
    logger.info("Optimizer converged: %s  (fun=%.6f)", result.success, result.fun)

    # --- Collect top-N by evaluating a grid of candidates from the final population ---
    # differential_evolution returns only the best; we re-evaluate the full population
    # by running a dense random sample near the best and collecting distinct results.
    rng = np.random.default_rng(seed + 1)
    candidates = [result.x]  # start with the global best

    # Sample extra candidates around the best to fill top-N
    scale = np.array([(hi - lo) * 0.05 for lo, hi in bounds])
    for _ in range(max(0, n_top * 20 - 1)):
        noise = rng.normal(0, 1, len(bounds)) * scale
        cand  = np.clip(result.x + noise,
                        [lo for lo, _ in bounds],
                        [hi for _, hi in bounds])
        candidates.append(cand)

    # Score all candidates, sort, deduplicate
    scored = []
    for x in candidates:
        x_norm = scaler.transform(x.reshape(1, -1)).astype(np.float32)
        x_t    = torch.from_numpy(x_norm)
        with torch.no_grad():
            logit, pred_reg = model(x_t)
            pass_prob = float(torch.sigmoid(logit).item())
        # Denormalise to physical units for result display
        reg_vals = F.denormalize_targets(
            pred_reg.cpu().numpy(), target_stats
        ).ravel()
        s = -score_fn(x)   # convert back to positive score

        row = {k: float(v) for k, v in zip(_GEO_KEYS, x)}
        row['pass_prob']    = pass_prob
        row['weighted_score'] = s
        for i, name in enumerate(F.REGRESSION_TARGETS):
            row[f'pred_{name}'] = float(reg_vals[i])
        scored.append(row)

    scored.sort(key=lambda r: r['weighted_score'], reverse=True)

    # Remove near-duplicates (within 1% of range on all dims)
    tol_abs = np.array([(hi - lo) * 0.01 for lo, hi in bounds])
    unique  = []
    for row in scored:
        x_row = np.array([row[k] for k in _GEO_KEYS])
        if all(
            np.any(np.abs(x_row - np.array([u[k] for k in _GEO_KEYS])) > tol_abs)
            for u in unique
        ):
            unique.append(row)
        if len(unique) >= n_top:
            break

    return unique
