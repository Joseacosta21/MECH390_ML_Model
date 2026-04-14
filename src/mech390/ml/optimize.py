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
               - Σ wᵢ * ood_penalty(objᵢ(x))
               - penalty * max(0, threshold - pass_prob(x))
               - hard_constraint_penalties

    normalize maps each objective to [0,1] using training-set min/max:
      - For 'minimize' objectives: score contribution = 1 - normalized_value
        (lower raw value → higher score contribution)
      - For 'maximize' objectives: score contribution = normalized_value
        (higher raw value → higher score contribution)

    OOD penalty: if a predicted value falls outside the training range by more
    than ood_tolerance (fraction), the excess is penalised proportionally to
    the objective's weight and ood_penalty_scale. This prevents the optimizer
    from exploiting wild surrogate extrapolations as free high scores.

      ood_amount = max(0, -norm - tol) + max(0, norm - 1 - tol)
      penalty_contribution = weight * ood_amount * ood_penalty_scale

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
from mech390.ml.models import (
    build_model_from_hparams,
    load_checkpoint,
    validate_checkpoint_version,
)
from mech390.physics import buckling as _buckling
from mech390.physics import kinematics as _kinematics

logger = logging.getLogger(__name__)

# Raw geometry variables the optimizer searches over (10-dim).
# These define the differential_evolution bounds and the result dict keys.
# Distinct from F.INPUT_FEATURES (12-dim) which also includes derived
# slenderness features — see F.raw_to_model_input() for the conversion.
_GEO_KEYS = F.RAW_GEO_KEYS  # ['r', 'l', 'e', 'width_r', ...]

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
        'd_shaft_A':      geo['pin_diameters']['d_shaft_A'],
        'pin_diameter_B': geo['pin_diameters']['pin_diameter_B'],
        'pin_diameter_C': geo['pin_diameters']['pin_diameter_C'],
    }
    return [(float(raw[k]['min']), float(raw[k]['max'])) for k in _GEO_KEYS]


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def _build_score_fn(
    model:                    torch.nn.Module,
    scaler,
    target_stats:             Dict[str, Dict[str, float]],
    objectives:               Dict[str, Dict[str, Any]],
    pass_threshold:           float,
    min_net_section:          float,
    n_buck_min:               float = 3.0,
    E_material:               float = 73.1e9,
    omega:                    float = 3.14159,
    m_block:                  float = 0.5,
    mu:                       float = 0.47,
    g:                        float = 9.81,
    penalty:                  float = 10.0,
    penalty_net_section_scale: float = 1000.0,
    penalty_kinematic_scale:   float = 100.0,
    penalty_buckling_scale:    float = 1.0,
    ood_tolerance:            float = 0.1,
    ood_penalty_scale:        float = 10.0,
    rom_target:               float = 0.25,
    rom_tolerance:            float = 0.0005,
    qrr_min:                  float = 1.5,
    qrr_max:                  float = 2.5,
    penalty_rom_scale:        float = 200.0,
    penalty_qrr_scale:        float = 20.0,
    device:                   torch.device = torch.device('cpu'),
):
    """
    Returns a callable f(x) → -score  (negative because scipy minimises).

    x : 1-D numpy array of length 10 (raw geometry values, un-normalised)

    Hard constraints applied as penalties:
    1. pass_prob >= pass_threshold  (surrogate classification gate)
    2. width - D_pin > min_net_section for every pin (prevents degenerate
       near-zero net sections that produce ~TPa stresses in the physics engine)
    3. l > r + e  (kinematic feasibility: rod must bridge crank to slider at all angles)
    4. Euler buckling FoS >= n_buck_min  (analytical estimate from rod geometry)
    5. |ROM - rom_target| <= rom_tolerance  (stroke must match 250 mm ± 0.5 mm;
       all training data satisfies this — without the penalty the optimizer freely
       picks r values that produce wildly wrong strokes)
    6. qrr_min <= QRR <= qrr_max  (quick-return ratio in [1.5, 2.5];
       same reasoning as ROM — training data is entirely within this band)
       Uses P_cr = π²EI/l² and a conservative F_rod estimate from slider dynamics.

    OOD penalty:
    Each regression prediction is normalised to [0, 1] using training-set
    min/max. If the raw normalised value falls outside [-ood_tolerance,
    1 + ood_tolerance], the excess is subtracted from the score, scaled by
    the objective weight and ood_penalty_scale. This prevents the optimizer
    from rewarding wildly extrapolated predictions with free high scores.

    Args:
        ood_tolerance:     Fractional grace band beyond [0, 1] before penalty
                           kicks in. Default 0.1 (10% of training range).
        ood_penalty_scale: Multiplier on the OOD excess, per unit weight.
                           Default 10.0 means a full-weight objective that
                           extrapolates 1.0 range-unit beyond the band loses
                           the equivalent of its entire weight from the score.
    """
    # Indices of width/pin pairs in _GEO_KEYS that must satisfy the net-section constraint:
    # (width_idx, pin_idx) matching the four physical constraints
    _NET_PAIRS = [
        (_GEO_KEYS.index('width_r'), _GEO_KEYS.index('d_shaft_A')),
        (_GEO_KEYS.index('width_r'), _GEO_KEYS.index('pin_diameter_B')),
        (_GEO_KEYS.index('width_l'), _GEO_KEYS.index('pin_diameter_B')),
        (_GEO_KEYS.index('width_l'), _GEO_KEYS.index('pin_diameter_C')),
    ]
    _R_IDX   = _GEO_KEYS.index('r')
    _L_IDX   = _GEO_KEYS.index('l')
    _E_IDX   = _GEO_KEYS.index('e')
    _W_L_IDX = _GEO_KEYS.index('width_l')
    _T_L_IDX = _GEO_KEYS.index('thickness_l')
    model.eval()

    def _score(x: np.ndarray) -> float:
        # x is 10-dim (RAW_GEO_KEYS); extend to 12-dim INPUT_FEATURES before scaling
        x_full = F.raw_to_model_input(x)
        x_norm = scaler.transform(x_full.reshape(1, -1)).astype(np.float32)
        x_t    = torch.from_numpy(x_norm).to(device)

        with torch.no_grad():
            logit, pred_reg = model(x_t)
            pass_prob = float(torch.sigmoid(logit).item())

        # Model outputs are normalised [0,1] — denormalise to physical units
        # before the score function applies its own normalisation step.
        reg_vals = F.denormalize_targets(
            pred_reg.cpu().numpy(), target_stats
        ).ravel()  # shape (len(REGRESSION_TARGETS),) in physical units

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
            weight = float(obj_cfg['weight'])

            # OOD penalty: subtract from score proportionally to how far the
            # prediction falls outside the training range (plus grace band).
            # Applied before clipping so the penalty captures the full excess.
            ood_low    = max(0.0, -norm - ood_tolerance)
            ood_high   = max(0.0,  norm - 1.0 - ood_tolerance)
            ood_amount = ood_low + ood_high
            if ood_amount > 0.0:
                score -= weight * ood_amount * ood_penalty_scale

            norm = float(np.clip(norm, 0.0, 1.0))

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
                score -= penalty * violation * penalty_net_section_scale

        # Hard constraint 3: kinematic feasibility  l > r + e
        # The rod must reach pin C from pin B at all crank angles.
        # Violation when l - (r + e) <= 0.
        kin_violation = (x[_R_IDX] + x[_E_IDX]) - x[_L_IDX]
        if kin_violation > 0:
            score -= penalty * kin_violation * penalty_kinematic_scale

        # Hard constraint 4: Euler buckling  n_buck >= n_buck_min
        # P_cr from buckling.critical_load() (shared formula with physics engine).
        # Conservative rod force estimate from slider dynamics:
        #   F_rod ≈ m_block * (r * ω² + μ * g)  (inertia + friction, upper bound)
        w_l  = x[_W_L_IDX]
        t_l  = x[_T_L_IDX]
        l_v  = x[_L_IDX]
        P_cr      = _buckling.critical_load(w_l, t_l, l_v, E_material)
        F_rod_est = max(m_block * (x[_R_IDX] * omega**2 + mu * g), 1e-6)
        n_buck_est = P_cr / F_rod_est
        buck_violation = n_buck_min - n_buck_est
        if buck_violation > 0:
            score -= penalty * buck_violation * penalty_buckling_scale

        # Hard constraint 5 & 6: ROM and QRR
        # Every training sample satisfies ROM = 250 mm ± 0.5 mm and QRR ∈ [1.5, 2.5].
        # Without these penalties the optimizer freely picks r values that produce
        # wildly wrong strokes — the surrogate pass_prob is blind to kinematics.
        # kinematics.calculate_metrics() is the same formula used by Stage 1 and
        # the physics engine, so no inconsistency is introduced.
        kin_metrics = _kinematics.calculate_metrics(x[_R_IDX], x[_L_IDX], x[_E_IDX])
        if not kin_metrics.get('valid', False):
            # Geometry invalid for kinematics (e.g. sqrt of negative): heavy penalty
            score -= penalty * 1.0 * (penalty_rom_scale + penalty_qrr_scale)
        else:
            rom_actual = kin_metrics['ROM']
            rom_err    = abs(rom_actual - rom_target) - rom_tolerance
            if rom_err > 0:
                score -= penalty * rom_err * penalty_rom_scale

            qrr_actual = kin_metrics['QRR']
            qrr_low    = max(0.0, qrr_min - qrr_actual)
            qrr_high   = max(0.0, qrr_actual - qrr_max)
            qrr_viol   = qrr_low + qrr_high
            if qrr_viol > 0:
                score -= penalty * qrr_viol * penalty_qrr_scale

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
    validate_checkpoint_version(ckpt)
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
    delta_m         = float(stress_cfg.get('diametral_clearance_m', 1e-4))
    min_wall_m      = float(stress_cfg.get('min_wall_m', 0.5e-3))
    min_net_section = delta_m + 2.0 * min_wall_m

    # Penalty scaling factors (from search.yaml constraints section)
    constraint_cfg             = opt_cfg.get('constraints', {})
    penalty_net_section_scale  = float(constraint_cfg.get('penalty_net_section_scale', 1000.0))
    penalty_kinematic_scale    = float(constraint_cfg.get('penalty_kinematic_scale',   100.0))
    penalty_buckling_scale     = float(constraint_cfg.get('penalty_buckling_scale',    1.0))
    ood_tolerance              = float(constraint_cfg.get('ood_tolerance',             0.1))
    ood_penalty_scale          = float(constraint_cfg.get('ood_penalty_scale',         10.0))
    penalty_rom_scale          = float(constraint_cfg.get('penalty_rom_scale',         200.0))
    penalty_qrr_scale          = float(constraint_cfg.get('penalty_qrr_scale',         20.0))

    # ROM and QRR targets (from operating config — same values Stage 1 enforces)
    operating_cfg = gen_cfg.get('operating', {})
    rom_target    = float(operating_cfg.get('ROM', 0.25))
    rom_tolerance = float(operating_cfg.get('ROM_tolerance', 0.0005))
    qrr_cfg       = operating_cfg.get('QRR', {})
    qrr_min_v     = float(qrr_cfg.get('min', 1.5)) if isinstance(qrr_cfg, dict) else 1.5
    qrr_max_v     = float(qrr_cfg.get('max', 2.5)) if isinstance(qrr_cfg, dict) else 2.5

    # Analytical buckling parameters (from material + operating config)
    material_cfg    = gen_cfg.get('material', {})
    limits_cfg      = gen_cfg.get('limits', {})
    E_mat   = float(material_cfg.get('E',   73.1e9))
    rpm_val = float(operating_cfg.get('RPM', 30))
    omega_v = rpm_val * 2.0 * 3.14159265 / 60.0
    m_blk   = float(operating_cfg.get('m_block', 0.5))
    mu_v    = float(operating_cfg.get('mu',  0.47))
    g_v     = float(operating_cfg.get('g',   9.81))
    nbuck_min = float(limits_cfg.get('n_buck_min',
                      stress_cfg.get('n_buck_target', 3.0)))

    score_fn = _build_score_fn(
        model, scaler, target_stats, objectives, pass_threshold,
        min_net_section           = min_net_section,
        n_buck_min                = nbuck_min,
        E_material                = E_mat,
        omega                     = omega_v,
        m_block                   = m_blk,
        mu                        = mu_v,
        g                         = g_v,
        penalty_net_section_scale = penalty_net_section_scale,
        penalty_kinematic_scale   = penalty_kinematic_scale,
        penalty_buckling_scale    = penalty_buckling_scale,
        ood_tolerance             = ood_tolerance,
        ood_penalty_scale         = ood_penalty_scale,
        rom_target                = rom_target,
        rom_tolerance             = rom_tolerance,
        qrr_min                   = qrr_min_v,
        qrr_max                   = qrr_max_v,
        penalty_rom_scale         = penalty_rom_scale,
        penalty_qrr_scale         = penalty_qrr_scale,
        device                    = device,
    )

    logger.info("Running differential_evolution over %d-D space …", len(bounds))
    
    global_convergence_log = []
    
    def convergence_callback(xk, convergence=None):
        global_convergence_log.append(float(-score_fn(xk))) # Record the un-negated score (maximize)

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
        callback   = convergence_callback,
    )
    logger.info("Optimizer converged: %s  (fun=%.6f)", result.success, result.fun)

    # Save convergence log
    out_dir = Path('data/results')
    out_dir.mkdir(parents=True, exist_ok=True)
    conv_path = out_dir / 'convergence_log.json'
    logger.info("Saving convergence log to %s", conv_path)
    with open(conv_path, 'w') as f:
        json.dump(global_convergence_log, f)

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
        x_full = F.raw_to_model_input(x)
        x_norm = scaler.transform(x_full.reshape(1, -1)).astype(np.float32)
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

    # Remove near-duplicates (within 1% of range on all dims).
    # Only keep candidates that passed the surrogate gate (pass_prob >= threshold)
    # and have a positive score — prevents degenerate candidates from filling slots.
    tol_abs = np.array([(hi - lo) * 0.01 for lo, hi in bounds])
    unique  = []
    for row in scored:
        if row['pass_prob'] < pass_threshold or row['weighted_score'] <= 0:
            continue
        x_row = np.array([row[k] for k in _GEO_KEYS])
        if all(
            np.any(np.abs(x_row - np.array([u[k] for k in _GEO_KEYS])) > tol_abs)
            for u in unique
        ):
            unique.append(row)
        if len(unique) >= n_top:
            break

    if not unique:
        # Fallback: optimizer found no gate-passing candidates — return best by score
        logger.warning(
            "No candidates passed pass_prob >= %.2f gate. "
            "Returning top-%d by score regardless. Consider retraining.",
            pass_threshold, n_top,
        )
        unique = scored[:n_top]

    return unique


# ---------------------------------------------------------------------------
# Dataset best finder
# ---------------------------------------------------------------------------

def find_dataset_best(
    csv_pass_path: str,
    objectives:   Dict[str, Dict[str, Any]],
    target_stats: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Find the highest-scoring design in passed_configs.csv using the same
    objective weights / normalisation as the surrogate optimizer (but using
    actual physics values from the CSV, not surrogate predictions).

    Returns a result dict in the same format as run_optimization():
      - RAW_GEO_KEYS values
      - pass_prob = 1.0  (it came from passed_configs)
      - weighted_score   (objective-weighted score on actual physics values)
      - pred_<target>    (actual CSV value, labelled as "CSV" in the report)
      - source = 'dataset'

    Returns None if the CSV is not found or empty.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas required for find_dataset_best()")
        return None

    p = Path(csv_pass_path)
    if not p.exists():
        logger.warning("Dataset best: passed CSV not found at %s", p)
        return None

    df = pd.read_csv(p)
    if df.empty:
        return None

    # Derive composite targets if not already in CSV
    if 'min_n_fatigue' not in df.columns:
        fat_cols = [c for c in ('n_rod', 'n_crank', 'n_pin') if c in df.columns]
        if fat_cols:
            df['min_n_fatigue'] = df[fat_cols].min(axis=1)

    if 'min_n_static' not in df.columns:
        sta_cols = [c for c in ('n_static_rod', 'n_static_crank', 'n_static_pin') if c in df.columns]
        if sta_cols:
            df['min_n_static'] = df[sta_cols].min(axis=1)

    def _score_row(row) -> float:
        s = 0.0
        for name, cfg in objectives.items():
            if name not in row.index or name not in target_stats:
                continue
            raw   = float(row[name])
            stats = target_stats[name]
            rng   = stats['max'] - stats['min']
            norm  = float(np.clip((raw - stats['min']) / rng if rng > 0 else 0.5, 0.0, 1.0))
            w     = float(cfg['weight'])
            s    += w * (1.0 - norm) if cfg['direction'] == 'minimize' else w * norm
        return s

    scores   = df.apply(_score_row, axis=1)
    best_idx = int(scores.idxmax())
    best_row = df.loc[best_idx]

    result = {k: float(best_row[k]) for k in _GEO_KEYS if k in best_row.index}
    result['pass_prob']      = 1.0
    result['weighted_score'] = float(scores[best_idx])
    result['source']         = 'dataset'

    for target in F.REGRESSION_TARGETS:
        result[f'pred_{target}'] = float(best_row[target]) if target in best_row.index else None

    logger.info("Dataset best: score=%.4f  r=%.1f mm  l=%.1f mm  e=%.1f mm",
                result['weighted_score'],
                result.get('r', 0) * 1e3,
                result.get('l', 0) * 1e3,
                result.get('e', 0) * 1e3)
    return result
