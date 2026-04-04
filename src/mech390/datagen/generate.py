"""
Data Generation Orchestrator.

Runs the full pipeline (Stage 1 → Stage 2 → mass properties → physics engine)
and returns a DatasetResult containing seven DataFrames ready for CSV export:

  kinematics_df  — one row per (design, crank angle): positions, velocities, accelerations
  dynamics_df    — one row per (design, crank angle): joint forces, torque
  stresses_df    — one row per (design, crank angle): per-component normal + shear stresses
  fatigue_df     — one row per design: Goodman / Miner fatigue metrics per component
  buckling_df    — one row per design: Euler buckling metrics for connecting rod
  passed_df      — one row per passing design: geometry + summary metrics + check columns
  failed_df      — one row per failing design: geometry + summary metrics + check columns

Designs that fail physics evaluation (valid_physics=False) are silently dropped.
Pass/fail is determined by ALL of the following checks passing:
  - Static stress:   utilization = max(sigma_max/sigma_allow, tau_max/tau_allow) <= 1.0
  - Buckling:        n_buck >= n_buck_target  (default 3.0)
  - Fatigue Goodman: n_rod, n_crank, n_pin   >= 1.0
  - Fatigue Miner:   D_rod, D_crank, D_pin   <  1.0  (when total_cycles is set)
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mech390.datagen import stage1_kinematic, stage2_embodiment
from mech390.physics._utils import get_or_warn

try:
    from mech390.physics import engine
    from mech390.physics import mass_properties as mp
except ImportError:
    engine = None
    mp = None
    logging.warning("mech390.physics not found. Physics evaluation will be skipped.")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geometry columns included in every CSV row (self-contained)
# ---------------------------------------------------------------------------
_GEOM_COLS = [
    'r', 'l', 'e',
    'ROM', 'QRR', 'omega',
    'width_r', 'thickness_r',
    'width_l', 'thickness_l',
    'pin_diameter_A', 'pin_diameter_B', 'pin_diameter_C',
    'mass_crank', 'mass_rod', 'mass_slider',
]


class DatasetResult:
    """Container for all pipeline outputs."""

    def __init__(
        self,
        kinematics_df: pd.DataFrame,
        dynamics_df:   pd.DataFrame,
        stresses_df:   pd.DataFrame,
        fatigue_df:    pd.DataFrame,
        buckling_df:   pd.DataFrame,
        passed_df:     pd.DataFrame,
        failed_df:     pd.DataFrame,
        summary:       Dict[str, Any],
    ):
        self.kinematics_df = kinematics_df
        self.dynamics_df   = dynamics_df
        self.stresses_df   = stresses_df
        self.fatigue_df    = fatigue_df
        self.buckling_df   = buckling_df
        self.passed_df     = passed_df
        self.failed_df     = failed_df
        self.summary       = summary

    # Convenience aliases kept for any legacy callers
    @property
    def all_cases(self) -> pd.DataFrame:
        parts = [df for df in (self.passed_df, self.failed_df) if not df.empty]
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts).sort_values('design_id').reset_index(drop=True)

    @property
    def pass_cases(self) -> pd.DataFrame:
        return self.passed_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_geom(design: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the geometry columns from a design dict."""
    return {col: design.get(col) for col in _GEOM_COLS}


def _compute_checks(
    metrics:       Dict[str, Any],
    sigma_limit:   float,
    tau_limit:     float,
    n_buck_target: float,
) -> Dict[str, Any]:
    """
    Compute all pass/fail checks and the overall pass_fail label.

    Returns a flat dict of check columns to be merged into the per-design row.
    """
    checks: Dict[str, Any] = {}

    # --- Static stress ---
    s_max = metrics.get('sigma_max', 0.0)
    t_max = metrics.get('tau_max',   0.0)
    u_sigma = s_max / sigma_limit if sigma_limit > 0 else 0.0
    u_tau   = t_max / tau_limit   if tau_limit   > 0 else 0.0
    utilization = max(u_sigma, u_tau)
    checks['utilization']         = utilization
    checks['check_static_passed'] = bool(utilization <= 1.0)

    # --- Buckling ---
    n_buck = metrics.get('n_buck', float('inf'))
    checks['n_buck']               = n_buck
    checks['P_cr']                 = metrics.get('P_cr', float('nan'))
    checks['N_max_comp']           = metrics.get('N_max_comp', float('nan'))
    checks['check_buckling_passed'] = bool(
        n_buck >= n_buck_target if np.isfinite(n_buck) else True
    )

    # --- Fatigue Goodman+ECY governing safety factor ---
    for comp in ('rod', 'crank', 'pin'):
        n = metrics.get(f'n_{comp}', float('inf'))
        checks[f'n_{comp}']                   = n
        checks[f'S_e_{comp}']                 = metrics.get(f'S_e_{comp}',       float('nan'))
        checks[f'sigma_a_eq_{comp}']          = metrics.get(f'sigma_a_eq_{comp}', float('nan'))
        checks[f'sigma_m_eq_{comp}']          = metrics.get(f'sigma_m_eq_{comp}', float('nan'))
        checks[f'n_f_{comp}']                 = metrics.get(f'n_f_{comp}',        float('nan'))
        checks[f'n_y_{comp}']                 = metrics.get(f'n_y_{comp}',        float('nan'))
        checks[f'N_f_{comp}']                 = metrics.get(f'N_f_{comp}',        float('nan'))
        checks[f't_f_{comp}']                 = metrics.get(f't_f_{comp}',        float('nan'))
        checks[f'check_fatigue_{comp}_passed'] = bool(
            n >= 1.0 if np.isfinite(n) else True
        )

    # --- Miner's rule damage ---
    for comp in ('rod', 'crank', 'pin'):
        D = metrics.get(f'D_{comp}', None)
        checks[f'D_{comp}'] = D
        if D is not None:
            checks[f'check_miner_{comp}_passed'] = bool(D < 1.0)
        else:
            checks[f'check_miner_{comp}_passed'] = True  # not computed → not failed

    # --- sigma/tau sweep summary ---
    checks['sigma_max']       = metrics.get('sigma_max',       float('nan'))
    checks['tau_max']         = metrics.get('tau_max',         float('nan'))
    checks['theta_sigma_max'] = metrics.get('theta_sigma_max', float('nan'))
    checks['theta_tau_max']   = metrics.get('theta_tau_max',   float('nan'))

    # --- Overall pass/fail ---
    bool_checks = [
        checks['check_static_passed'],
        checks['check_buckling_passed'],
        checks['check_fatigue_rod_passed'],
        checks['check_fatigue_crank_passed'],
        checks['check_fatigue_pin_passed'],
        checks['check_miner_rod_passed'],
        checks['check_miner_crank_passed'],
        checks['check_miner_pin_passed'],
    ]
    checks['pass_fail'] = 1 if all(bool_checks) else 0

    return checks


def _prefix_rows(
    history: List[Dict[str, Any]],
    prefix:  Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Prepend prefix columns (design_id + geometry) to each history row."""
    return [{**prefix, **row} for row in history]


def _drop_duplicate_designs(
    dfs: List[pd.DataFrame],
    per_design_df: pd.DataFrame,
) -> tuple:
    """
    Remove designs whose rounded geometry is identical to an earlier design.

    Uses the per-design DataFrame (one row per design_id) to find duplicates
    based on the geometry columns, then filters all DataFrames to the surviving
    design_ids. The first occurrence of each unique geometry is kept.

    Args:
        dfs:            All seven DataFrames to filter, in order.
        per_design_df:  One of the per-design DFs (e.g. fatigue_df) used to
                        identify duplicates. Must contain 'design_id' and
                        all _GEOM_COLS that are present.

    Returns:
        Tuple of filtered DataFrames (same order as input dfs) and the
        number of duplicate designs removed.
    """
    if per_design_df.empty:
        return tuple(dfs), 0

    geom_cols = [c for c in _GEOM_COLS if c in per_design_df.columns]
    unique_rows = per_design_df.drop_duplicates(subset=geom_cols, keep='first')
    keep_ids = set(unique_rows['design_id'])
    n_removed = len(per_design_df) - len(keep_ids)

    if n_removed == 0:
        return tuple(dfs), 0

    filtered = []
    for df in dfs:
        if df.empty or 'design_id' not in df.columns:
            filtered.append(df)
        else:
            filtered.append(df[df['design_id'].isin(keep_ids)].reset_index(drop=True))

    return tuple(filtered), n_removed


def _build_summary(
    n_stage1:   int,
    n_stage2:   int,
    n_dropped:  int,
    passed_df:  pd.DataFrame,
    failed_df:  pd.DataFrame,
    start_time: float,
) -> Dict[str, Any]:
    n_pass = len(passed_df)
    n_fail = len(failed_df)
    n_eval = n_pass + n_fail
    return {
        'n_stage1':        n_stage1,
        'n_stage2':        n_stage2,
        'n_dropped':       n_dropped,
        'n_evaluated':     n_eval,
        'n_passed':        n_pass,
        'n_failed':        n_fail,
        'pass_rate':       n_pass / n_eval if n_eval > 0 else 0.0,
        'time_taken_sec':  time.time() - start_time,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_dataset(config: Dict[str, Any], seed: Optional[int] = None) -> DatasetResult:
    """
    Run the full pipeline and return a DatasetResult with all seven DataFrames.

    Args:
        config: Configuration dictionary (loaded from YAML via load_config).
        seed:   Random seed (overrides config.random_seed if provided).

    Returns:
        DatasetResult
    """
    start_time = time.time()

    run_seed = seed if seed is not None else config.get('random_seed', 42)
    np.random.seed(run_seed)

    # -------------------------------------------------------------------------
    # Stage 1 — 2D kinematic synthesis
    # -------------------------------------------------------------------------
    logger.info("Stage 1: kinematic synthesis …")
    valid_2d = stage1_kinematic.generate_valid_2d_mechanisms(config)
    n_stage1 = len(valid_2d)
    logger.info("Stage 1 complete — %d valid 2D mechanisms.", n_stage1)

    if n_stage1 == 0:
        logger.warning("No valid 2D mechanisms. Aborting.")
        empty = pd.DataFrame()
        return DatasetResult(empty, empty, empty, empty, empty, empty, empty,
                             {'error': 'No valid 2D mechanisms'})

    # -------------------------------------------------------------------------
    # Operating / limit parameters (read once from config)
    # -------------------------------------------------------------------------
    _ctx       = 'generate.generate_dataset'
    operating  = config.get('operating', {})
    limits_cfg = config.get('limits', {})
    material   = config.get('material', {})
    sa_cfg     = config.get('stress_analysis', {})

    rpm           = float(get_or_warn(operating,  'RPM',          30,    context=_ctx))
    omega         = rpm * 2.0 * np.pi / 60.0
    mu_default    = float(get_or_warn(operating,  'mu',           0.0,   context=_ctx))
    g_default     = float(get_or_warn(operating,  'g',            9.81,  context=_ctx))
    sigma_allow   = float(get_or_warn(limits_cfg, 'sigma_allow',  1e20,  context=_ctx))
    tau_allow     = float(get_or_warn(limits_cfg, 'tau_allow',    1e20,  context=_ctx))
    safety_factor = float(get_or_warn(limits_cfg, 'safety_factor', 1.0,  context=_ctx))
    n_buck_target = float(get_or_warn(sa_cfg,     'n_buck_target', 3.0,  context=_ctx))

    sigma_limit = sigma_allow / safety_factor
    tau_limit   = tau_allow   / safety_factor

    # Material props injected into every design dict before physics eval
    _mat = {
        'E':             float(get_or_warn(material, 'E',             73.1e9, context=_ctx)),
        'S_ut':          float(get_or_warn(material, 'S_ut',          483e6,  context=_ctx)),
        'S_y':           float(get_or_warn(material, 'S_y',           345e6,  context=_ctx)),
        'S_prime_e':     float(get_or_warn(material, 'S_prime_e',     130e6,  context=_ctx)),
        'sigma_f_prime': float(get_or_warn(material, 'sigma_f_prime', 807e6,  context=_ctx)),
    }

    # Stress-analysis constants injected into every design dict
    _sa = {
        'delta':            float(get_or_warn(sa_cfg, 'delta',            1e-4,   context=_ctx)),
        'Kt_lug':           float(get_or_warn(sa_cfg, 'Kt_lug',           2.34,   context=_ctx)),
        'Kt_hole_torsion':  float(get_or_warn(sa_cfg, 'Kt_hole_torsion',  4.0,    context=_ctx)),
        'n_buck_target':    n_buck_target,
        'N_basquin_anchor': float(get_or_warn(sa_cfg, 'N_basquin_anchor', 2.0e6,  context=_ctx)),
        'z_a_reliability':  float(get_or_warn(sa_cfg, 'z_a_reliability',  3.091,  context=_ctx)),
        # Crank angular acceleration — 0.0 for constant RPM (per instructions.md)
        'alpha_r':          0.0,
    }

    # -------------------------------------------------------------------------
    # Stage 2 — 3D embodiment → mass props → physics eval
    # -------------------------------------------------------------------------
    logger.info("Stage 2 + physics evaluation …")

    # Accumulator lists for the seven DataFrames
    kin_rows:  List[Dict] = []
    dyn_rows:  List[Dict] = []
    str_rows:  List[Dict] = []
    fat_rows:  List[Dict] = []
    buck_rows: List[Dict] = []
    pass_rows: List[Dict] = []
    fail_rows: List[Dict] = []

    design_id = 0
    n_stage2  = 0
    n_dropped = 0

    for design in stage2_embodiment.iter_expand_to_3d(valid_2d, config):
        n_stage2 += 1

        # --- Compute mass properties ---
        design_eval = design.copy()
        design_eval['omega'] = omega
        design_eval.setdefault('mu', mu_default)
        design_eval.setdefault('g',  g_default)

        if mp is not None:
            try:
                mass_props = mp.compute_design_mass_properties(design_eval, config)
                design_eval.update(mass_props)
            except Exception as exc:
                logger.warning("Mass props failed for stage2 design #%d: %s", n_stage2, exc)
                n_dropped += 1
                continue

        # --- Inject material + stress-analysis constants ---
        design_eval.update(_mat)
        design_eval.update(_sa)
        design_eval['n_rpm']        = rpm
        design_eval['total_cycles'] = float(
            get_or_warn(operating, 'TotalCycles', 18720000, context=_ctx)
        )

        # --- Physics evaluation ---
        if engine is None:
            n_dropped += 1
            continue

        try:
            metrics = engine.evaluate_design(design_eval)
        except Exception as exc:
            logger.warning("Engine failed for design #%d: %s", n_stage2, exc)
            n_dropped += 1
            continue

        if not metrics.get('valid_physics', False):
            n_dropped += 1
            continue

        # --- Assign design_id and build geometry prefix ---
        design_id += 1
        geom = _extract_geom(design_eval)
        prefix_geom = {'design_id': design_id, **geom}

        # --- Per-angle rows (kinematics / dynamics / stresses) ---
        kin_rows.extend(_prefix_rows(metrics['kinematics_history'], prefix_geom))
        dyn_rows.extend(_prefix_rows(metrics['dynamics_history'],   prefix_geom))
        str_rows.extend(_prefix_rows(metrics['stresses_history'],   prefix_geom))

        # --- Fatigue row (per-design) ---
        fat_row: Dict[str, Any] = {'design_id': design_id, **geom}
        for comp in ('rod', 'crank', 'pin'):
            for key in ('sigma_max', 'sigma_min', 'sigma_m', 'sigma_a',
                        'tau_m', 'tau_a', 'R', 'sigma_a_eq', 'sigma_m_eq',
                        'S_e', 'n_f', 'n_y', 'n', 'b_B', 'N_f', 't_f',
                        'D', 'failed_miner'):
                fat_row[f'{key}_{comp}'] = metrics.get(f'{key}_{comp}')
        fat_rows.append(fat_row)

        # --- Buckling row (per-design) ---
        # I_min_r = w_rod * t_rod^3 / 12  (weak axis, same formula as buckling.py)
        w_rod = float(design_eval.get('width_l', 0.0))
        t_rod = float(design_eval.get('thickness_l', 0.0))
        buck_rows.append({
            'design_id':       design_id,
            **geom,
            'I_min_r':         w_rod * t_rod**3 / 12.0,
            'P_cr':            metrics.get('P_cr'),
            'N_max_comp':      metrics.get('N_max_comp'),
            'n_buck':          metrics.get('n_buck'),
            'buckling_passed': metrics.get('buckling_passed'),
        })

        # --- Checks + pass/fail ---
        checks = _compute_checks(metrics, sigma_limit, tau_limit, n_buck_target)

        config_row: Dict[str, Any] = {
            'design_id': design_id,
            **geom,
            **checks,
        }

        if checks['pass_fail'] == 1:
            pass_rows.append(config_row)
        else:
            fail_rows.append(config_row)

    logger.info(
        "Pipeline complete — %d evaluated, %d passed, %d failed, %d dropped.",
        design_id, len(pass_rows), len(fail_rows), n_dropped,
    )

    # -------------------------------------------------------------------------
    # Build DataFrames
    # -------------------------------------------------------------------------
    def _to_df(rows: List[Dict]) -> pd.DataFrame:
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    kinematics_df = _to_df(kin_rows)
    dynamics_df   = _to_df(dyn_rows)
    stresses_df   = _to_df(str_rows)
    fatigue_df    = _to_df(fat_rows)
    buckling_df   = _to_df(buck_rows)
    passed_df     = _to_df(pass_rows)
    failed_df     = _to_df(fail_rows)

    # -------------------------------------------------------------------------
    # Deduplicate — drop designs with identical rounded geometry (post-rounding
    # collision: two raw samples that round to the same dimensions)
    # -------------------------------------------------------------------------
    (
        kinematics_df, dynamics_df, stresses_df,
        fatigue_df, buckling_df, passed_df, failed_df,
    ), n_duplicates = _drop_duplicate_designs(
        [kinematics_df, dynamics_df, stresses_df,
         fatigue_df, buckling_df, passed_df, failed_df],
        fatigue_df,
    )
    if n_duplicates > 0:
        logger.info("Removed %d duplicate geometry design(s) (post-rounding collision).", n_duplicates)

    summary = _build_summary(n_stage1, n_stage2, n_dropped, passed_df, failed_df, start_time)
    summary['n_duplicates_removed'] = n_duplicates

    return DatasetResult(
        kinematics_df, dynamics_df, stresses_df,
        fatigue_df, buckling_df,
        passed_df, failed_df,
        summary,
    )
