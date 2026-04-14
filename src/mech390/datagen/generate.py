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
  - Static stress:   utilization <= limits.utilization_max
  - Static FoS:      n_static_* >= limits.n_static_min
  - Buckling:        n_buck >= limits.n_buck_min
  - Fatigue Goodman: n_rod, n_crank, n_pin >= configured minima
  - Fatigue Miner:   D_rod, D_crank, D_pin < configured maxima (when total_cycles is set)
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mech390.datagen import stage1_kinematic, stage2_embodiment
from mech390.datagen.validation import compute_checks
from mech390.physics import buckling as _buckling
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
    'd_shaft_A', 'pin_diameter_B', 'pin_diameter_C',
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


def _as_scalar(name: str, value: Any, default: float) -> float:
    """
    Coerce a config value to a scalar float.

    Supports:
      - scalar numeric values
      - range dicts {'min': x, 'max': y} (uses conservative min)
    """
    if value is None:
        return float(default)
    if isinstance(value, dict):
        v_min = float(value.get('min', default))
        v_max = float(value.get('max', v_min))
        if abs(v_max - v_min) > 1e-12:
            logger.warning(
                "%s provided as a range (min=%g, max=%g); using conservative min=%g.",
                name, v_min, v_max, v_min,
            )
        return v_min
    return float(value)


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

def generate_dataset(
    config: Dict[str, Any], 
    seed: Optional[int] = None,
    out_dir: Optional[Any] = None,
    chunk_size: int = 10000
) -> DatasetResult:
    """
    Run the full pipeline and return a DatasetResult with all seven DataFrames.
    If out_dir is provided, writes CSV chunks on the fly to prevent RAM exhaustion.

    Args:
        config:     Configuration dictionary (loaded from YAML via load_config).
        seed:       Random seed (overrides config.random_seed if provided).
        out_dir:    Path to output directory. If set, streams to CSV.
        chunk_size: Number of evaluated designs to accumulate before writing to disk.

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

    rpm            = float(get_or_warn(operating, 'RPM',            30,   context=_ctx))
    sweep_step_deg = float(get_or_warn(operating, 'sweep_step_deg', 15.0, context=_ctx))
    omega          = rpm * 2.0 * np.pi / 60.0
    mu_default    = float(get_or_warn(operating,  'mu',           0.0,   context=_ctx))
    g_default     = float(get_or_warn(operating,  'g',            9.81,  context=_ctx))
    m_block       = float(get_or_warn(operating,  'm_block',      0.0,   context=_ctx))
    safety_factor = float(get_or_warn(limits_cfg, 'safety_factor', 1.0,  context=_ctx))
    n_buck_min = float(get_or_warn(limits_cfg, 'n_buck_min', get_or_warn(sa_cfg, 'n_buck_target', 3.0, context=_ctx), context=_ctx))
    utilization_max = float(get_or_warn(limits_cfg, 'utilization_max', 1.0, context=_ctx))
    n_static_min = float(get_or_warn(limits_cfg, 'n_static_min', 1.0, context=_ctx))
    n_fatigue_default = float(get_or_warn(limits_cfg, 'n_fatigue_min', 1.0, context=_ctx))
    d_miner_max = float(get_or_warn(limits_cfg, 'D_miner_max', 1.0, context=_ctx))

    # Static stress limits derived from S_y (Von Mises for shear yield: S_sy = 0.577 * S_y).
    sigma_yield_material = float(get_or_warn(material, 'S_y', 345e6, context=_ctx))
    tau_yield_material   = 0.577 * sigma_yield_material
    sigma_limit = sigma_yield_material / safety_factor
    tau_limit   = tau_yield_material   / safety_factor

    # Material props injected into every design dict before physics eval
    _mat = {
        'E':             float(get_or_warn(material, 'E',             73.1e9, context=_ctx)),
        'S_ut':          float(get_or_warn(material, 'S_ut',          483e6,  context=_ctx)),
        'S_y':           float(get_or_warn(material, 'S_y',           345e6,  context=_ctx)),
        'Sn':            float(get_or_warn(material, 'Sn',  133e6, context=_ctx)),
    }

    # Stress-analysis constants injected into every design dict
    _sa = {
        'delta':            float(get_or_warn(sa_cfg, 'diametral_clearance_m', 1e-4, context=_ctx)),
        'Kt_lug':           float(get_or_warn(sa_cfg, 'Kt_lug',           2.34,   context=_ctx)),
        'Kt_hole_torsion':  float(get_or_warn(sa_cfg, 'Kt_hole_torsion',  4.0,    context=_ctx)),
        'n_buck_target':    n_buck_min,
        'L_bearing':        float(get_or_warn(sa_cfg, 'L_bearing',  0.010, context=_ctx)),
        'n_shaft_min':      float(get_or_warn(limits_cfg, 'n_shaft_min', 2.0, context=_ctx)),
        'basquin_A':        float(get_or_warn(sa_cfg, 'basquin_A', 924e6,  context=_ctx)),
        'basquin_b':        float(get_or_warn(sa_cfg, 'basquin_b', -0.086, context=_ctx)),
        'C_sur':            float(get_or_warn(sa_cfg, 'C_sur',           0.88,   context=_ctx)),
        'C_st':             float(get_or_warn(sa_cfg, 'C_st',            1.0,    context=_ctx)),
        'C_R':              float(get_or_warn(sa_cfg, 'C_R',             0.81,   context=_ctx)),
        'C_f':              float(get_or_warn(sa_cfg, 'C_f',             1.0,    context=_ctx)),
        'C_m':              float(get_or_warn(sa_cfg, 'C_m',             1.0,    context=_ctx)),
        # Crank angular acceleration — 0.0 for constant RPM (per instructions.md)
        'alpha_r':          0.0,
    }

    check_limits = {
        'utilization_max': utilization_max,
        'n_buck_min': n_buck_min,
        'n_static_rod_min': float(get_or_warn(limits_cfg, 'n_static_rod_min', n_static_min, context=_ctx)),
        'n_static_crank_min': float(get_or_warn(limits_cfg, 'n_static_crank_min', n_static_min, context=_ctx)),
        'n_static_pin_min': float(get_or_warn(limits_cfg, 'n_static_pin_min', n_static_min, context=_ctx)),
        'n_fatigue_rod_min': float(get_or_warn(limits_cfg, 'n_fatigue_rod_min', n_fatigue_default, context=_ctx)),
        'n_fatigue_crank_min': float(get_or_warn(limits_cfg, 'n_fatigue_crank_min', n_fatigue_default, context=_ctx)),
        'n_fatigue_pin_min': float(get_or_warn(limits_cfg, 'n_fatigue_pin_min', n_fatigue_default, context=_ctx)),
        'D_miner_rod_max': float(get_or_warn(limits_cfg, 'D_miner_rod_max', d_miner_max, context=_ctx)),
        'D_miner_crank_max': float(get_or_warn(limits_cfg, 'D_miner_crank_max', d_miner_max, context=_ctx)),
        'D_miner_pin_max': float(get_or_warn(limits_cfg, 'D_miner_pin_max', d_miner_max, context=_ctx)),
        'n_shaft_min':     float(get_or_warn(limits_cfg, 'n_shaft_min', 2.0, context=_ctx)),
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
    n_duplicates = 0
    seen_geoms = set()

    total_passed = 0
    total_failed = 0
    files_written = set()

    def _to_df(rows: List[Dict]) -> pd.DataFrame:
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _write_chunk(dest_dir):
        nonlocal total_passed, total_failed
        from pathlib import Path
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        def _write_df(rows, filename):
            if not rows:
                return
            df = _to_df(rows)
            mode = 'a' if filename in files_written else 'w'
            header = filename not in files_written
            df.to_csv(dest / filename, mode=mode, header=header, index=False)
            files_written.add(filename)

        _write_df(kin_rows,  "kinematics.csv")
        _write_df(dyn_rows,  "dynamics.csv")
        _write_df(str_rows,  "stresses.csv")
        _write_df(fat_rows,  "fatigue.csv")
        _write_df(buck_rows, "buckling.csv")
        _write_df(pass_rows, "passed_configs.csv")
        _write_df(fail_rows, "failed_configs.csv")
            
        total_passed += len(pass_rows)
        total_failed += len(fail_rows)
        
        kin_rows.clear()
        dyn_rows.clear()
        str_rows.clear()
        fat_rows.clear()
        buck_rows.clear()
        pass_rows.clear()
        fail_rows.clear()

    for design in stage2_embodiment.iter_expand_to_3d(valid_2d, config):
        n_stage2 += 1

        # --- Compute mass properties ---
        design_eval = design.copy()
        design_eval['omega']   = omega
        design_eval['m_block'] = m_block
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

        # --- On the fly deduplication ---
        geom = _extract_geom(design_eval)
        geom_tuple = tuple(geom.get(c, 0.0) for c in _GEOM_COLS)
        if geom_tuple in seen_geoms:
            n_duplicates += 1
            continue
        seen_geoms.add(geom_tuple)

        # --- Derived design metrics (geometry + mass, computed once) ---
        design_eval['total_mass'] = (
            design_eval.get('mass_crank',  0.0)
            + design_eval.get('mass_rod',   0.0)
            + design_eval.get('mass_slider', 0.0)
        )
        _slider_cfg = config.get('geometry', {}).get('slider', {})
        _s_h = float(_slider_cfg.get('height', 0.02))   # slider block height (y) [m] — vertical extent
        _s_w = float(_slider_cfg.get('width',  0.02))   # slider block width (z) [m]  — OOP thickness for Pin C bearing
        _s_l = float(_slider_cfg.get('length', 0.02))   # slider block length (x) [m]
        _r   = float(design_eval['r'])
        _l   = float(design_eval['l'])
        _e   = float(design_eval['e'])
        _tr  = float(design_eval.get('thickness_r',    0.0))
        _tl  = float(design_eval.get('thickness_l',    0.0))
        _pA  = float(design_eval.get('d_shaft_A', 0.0))  # shaft A diameter
        # Bounding-box dimensions — see instructions.md Section 3.4 for full derivation.
        #
        # Assembly cross-section (z-axis, out-of-plane view from +z):
        #   R and L touch face-to-face at a shared contact plane (blue line).
        #   L and S share the same centreline (centred on the contact plane side of L).
        #   Right of contact plane: t_l/2 (half of L) + t_s/2 (half of S, centred on L)
        #   Left  of contact plane: max(t_r, (t_s - t_l)/2)
        #     — t_r if the crank is thicker than the slider overhang; else slider overhang dominates
        _T = (_tl + _s_h) / 2.0 + max(_tr, (_s_h - _tl) / 2.0)
        #
        # Vertical extent (y-axis): crank pivot A is at y=0; slider guide is e below A (y = -e).
        #   Top:    crank pin B at y = +r
        #   Bottom: lower of crank pin at y = -r  OR  slider block bottom at y = -(e + s_h/2)
        _H = _r + max(_r, _e + _s_h / 2.0)
        #
        # Horizontal extent (x-axis): crank pivot A at origin.
        #   Left:  crank pin B at x = -r (θ = 180°)
        #   Right: slider pin C at maximum extension x = sqrt((r+l)² - e²), plus half slider block
        _L = _r + float(np.sqrt(max((_r + _l)**2 - _e**2, 0.0))) + _s_l / 2.0
        design_eval['volume_envelope'] = _T * _H * _L
        design_eval['slider_height']  = _s_w   # slider OOP thickness (z, width in YAML) — needed by _pin_stresses bearing at C

        # --- Inject material + stress-analysis constants ---
        design_eval.update(_mat)
        design_eval.update(_sa)
        design_eval['n_rpm']          = rpm
        design_eval['sweep_step_deg'] = sweep_step_deg
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
                        'S_n_prime', 'n_f', 'n_y', 'n', 'b_B', 'N_f', 't_f',
                        'D', 'failed_miner'):
                fat_row[f'{key}_{comp}'] = metrics.get(f'{key}_{comp}')
        fat_rows.append(fat_row)

        # --- Buckling row (per-design) ---
        # I_min_r = min(w·t³, t·w³) / 12  — always the weaker axis (matches buckling.py).
        w_rod = float(design_eval.get('width_l', 0.0))
        t_rod = float(design_eval.get('thickness_l', 0.0))
        buck_rows.append({
            'design_id':       design_id,
            **geom,
            'I_min_r':         _buckling.I_weak_axis(w_rod, t_rod),
            'P_cr':            metrics.get('P_cr'),
            'N_max_comp':      metrics.get('N_max_comp'),
            'n_buck':          metrics.get('n_buck'),
            'buckling_passed': metrics.get('buckling_passed'),
        })

        # --- Checks + pass/fail ---
        checks = compute_checks(metrics, sigma_limit, tau_limit, check_limits)

        config_row: Dict[str, Any] = {
            'design_id': design_id,
            **geom,
            **checks,
        }

        # --- Additional ML features (not part of pass/fail logic) ---
        config_row['total_mass']      = design_eval.get('total_mass')
        config_row['volume_envelope'] = design_eval.get('volume_envelope')
        config_row['tau_A_max']       = metrics.get('tau_A_max')
        config_row['E_rev']           = metrics.get('E_rev')
        config_row['F_A_max']         = metrics.get('F_A_max')
        config_row['F_B_max']         = metrics.get('F_B_max')
        config_row['F_C_max']         = metrics.get('F_C_max')
        if checks['pass_fail'] == 1:
            pass_rows.append(config_row)
        else:
            fail_rows.append(config_row)

        if out_dir is not None and (len(pass_rows) + len(fail_rows)) >= chunk_size:
            _write_chunk(out_dir)

    # Write remaining items if out_dir is provided
    if out_dir is not None and (pass_rows or fail_rows):
        _write_chunk(out_dir)

    logger.info(
        "Pipeline complete — %d evaluated, %d passed, %d failed, %d dropped.",
        design_id, total_passed if out_dir else len(pass_rows),
        total_failed if out_dir else len(fail_rows), n_dropped,
    )

    # -------------------------------------------------------------------------
    # Build DataFrames (if not chunked to disk)
    # -------------------------------------------------------------------------
    if out_dir is None:
        kinematics_df = _to_df(kin_rows)
        dynamics_df   = _to_df(dyn_rows)
        stresses_df   = _to_df(str_rows)
        fatigue_df    = _to_df(fat_rows)
        buckling_df   = _to_df(buck_rows)
        passed_df     = _to_df(pass_rows)
        failed_df     = _to_df(fail_rows)
    else:
        empty = pd.DataFrame()
        kinematics_df = dynamics_df = stresses_df = fatigue_df = buckling_df = passed_df = failed_df = empty

    # On-the-fly deduplication was already applied.
    if n_duplicates > 0:
        logger.info("Removed %d duplicate geometry design(s) (post-rounding collision).", n_duplicates)

    # Note: Using out_dir implies passed_df and failed_df are empty internally. 
    # To build the correct summary if out_dir is used, use total counts.
    # We pass mock DFs of the correct length to _build_summary, or we just pass the count?
    # Actually, _build_summary uses `len(passed_df)`, so let's adjust it!
    n_pass = total_passed if out_dir else len(passed_df)
    n_fail = total_failed if out_dir else len(failed_df)
    n_eval = n_pass + n_fail
    
    summary = {
        'n_stage1':        n_stage1,
        'n_stage2':        n_stage2,
        'n_dropped':       n_dropped,
        'n_evaluated':     n_eval,
        'n_passed':        n_pass,
        'n_failed':        n_fail,
        'pass_rate':       n_pass / n_eval if n_eval > 0 else 0.0,
        'time_taken_sec':  time.time() - start_time,
        'n_duplicates_removed': n_duplicates
    }

    return DatasetResult(
        kinematics_df, dynamics_df, stresses_df,
        fatigue_df, buckling_df,
        passed_df, failed_df,
        summary,
    )
