"""
preview_stage2.py
-----------------
Runs **Stage 1** (2D kinematic screening) followed by **Stage 2**
(3D embodiment expansion) of the MECH390 data-generation pipeline.

For every valid 3D design the script also computes mass properties
via ``mech390.physics.mass_properties.compute_design_mass_properties``.
Results are written incrementally to a CSV file.

Stress evaluation is NOT performed here — ``stresses.py`` is not yet
fully implemented.  The pass/fail column is therefore NOT included in
this preview output.

Usage
-----
  # Default — baseline.yaml, seed from config, output in data/stage2_preview/
  python scripts/preview_stage2.py

  # With overrides
  python scripts/preview_stage2.py \\
      --config  configs/generate/test_small.yaml \\
      --seed    123 \\
      --out-dir data/stage2_preview \\
      --max-2d  200

Output
------
  <out_dir>/stage2_designs.csv
      Columns: all Stage-1 columns, all 3D geometry columns,
               all mass-property columns.
"""

import argparse
import csv
import logging
import sys
import os
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project src/ is on the path when the script is executed directly
# from the project root (e.g. `python scripts/preview_stage2.py`).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mech390.config import load_config, get_baseline_config
from mech390.datagen import stage1_kinematic, stage2_embodiment
from mech390.physics import mass_properties as mp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preview_stage2")

# ---------------------------------------------------------------------------
# Default paths / constants
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR   = PROJECT_ROOT / "data" / "stage2_preview"
OUTPUT_FILENAME   = "stage2_designs.csv"
FLUSH_EVERY       = 500          # rows between CSV flushes
DESCRIBE_MAX_ROWS = 200_000

# Ordered column groups for the output CSV.
STAGE1_COLUMNS = ["r", "l", "e", "ROM", "QRR", "theta_min", "theta_max"]

STAGE2_GEO_COLUMNS = [
    "width_r", "width_l",
    "thickness_r", "thickness_l",
    "pin_diameter_A", "pin_diameter_B", "pin_diameter_C",
]

MASS_PROP_COLUMNS = [
    "rho",
    "mass_crank", "mass_rod", "mass_slider",
    "I_mass_crank_cg_z", "I_mass_rod_cg_z", "I_mass_slider_cg_z",
    "I_area_crank_yy", "I_area_crank_zz",
    "I_area_rod_yy",   "I_area_rod_zz",
    "I_area_slider_yy","I_area_slider_zz",
]

CSV_COLUMNS = STAGE1_COLUMNS + STAGE2_GEO_COLUMNS + MASS_PROP_COLUMNS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preview_stage2",
        description=(
            "Run Stage 1 kinematic synthesis + Stage 2 embodiment expansion "
            "and export valid 3D designs (with mass properties) to a CSV file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a YAML configuration file (default: configs/generate/baseline.yaml).",
    )
    parser.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=None,
        help="Random seed override (default: value from config file).",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where the output CSV will be written.",
    )
    parser.add_argument(
        "--max-2d",
        metavar="INT",
        type=int,
        default=None,
        help=(
            "Cap the number of Stage-1 designs forwarded to Stage 2. "
            "Useful for quick preview runs on configs with large n_samples. "
            "(default: no cap — use all Stage-1 results)"
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run(
    config_path: str | None,
    seed: int | None,
    out_dir: Path,
    max_2d: int | None,
) -> None:
    """Execute Stage 1 → Stage 2, compute mass properties, persist to CSV."""

    # ---- Load configuration ------------------------------------------------
    if config_path is not None:
        logger.info("Loading configuration from: %s", config_path)
        config = load_config(config_path)
    else:
        logger.info("Loading default baseline configuration.")
        config = get_baseline_config()

    # ---- Apply seed override -----------------------------------------------
    if seed is not None:
        logger.info("Overriding random_seed with: %d", seed)
        config["random_seed"] = seed

    effective_seed = config.get("random_seed", 42)

    # ---- Configuration summary ---------------------------------------------
    geo   = config.get("geometry", {})
    ops   = config.get("operating", {})
    samp  = config.get("sampling", {})

    logger.info(
        "Configuration — seed=%d | method=%s | n_samples=%s | "
        "ROM=%.4f | QRR=[%.2f, %.2f] | n_variants_per_2d=%s",
        effective_seed,
        samp.get("method", "n/a"),
        samp.get("n_samples", "n/a"),
        ops.get("ROM", float("nan")),
        ops.get("QRR", {}).get("min", float("nan")),
        ops.get("QRR", {}).get("max", float("nan")),
        samp.get("n_variants_per_2d", "n/a"),
    )
    logger.info(
        "Geometry bounds — r:[%.3f, %.3f]  l:[%.3f, %.3f]  e:[%.3f, %.3f]",
        geo.get("r", {}).get("min", float("nan")),
        geo.get("r", {}).get("max", float("nan")),
        geo.get("l", {}).get("min", float("nan")),
        geo.get("l", {}).get("max", float("nan")),
        geo.get("e", {}).get("min", float("nan")),
        geo.get("e", {}).get("max", float("nan")),
    )

    if max_2d is not None:
        logger.info("Stage-1 cap: forwarding at most %d 2D designs to Stage 2.", max_2d)

    # ---- Stage 1 -----------------------------------------------------------
    logger.info("Starting Stage 1 kinematic synthesis …")
    t0 = time.perf_counter()

    valid_2d = list(stage1_kinematic.iter_valid_2d_mechanisms(config))
    n_stage1 = len(valid_2d)

    t1 = time.perf_counter()
    logger.info(
        "Stage 1 complete in %.2f s — %d valid 2D mechanisms.", t1 - t0, n_stage1
    )

    if n_stage1 == 0:
        logger.warning(
            "No valid 2D mechanisms found. "
            "Check config bounds (ROM, QRR, geometry ranges) and retry."
        )
        sys.exit(1)

    # Apply optional Stage-1 cap before Stage-2 expansion.
    designs_for_stage2 = valid_2d[:max_2d] if max_2d is not None else valid_2d
    n_forwarded = len(designs_for_stage2)
    if max_2d is not None and n_forwarded < n_stage1:
        logger.info(
            "Capped Stage-1 output: forwarding %d / %d designs.",
            n_forwarded, n_stage1,
        )

    # ---- Stage 2 -----------------------------------------------------------
    logger.info(
        "Starting Stage 2 embodiment expansion (%d 2D designs × up to %s variants) …",
        n_forwarded,
        samp.get("n_variants_per_2d", "?"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILENAME

    n_total    = 0
    n_mp_ok    = 0
    n_mp_err   = 0

    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for design_3d in stage2_embodiment.iter_expand_to_3d(designs_for_stage2, config):
            n_total += 1

            # ---- Compute mass properties -----------------------------------
            try:
                mass_props = mp.compute_design_mass_properties(design_3d, config)
            except Exception as exc:
                logger.debug("Mass-properties failed for design #%d: %s", n_total, exc)
                n_mp_err += 1
                mass_props = {col: float("nan") for col in MASS_PROP_COLUMNS}
            else:
                n_mp_ok += 1

            # ---- Build output row ------------------------------------------
            row = {col: design_3d.get(col) for col in STAGE1_COLUMNS + STAGE2_GEO_COLUMNS}
            row.update({col: mass_props.get(col) for col in MASS_PROP_COLUMNS})
            writer.writerow(row)

            if n_total % FLUSH_EVERY == 0:
                csv_file.flush()
                logger.info("  … %d rows written so far.", n_total)

    elapsed = time.perf_counter() - t0

    # ---- Summary -----------------------------------------------------------
    logger.info(
        "Stage 2 complete in %.2f s total — %d 3D designs written "
        "(%d mass-props OK, %d mass-props failed).",
        elapsed, n_total, n_mp_ok, n_mp_err,
    )
    logger.info("CSV written → %s  (%d rows × %d columns)", out_path, n_total, len(CSV_COLUMNS))

    n_requested_stage2 = n_forwarded * samp.get("n_variants_per_2d", 1)
    if isinstance(n_requested_stage2, int) and n_requested_stage2 > 0:
        s2_rate = n_total / n_requested_stage2 * 100.0
    else:
        s2_rate = float("nan")

    print("\n" + "=" * 64)
    print("  Stage 2 Preview — Summary")
    print("=" * 64)
    print(f"  Stage-1 valid designs  : {n_stage1:>10,}")
    print(f"  Forwarded to Stage 2   : {n_forwarded:>10,}")
    print(f"  Stage-2 designs written: {n_total:>10,}")
    print(f"  Mass-props computed OK : {n_mp_ok:>10,}")
    print(f"  Mass-props errors      : {n_mp_err:>10,}")
    print(f"  Elapsed time           : {elapsed:>10.2f} s")
    print(f"  Output file            : {out_path}")
    print("=" * 64)

    if n_total == 0:
        logger.warning("No Stage-2 designs were produced.")
        sys.exit(1)

    if n_total <= DESCRIBE_MAX_ROWS:
        df = pd.read_csv(out_path)
        print("\nDescriptive statistics:\n")
        print(df.describe().to_string())
        print()
    else:
        print(
            f"\nDescriptive statistics skipped because row count ({n_total}) "
            f"exceeds {DESCRIBE_MAX_ROWS}.\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()

    try:
        run(
            config_path=args.config,
            seed=args.seed,
            out_dir=out_dir,
            max_2d=args.max_2d,
        )
    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
