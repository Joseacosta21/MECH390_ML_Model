"""
preview_stage2.py - Stage 1 + Stage 2 embodiment, writes 3D designs with mass props to CSV.

Output: data/stage2_preview/stage2_designs.csv
Usage:  python scripts/preview_stage2.py
"""

import csv
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mech390.config import get_baseline_config
from mech390.datagen import stage1_kinematic, stage2_embodiment
from mech390.physics import mass_properties as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preview_stage2")

DEFAULT_OUT_DIR   = PROJECT_ROOT / "data" / "stage2_preview"
OUTPUT_FILENAME   = "stage2_designs.csv"
FLUSH_EVERY       = 500          # rows between CSV flushes
DESCRIBE_MAX_ROWS = 200_000

# column groups for the output CSV
STAGE1_COLUMNS = ["r", "l", "e", "ROM", "QRR", "theta_min", "theta_max"]

STAGE2_GEO_COLUMNS = [
    "width_r", "width_l",
    "thickness_r", "thickness_l",
    "d_shaft_A", "pin_diameter_B", "pin_diameter_C",
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


### Core

def run() -> None:
    config = get_baseline_config()
    effective_seed = config.get("random_seed", 42)
    out_dir = DEFAULT_OUT_DIR
    max_2d = None

    ### Configuration
    geo   = config.get("geometry", {})
    ops   = config.get("operating", {})
    samp  = config.get("sampling", {})

    logger.info(
        "Configuration - seed=%d | method=%s | n_samples=%s | "
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
        "Geometry bounds - r:[%.3f, %.3f]  l:[%.3f, %.3f]  e:[%.3f, %.3f]",
        geo.get("r", {}).get("min", float("nan")),
        geo.get("r", {}).get("max", float("nan")),
        geo.get("l", {}).get("min", float("nan")),
        geo.get("l", {}).get("max", float("nan")),
        geo.get("e", {}).get("min", float("nan")),
        geo.get("e", {}).get("max", float("nan")),
    )

    ### Stage 1
    logger.info("Starting Stage 1 kinematic synthesis ...")
    t0 = time.perf_counter()

    valid_2d = list(stage1_kinematic.iter_valid_2d_mechanisms(config))
    n_stage1 = len(valid_2d)

    t1 = time.perf_counter()
    logger.info("Stage 1 complete in %.2f s - %d valid 2D mechanisms.", t1 - t0, n_stage1)

    if n_stage1 == 0:
        logger.warning(
            "No valid 2D mechanisms found. "
            "Check config bounds (ROM, QRR, geometry ranges) and retry."
        )
        sys.exit(1)

    # trim stage 1 results if a cap was set
    designs_for_stage2 = valid_2d[:max_2d] if max_2d is not None else valid_2d
    n_forwarded = len(designs_for_stage2)
    if max_2d is not None and n_forwarded < n_stage1:
        logger.info(
            "Capped Stage-1 output: forwarding %d / %d designs.",
            n_forwarded, n_stage1,
        )

    ### Stage 2
    logger.info(
        "Starting Stage 2 embodiment expansion (%d 2D designs x up to %s variants) ...",
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

            ### Compute mass properties
            try:
                mass_props = mp.compute_design_mass_properties(design_3d, config)
            except Exception as exc:
                logger.debug("Mass-properties failed for design #%d: %s", n_total, exc)
                n_mp_err += 1
                mass_props = {col: float("nan") for col in MASS_PROP_COLUMNS}
            else:
                n_mp_ok += 1

            ### Build output row
            row = {col: design_3d.get(col) for col in STAGE1_COLUMNS + STAGE2_GEO_COLUMNS}
            row.update({col: mass_props.get(col) for col in MASS_PROP_COLUMNS})
            writer.writerow(row)

            if n_total % FLUSH_EVERY == 0:
                csv_file.flush()
                logger.info("  ... %d rows written so far.", n_total)

    elapsed = time.perf_counter() - t0

    ### Summary
    logger.info(
        "Stage 2 complete in %.2f s total - %d 3D designs written "
        "(%d mass-props OK, %d mass-props failed).",
        elapsed, n_total, n_mp_ok, n_mp_err,
    )
    logger.info("CSV written -> %s  (%d rows x %d columns)", out_path, n_total, len(CSV_COLUMNS))

    n_requested_stage2 = n_forwarded * samp.get("n_variants_per_2d", 1)
    if isinstance(n_requested_stage2, int) and n_requested_stage2 > 0:
        s2_rate = n_total / n_requested_stage2 * 100.0
    else:
        s2_rate = float("nan")

    print("\n" + "=" * 64)
    print("  Stage 2 Preview - Summary")
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


### Entry point

def main() -> None:
    try:
        run()
    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
