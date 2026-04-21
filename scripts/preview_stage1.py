"""
preview_stage1.py - Stage 1 kinematic screening, writes valid 2D geometries to CSV.

Output: data/stage1_preview/stage1_geometries.csv
Usage:  python scripts/preview_stage1.py
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
from mech390.datagen import stage1_kinematic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preview_stage1")

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "stage1_preview"
OUTPUT_FILENAME = "stage1_geometries.csv"
CSV_COLUMNS = ["r", "l", "e", "ROM", "QRR", "theta_min", "theta_max"]
DESCRIBE_MAX_ROWS = 200000

### Core

def run() -> None:
    config = get_baseline_config()
    effective_seed = config.get("random_seed", 42)
    out_dir = DEFAULT_OUT_DIR

    ### Configuration
    geo    = config.get("geometry", {})
    ops    = config.get("operating", {})
    samp   = config.get("sampling", {})

    logger.info(
        "Configuration - seed=%d | method=%s | n_samples=%s | "
        "ROM=%.4f | QRR=[%.2f, %.2f]",
        effective_seed,
        samp.get("method", "n/a"),
        samp.get("n_samples", "n/a"),
        ops.get("ROM", float("nan")),
        ops.get("QRR", {}).get("min", float("nan")),
        ops.get("QRR", {}).get("max", float("nan")),
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

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILENAME

    n_valid = 0
    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for design in stage1_kinematic.iter_valid_2d_mechanisms(config):
            writer.writerow({k: design.get(k) for k in CSV_COLUMNS})
            n_valid += 1
            # flush every 1000 rows so the file grows visibly during long runs
            if n_valid % 1000 == 0:
                csv_file.flush()

    elapsed = time.perf_counter() - t0
    n_candidates = samp.get("n_samples", "n/a")

    if isinstance(n_candidates, int) and n_candidates > 0:
        acceptance_rate = n_valid / n_candidates * 100.0
    else:
        acceptance_rate = float("nan")

    logger.info(
        "Stage 1 complete in %.2f s - %d / %s designs passed (%.1f %%).",
        elapsed, n_valid, n_candidates, acceptance_rate,
    )

    if n_valid == 0:
        logger.warning(
            "No valid designs were produced. "
            "Check config bounds (ROM, QRR, geometry ranges) and retry."
        )
        sys.exit(1)

    logger.info("CSV written -> %s  (%d rows x %d columns)", out_path, n_valid, len(CSV_COLUMNS))

    ### Quick statistics
    print("\n" + "=" * 64)
    print("  Stage 1 Preview - Summary")
    print("=" * 64)
    print(f"  Valid designs   : {n_valid:>10,}")
    print(f"  Acceptance rate : {acceptance_rate:>10.2f} %")
    print(f"  Elapsed time    : {elapsed:>10.2f} s")
    print(f"  Output file     : {out_path}")
    print("=" * 64)

    if n_valid <= DESCRIBE_MAX_ROWS:
        df = pd.read_csv(out_path)
        print("\nDescriptive statistics:\n")
        print(df.describe().to_string())
        print()
    else:
        print(
            f"\nDescriptive statistics skipped because row count ({n_valid}) exceeds "
            f"{DESCRIBE_MAX_ROWS}.\n"
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
