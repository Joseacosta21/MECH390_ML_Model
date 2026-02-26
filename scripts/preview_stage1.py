"""
preview_stage1.py
-----------------
Runs **Stage 1** (2D kinematic screening) of the MECH390 data-generation
pipeline and writes all valid mechanism geometries to a CSV file.

Stage 1 summary
~~~~~~~~~~~~~~~
  1. Sample rod length ``l`` and offset ``e`` from configured ranges.
  2. Numerically solve for crank radius ``r`` that satisfies the target ROM.
  3. Reject designs that violate ``l >= 2.5 * r`` or the QRR bounds.
  4. Return every passing design as a dict {r, l, e, ROM, QRR, theta_min,
     theta_max}.

No Stage 2 embodiment, 3-D expansion, or physics evaluation is performed.

Usage
-----
  # Default — baseline.yaml, seed from config, output in data/stage1_preview/
  python scripts/preview_stage1.py

  # With overrides
  python scripts/preview_stage1.py \\
      --config  configs/generate/baseline.yaml \\
      --seed    123 \\
      --out-dir data/stage1_preview

Output
------
  <out_dir>/stage1_geometries.csv
      Columns: r, l, e, ROM, QRR, theta_min, theta_max
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project src/ is on the path when the script is executed directly
# from the project root (e.g. `python scripts/preview_stage1.py`).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mech390.config import load_config, get_baseline_config
from mech390.datagen import stage1_kinematic

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preview_stage1")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "stage1_preview"
OUTPUT_FILENAME = "stage1_geometries.csv"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preview_stage1",
        description=(
            "Run Stage 1 kinematic synthesis and export valid 2D mechanism "
            "geometries to a CSV file."
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
    return parser


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run(config_path: str | None, seed: int | None, out_dir: Path) -> None:
    """Execute Stage 1 and persist results to CSV."""

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
    geo    = config.get("geometry", {})
    ops    = config.get("operating", {})
    samp   = config.get("sampling", {})

    logger.info(
        "Configuration — seed=%d | method=%s | n_samples=%s | "
        "ROM=%.4f | QRR=[%.2f, %.2f]",
        effective_seed,
        samp.get("method", "n/a"),
        samp.get("n_samples", "n/a"),
        ops.get("ROM", float("nan")),
        ops.get("QRR", {}).get("min", float("nan")),
        ops.get("QRR", {}).get("max", float("nan")),
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

    # ---- Stage 1 -----------------------------------------------------------
    logger.info("Starting Stage 1 kinematic synthesis …")
    t0 = time.perf_counter()

    valid_designs = stage1_kinematic.generate_valid_2d_mechanisms(config)

    elapsed = time.perf_counter() - t0
    n_candidates = samp.get("n_samples", "n/a")
    n_valid      = len(valid_designs)

    if isinstance(n_candidates, int) and n_candidates > 0:
        acceptance_rate = n_valid / n_candidates * 100.0
    else:
        acceptance_rate = float("nan")

    logger.info(
        "Stage 1 complete in %.2f s — %d / %s designs passed (%.1f %%).",
        elapsed,
        n_valid,
        n_candidates,
        acceptance_rate,
    )

    if n_valid == 0:
        logger.warning(
            "No valid designs were produced. "
            "Check config bounds (ROM, QRR, geometry ranges) and retry."
        )
        sys.exit(1)

    # ---- Build DataFrame ---------------------------------------------------
    df = pd.DataFrame(valid_designs)

    # Canonical column order matching Stage 1 output dict keys
    column_order = ["r", "l", "e", "ROM", "QRR", "theta_min", "theta_max"]
    # Only keep columns that actually exist (forward-compatible)
    df = df[[c for c in column_order if c in df.columns]]

    # ---- Write CSV ---------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILENAME

    df.to_csv(out_path, index=False, float_format="%.8g")
    logger.info("CSV written → %s  (%d rows × %d columns)", out_path, len(df), len(df.columns))

    # ---- Quick statistics --------------------------------------------------
    print("\n" + "=" * 64)
    print("  Stage 1 Preview — Summary")
    print("=" * 64)
    print(f"  Valid designs   : {n_valid:>10,}")
    print(f"  Acceptance rate : {acceptance_rate:>10.2f} %")
    print(f"  Elapsed time    : {elapsed:>10.2f} s")
    print(f"  Output file     : {out_path}")
    print("=" * 64)
    print("\nDescriptive statistics:\n")
    print(df.describe().to_string())
    print()


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
        )
    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
