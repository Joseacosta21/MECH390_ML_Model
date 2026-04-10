"""
generate_dataset.py
-------------------
Full dataset generation CLI. Runs the complete pipeline:

  Stage 1 (2D kinematic synthesis)
    → Stage 2 (3D embodiment)
      → Mass properties
        → Physics engine (15° sweep)
          → Stress / Fatigue / Buckling evaluation
            → Pass / Fail labeling

Writes seven CSV files to --out-dir:

  kinematics.csv      — one row per (design, crank angle): positions, velocities, accels
  dynamics.csv        — one row per (design, crank angle): joint forces, torque
  stresses.csv        — one row per (design, crank angle): per-component normal + shear
  fatigue.csv         — one row per design: Goodman / Miner metrics per component
  buckling.csv        — one row per design: Euler buckling metrics for connecting rod
  passed_configs.csv  — one row per passing design: geometry + metrics + check columns
  failed_configs.csv  — one row per failing design: geometry + metrics + check columns

Usage
-----
  python scripts/generate_dataset.py

  python scripts/generate_dataset.py \\
      --config  configs/generate/baseline.yaml \\
      --seed    42 \\
      --out-dir data/preview
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mech390.config import load_config, get_baseline_config
from mech390.datagen.generate import generate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_dataset")

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "preview"

# Ordered list of (attribute_name, filename) pairs
_OUTPUTS = [
    ("kinematics_df", "kinematics.csv"),
    ("dynamics_df",   "dynamics.csv"),
    ("stresses_df",   "stresses.csv"),
    ("fatigue_df",    "fatigue.csv"),
    ("buckling_df",   "buckling.csv"),
    ("passed_df",     "passed_configs.csv"),
    ("failed_df",     "failed_configs.csv"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_dataset",
        description=(
            "Run the full MECH390 pipeline and write seven CSV files "
            "(kinematics, dynamics, stresses, fatigue, buckling, "
            "passed_configs, failed_configs)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", metavar="PATH", default=None,
        help="Path to generation config YAML. Defaults to configs/generate/baseline.yaml.",
    )
    parser.add_argument(
        "--seed", metavar="INT", type=int, default=None,
        help="Random seed (overrides config.random_seed).",
    )
    parser.add_argument(
        "--out-dir", metavar="PATH", default=str(DEFAULT_OUT_DIR),
        help="Directory where all CSV files are written.",
    )
    parser.add_argument(
        "--log-level", metavar="LEVEL", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def run(config_path, seed, out_dir: Path) -> None:
    if seed is not None:
        _seed_everything(seed)

    # ---- Load config ---------------------------------------------------------
    if config_path is not None:
        logger.info("Loading config from: %s", config_path)
        config = load_config(config_path)
    else:
        logger.info("Loading default baseline config.")
        config = get_baseline_config()

    if seed is not None:
        config["random_seed"] = seed

    # ---- Run pipeline --------------------------------------------------------
    logger.info("Starting full pipeline …")
    t0 = time.perf_counter()
    result = generate_dataset(config, seed=seed)
    elapsed = time.perf_counter() - t0

    # ---- Write CSVs ----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for attr, filename in _OUTPUTS:
        df = getattr(result, attr)
        out_path = out_dir / filename
        df.to_csv(out_path, index=False)
        written.append((filename, len(df), out_path))
        logger.info("Wrote %s — %d rows → %s", filename, len(df), out_path)

    # ---- Summary -------------------------------------------------------------
    s = result.summary
    print("\n" + "=" * 68)
    print("  Dataset Generation — Summary")
    print("=" * 68)
    print(f"  Stage-1 valid 2D designs  : {s.get('n_stage1', '?'):>10,}")
    print(f"  Stage-2 3D candidates     : {s.get('n_stage2', '?'):>10,}")
    print(f"  Dropped (physics errors)  : {s.get('n_dropped', '?'):>10,}")
    print(f"  Duplicates removed        : {s.get('n_duplicates_removed', 0):>10,}")
    print(f"  Evaluated                 : {s.get('n_evaluated', '?'):>10,}")
    print(f"  Passed                    : {s.get('n_passed', '?'):>10,}")
    print(f"  Failed                    : {s.get('n_failed', '?'):>10,}")
    print(f"  Pass rate                 : {s.get('pass_rate', 0.0):>10.1%}")
    print(f"  Wall time                 : {elapsed:>10.2f} s")
    print()
    print("  Output files:")
    for filename, nrows, path in written:
        print(f"    {filename:<25}  {nrows:>6,} rows  →  {path}")
    print("=" * 68)

    if s.get('n_evaluated', 0) == 0:
        logger.error("No designs were evaluated. Check config bounds and physics setup.")
        sys.exit(1)


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)
    out_dir = Path(args.out_dir).resolve()

    try:
        run(config_path=args.config, seed=args.seed, out_dir=out_dir)
    except FileNotFoundError as exc:
        logger.error("Config file not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
