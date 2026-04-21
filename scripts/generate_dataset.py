"""
generate_dataset.py - full pipeline: Stage 1 -> Stage 2 -> physics -> pass/fail labeling.

Writes 7 CSVs to data/preview/. Accepts --log-level to control verbosity.
Usage: python scripts/generate_dataset.py
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mech390.config import get_baseline_config
from mech390.datagen.generate import generate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_dataset")

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "preview"

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
    parser = argparse.ArgumentParser(prog="generate_dataset")
    parser.add_argument(
        "--log-level", metavar="LEVEL", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run(out_dir: Path) -> None:
    config = get_baseline_config()

    logger.info("Starting full pipeline ...")
    t0 = time.perf_counter()
    result = generate_dataset(config, out_dir=out_dir)
    elapsed = time.perf_counter() - t0

    ### Summary
    s = result.summary
    logger.info("=" * 68)
    logger.info("  Dataset Generation - Summary")
    logger.info("=" * 68)
    logger.info("  Stage-1 valid 2D designs  : %10s", f"{s.get('n_stage1', '?'):,}" if isinstance(s.get('n_stage1'), int) else "?")
    logger.info("  Stage-2 3D candidates     : %10s", f"{s.get('n_stage2', '?'):,}" if isinstance(s.get('n_stage2'), int) else "?")
    logger.info("  Dropped (physics errors)  : %10s", f"{s.get('n_dropped', '?'):,}" if isinstance(s.get('n_dropped'), int) else "?")
    logger.info("  Duplicates removed        : %10s", f"{s.get('n_duplicates_removed', 0):,}")
    logger.info("  Evaluated                 : %10s", f"{s.get('n_evaluated', '?'):,}" if isinstance(s.get('n_evaluated'), int) else "?")
    logger.info("  Passed                    : %10s", f"{s.get('n_passed', '?'):,}" if isinstance(s.get('n_passed'), int) else "?")
    logger.info("  Failed                    : %10s", f"{s.get('n_failed', '?'):,}" if isinstance(s.get('n_failed'), int) else "?")
    logger.info("  Pass rate                 : %10.1f%%", s.get('pass_rate', 0.0) * 100)
    logger.info("  Wall time                 : %10.2f s", elapsed)
    logger.info("  Output files exported to  : %s", out_dir)
    logger.info("=" * 68)

    if s.get('n_evaluated', 0) == 0:
        logger.error("No designs were evaluated. Check config bounds and physics setup.")
        sys.exit(1)


def main() -> None:
    args = _build_parser().parse_args()
    logging.getLogger().setLevel(args.log_level)
    try:
        run(out_dir=DEFAULT_OUT_DIR)
    except FileNotFoundError as exc:
        logger.error("Config file not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
