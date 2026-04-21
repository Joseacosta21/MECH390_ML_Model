"""
run_pipeline.py - end-to-end pipeline: data generation -> surrogate training -> design optimization.

Usage: python scripts/run_pipeline.py [--skip-datagen] [--skip-training] [--top N]
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_pipeline")
    p.add_argument("--skip-datagen", action="store_true",
        help="Skip Step 1 (data generation). Use existing CSVs.")
    p.add_argument("--skip-training", action="store_true",
        help="Skip Step 2 (surrogate training). Use existing checkpoint.")
    p.add_argument("--top", "-n", type=int, default=3, metavar="N",
        help="Number of top candidates to validate and report.")
    return p


def _run_step(label: str, cmd: list[str]) -> None:
    logger.info("STEP: %s", label)
    logger.info("CMD:  %s", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        logger.error("Step '%s' failed with exit code %d (%.1fs elapsed).",
                     label, result.returncode, elapsed)
        sys.exit(result.returncode)
    logger.info("Step '%s' completed in %.1fs.", label, elapsed)


def main() -> None:
    args   = _build_parser().parse_args()
    python = sys.executable

    t_total = time.perf_counter()

    if args.skip_datagen:
        logger.info("Skipping Step 1 (--skip-datagen).")
    else:
        _run_step("Data generation", [python, "scripts/generate_dataset.py"])

    if args.skip_training:
        logger.info("Skipping Step 2 (--skip-training).")
    else:
        _run_step("Surrogate training", [python, "scripts/train_model.py"])

    _run_step("Design optimization + manufacturing report",
              [python, "scripts/optimize_design.py", "--top", str(args.top)])

    total = time.perf_counter() - t_total
    logger.info("Pipeline complete in %.1fs.", total)


if __name__ == "__main__":
    main()
