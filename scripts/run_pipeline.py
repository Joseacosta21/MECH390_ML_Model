"""
run_pipeline.py
---------------
End-to-end pipeline runner: data generation → surrogate training → design optimization.

Runs the three steps in sequence. If any step fails, the pipeline stops immediately
and prints a clear error. Each step's output becomes the next step's input
automatically — no manual path wiring needed.

Usage (all defaults — recommended first run)
--------------------------------------------
    python scripts/run_pipeline.py

Usage (explicit options)
------------------------
    python scripts/run_pipeline.py \\
        --generate-config configs/generate/baseline.yaml \\
        --train-config    configs/train/surrogate.yaml \\
        --optimize-config configs/optimize/search.yaml \\
        --seed            42 \\
        --out-dir         data/preview \\
        --log-level       INFO

Skip steps (e.g. skip datagen if data already exists)
------------------------------------------------------
    python scripts/run_pipeline.py --skip-datagen
    python scripts/run_pipeline.py --skip-datagen --skip-training

Steps
-----
1. generate_dataset.py  — physics simulation → 7 CSVs in --out-dir
2. train_model.py       — Optuna sweep → surrogate checkpoint in data/models/
3. optimize_design.py   — differential_evolution → top-N candidate geometries
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
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description=(
            "End-to-end pipeline: data generation → surrogate training → "
            "design optimization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--generate-config", "-g",
        default="configs/generate/baseline.yaml",
        metavar="PATH",
        help="Generation config YAML (geometry ranges, sampling settings).",
    )
    p.add_argument(
        "--train-config", "-t",
        default="configs/train/surrogate.yaml",
        metavar="PATH",
        help="Surrogate training config YAML (architecture search space, epochs).",
    )
    p.add_argument(
        "--optimize-config", "-o",
        default="configs/optimize/search.yaml",
        metavar="PATH",
        help="Optimizer config YAML (objective weights, constraint thresholds).",
    )
    p.add_argument(
        "--out-dir", "-d",
        default="data/preview",
        metavar="PATH",
        help="Output directory for generated CSVs (Step 1).",
    )
    p.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        metavar="INT",
        help="Global random seed passed to all three steps. Overrides config seeds.",
    )
    p.add_argument(
        "--skip-datagen",
        action="store_true",
        help="Skip Step 1 (data generation). Use existing CSVs in --out-dir.",
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Step 2 (surrogate training). Use existing checkpoint in data/models/.",
    )
    p.add_argument(
        "--candidates-json",
        default="data/results/candidates.json",
        metavar="PATH",
        help="Where optimize_design.py writes its top-N candidate JSON (read by Step 4).",
    )
    p.add_argument(
        "--top", "-n",
        type=int,
        default=3,
        metavar="N",
        help="Number of top candidates to validate and report in Step 4.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _run_step(label: str, cmd: list[str]) -> None:
    """
    Run a subprocess step. Streams output in real time.
    Raises SystemExit on non-zero return code.
    """
    logger.info("=" * 60)
    logger.info("STEP: %s", label)
    logger.info("CMD:  %s", " ".join(cmd))
    logger.info("=" * 60)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        logger.error(
            "Step '%s' failed with exit code %d (%.1fs elapsed).",
            label, result.returncode, elapsed,
        )
        sys.exit(result.returncode)

    logger.info("Step '%s' completed in %.1fs.", label, elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Apply log level
    logging.getLogger().setLevel(args.log_level)

    python = sys.executable  # same interpreter that launched this script

    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Data generation
    # ------------------------------------------------------------------
    if args.skip_datagen:
        logger.info("Skipping Step 1 (--skip-datagen). Using existing data in '%s'.", args.out_dir)
    else:
        cmd = [python, "scripts/generate_dataset.py",
               "--config",  args.generate_config,
               "--out-dir", args.out_dir,
               "--log-level", args.log_level]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        _run_step("Data generation", cmd)

    # ------------------------------------------------------------------
    # Step 2 — Surrogate training
    # ------------------------------------------------------------------
    if args.skip_training:
        logger.info("Skipping Step 2 (--skip-training). Using existing checkpoint.")
    else:
        cmd = [python, "scripts/train_model.py",
               "--config",    args.train_config,
               "--log-level", args.log_level]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        _run_step("Surrogate training", cmd)

    # ------------------------------------------------------------------
    # Step 3 — Design optimization + pre-manufacturing report
    # optimize_design.py runs physics validation inline and prints the
    # unified report — no separate summarize_results.py step needed.
    # ------------------------------------------------------------------
    cmd = [python, "scripts/optimize_design.py",
           "--generate-config", args.generate_config,
           "--optimize-config", args.optimize_config,
           "--out-json",        args.candidates_json,
           "--top",             str(args.top),
           "--log-level",       args.log_level]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    _run_step("Design optimization + manufacturing report", cmd)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    total = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs.", total)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
