"""
preview_forces.py - Stage 1 -> Stage 2 -> dynamics sweep, one row per (design, crank angle).

Output: data/preview/forces_sweep.csv
Usage:  python scripts/preview_forces.py
"""

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mech390.config import get_baseline_config
from mech390.datagen import stage1_kinematic, stage2_embodiment
from mech390.physics import dynamics, mass_properties as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preview_forces")

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "preview"
OUTPUT_FILENAME = "forces_sweep.csv"

CSV_COLUMNS = [
    "design_index",
    "r", "l", "e",
    "theta_deg", "theta_rad",
    "F_Ax", "F_Ay",
    "F_Bx", "F_By",
    "F_Cx", "F_Cy",
    "N", "F_f",
    "tau_A",
]

THETA_STEP_DEG = 15
THETAS_DEG = list(range(0, 360, THETA_STEP_DEG))


def run():
    config = get_baseline_config()
    out_dir = DEFAULT_OUT_DIR
    max_2d = None

    operating = config.get("operating", {})
    rpm     = float(operating.get("RPM", 30))
    omega   = rpm * 2.0 * np.pi / 60.0
    mu      = float(operating.get("mu", 0.0))
    g       = float(operating.get("g", 9.81))

    logger.info("omega = %.4f rad/s  (RPM=%.1f)  mu=%.3f", omega, rpm, mu)

    ### Stage 1
    logger.info("Stage 1: kinematic synthesis ...")
    t0 = time.perf_counter()
    valid_2d = list(stage1_kinematic.iter_valid_2d_mechanisms(config))
    n_stage1 = len(valid_2d)
    logger.info("Stage 1 done in %.2f s - %d valid 2D designs.", time.perf_counter() - t0, n_stage1)

    if n_stage1 == 0:
        logger.error("No valid 2D designs. Check config bounds.")
        sys.exit(1)

    designs_for_stage2 = valid_2d[:max_2d] if max_2d is not None else valid_2d
    n_forwarded = len(designs_for_stage2)

    ### Stage 2 + dynamics sweep
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILENAME

    n_designs   = 0
    n_rows      = 0
    n_dyn_ok    = 0
    n_dyn_err   = 0

    with out_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for design_3d in stage2_embodiment.iter_expand_to_3d(designs_for_stage2, config):
            n_designs += 1

            try:
                mass_props = mp.compute_design_mass_properties(design_3d, config)
            except Exception as exc:
                logger.warning("Mass props failed for design #%d: %s", n_designs, exc)
                continue

            r = design_3d["r"]
            l = design_3d["l"]
            e = design_3d["e"]
            mass_crank  = mass_props["mass_crank"]
            mass_rod    = mass_props["mass_rod"]
            mass_slider = mass_props["mass_slider"]
            I_crank     = mass_props["I_mass_crank_cg_z"]
            I_rod       = mass_props["I_mass_rod_cg_z"]

            for theta_deg in THETAS_DEG:
                theta_rad = np.deg2rad(theta_deg)
                try:
                    forces = dynamics.joint_reaction_forces(
                        theta_rad, omega, r, l, e,
                        mass_crank, mass_rod, mass_slider,
                        I_crank=I_crank,
                        I_rod=I_rod,
                        mu=mu,
                        g=g,
                        alpha_r=0.0,
                    )
                    row = {
                        "design_index": n_designs,
                        "r": r, "l": l, "e": e,
                        "theta_deg": theta_deg,
                        "theta_rad": round(theta_rad, 6),
                        "F_Ax": forces["F_A"][0], "F_Ay": forces["F_A"][1],
                        "F_Bx": forces["F_B"][0], "F_By": forces["F_B"][1],
                        "F_Cx": forces["F_C"][0], "F_Cy": forces["F_C"][1],
                        "N":    forces["N"],
                        "F_f":  forces["F_f"],
                        "tau_A": forces["tau_A"],
                    }
                    writer.writerow(row)
                    n_rows += 1
                    n_dyn_ok += 1
                except Exception as exc:
                    logger.debug(
                        "Dynamics failed - design #%d theta=%d deg: %s",
                        n_designs, theta_deg, exc
                    )
                    n_dyn_err += 1

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 64)
    print("  Forces Preview - Summary")
    print("=" * 64)
    print(f"  Stage-1 valid designs  : {n_stage1:>10,}")
    print(f"  Forwarded to Stage 2   : {n_forwarded:>10,}")
    print(f"  3D designs processed   : {n_designs:>10,}")
    print(f"  Dynamics rows OK       : {n_dyn_ok:>10,}")
    print(f"  Dynamics rows failed   : {n_dyn_err:>10,}")
    print(f"  Total CSV rows         : {n_rows:>10,}")
    print(f"  Elapsed time           : {elapsed:>10.2f} s")
    print(f"  Output file            : {out_path}")
    print("=" * 64)

    if n_rows == 0:
        logger.error("No rows written - check config and dynamics setup.")
        sys.exit(1)

    ### Quick stats
    import pandas as pd
    df = pd.read_csv(out_path)
    print("\nDescriptive statistics (force columns):\n")
    force_cols = ["F_Ax", "F_Ay", "F_Bx", "F_By", "F_Cx", "F_Cy", "N", "F_f", "tau_A"]
    print(df[force_cols].describe().to_string())
    print()


def main():
    try:
        run()
    except FileNotFoundError as exc:
        logger.error("Config file not found: %s", exc)
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
