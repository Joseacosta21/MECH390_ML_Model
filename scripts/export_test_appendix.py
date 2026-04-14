"""
Export test-set inputs + ground truth + model predictions for the report appendix.

Produces two files in the output directory:
  appendix_test_predictions.csv  — one row per test design, columns:
      [INPUT_FEATURES] | pass_fail (ground truth) | pass_fail_pred | pass_prob |
      [REGRESSION_TARGETS ground truth] | [REGRESSION_TARGETS predicted]
  appendix_test_predictions_rounded.csv — same, values rounded to 4 sig figs

Usage
-----
    .venv/bin/python scripts/export_test_appendix.py \
        --test     data/splits/test.csv \
        --ckpt     data/models/surrogate_best.pt \
        --scaler   data/models/scaler.pkl \
        --stats    data/models/target_stats.json \
        --out-dir  data/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make src/ importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mech390.ml import features as F
from mech390.ml.infer import SurrogatePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test-set appendix table")
    parser.add_argument("--test",    default="data/splits/test.csv")
    parser.add_argument("--ckpt",    default="data/models/surrogate_best.pt")
    parser.add_argument("--scaler",  default="data/models/scaler.pkl")
    parser.add_argument("--stats",   default="data/models/target_stats.json")
    parser.add_argument("--out-dir", default="data/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load test split
    # ------------------------------------------------------------------
    test_df = pd.read_csv(args.test)
    print(f"Loaded test set: {len(test_df)} rows")

    # Derive slenderness features (needed by predictor)
    test_df = F.derive_input_features(test_df)

    # ------------------------------------------------------------------
    # Run surrogate on all test rows
    # ------------------------------------------------------------------
    predictor = SurrogatePredictor(
        checkpoint=args.ckpt,
        scaler_path=args.scaler,
        stats_path=args.stats,
    )
    preds_df = predictor.predict(test_df)  # DataFrame with pass_prob, pass_fail_pred, + reg targets

    # ------------------------------------------------------------------
    # Build appendix table
    # ------------------------------------------------------------------
    # Ground-truth regression targets (may be missing for fail rows — keep NaN)
    gt_reg_cols = []
    for t in F.REGRESSION_TARGETS:
        col = t + "_gt"
        test_df[col] = test_df[t] if t in test_df.columns else float("nan")
        gt_reg_cols.append(col)

    # Predicted regression targets
    pred_reg_cols = [t + "_pred" for t in F.REGRESSION_TARGETS]
    preds_renamed = preds_df[F.REGRESSION_TARGETS].copy()
    preds_renamed.columns = pred_reg_cols

    appendix = pd.concat(
        [
            test_df[F.INPUT_FEATURES].reset_index(drop=True),
            test_df[[F.CLASSIFICATION_TARGET]].reset_index(drop=True),
            preds_df[["pass_fail_pred", "pass_prob"]].reset_index(drop=True),
            test_df[gt_reg_cols].reset_index(drop=True),
            preds_renamed.reset_index(drop=True),
        ],
        axis=1,
    )

    # ------------------------------------------------------------------
    # Write full-precision CSV
    # ------------------------------------------------------------------
    full_path = out_dir / "appendix_test_predictions.csv"
    appendix.to_csv(full_path, index=False)
    print(f"Written: {full_path}  ({len(appendix)} rows × {len(appendix.columns)} cols)")

    # ------------------------------------------------------------------
    # Write rounded CSV (4 significant figures — cleaner for print table)
    # ------------------------------------------------------------------
    rounded = appendix.copy()
    float_cols = rounded.select_dtypes(include="float").columns
    rounded[float_cols] = rounded[float_cols].apply(
        lambda col: col.map(lambda v: float(f"{v:.4g}") if pd.notna(v) else v)
    )
    rounded_path = out_dir / "appendix_test_predictions_rounded.csv"
    rounded.to_csv(rounded_path, index=False)
    print(f"Written: {rounded_path}  (rounded to 4 sig figs)")

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    n = len(appendix)
    n_pass_gt   = int(appendix[F.CLASSIFICATION_TARGET].sum())
    n_pass_pred = int(appendix["pass_fail_pred"].sum())
    n_correct   = int((appendix[F.CLASSIFICATION_TARGET] == appendix["pass_fail_pred"]).sum())
    print(f"\nTest set summary")
    print(f"  Rows          : {n}")
    print(f"  GT pass       : {n_pass_gt}  ({100*n_pass_gt/n:.1f}%)")
    print(f"  Pred pass     : {n_pass_pred}  ({100*n_pass_pred/n:.1f}%)")
    print(f"  Accuracy      : {100*n_correct/n:.2f}%")


if __name__ == "__main__":
    main()
