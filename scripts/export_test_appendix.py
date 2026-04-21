"""
export_test_appendix.py - exports test-set predictions for the report appendix.

Writes two files to data/:
  appendix_test_predictions.csv
  appendix_test_predictions_rounded.csv  (4 sig figs)

Usage: python scripts/export_test_appendix.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mech390.ml import features as F
from mech390.ml.infer import SurrogatePredictor

_ROOT    = Path(__file__).resolve().parent.parent
_TEST    = _ROOT / "data" / "splits" / "test.csv"
_CKPT    = _ROOT / "data" / "models" / "surrogate_best.pt"
_SCALER  = _ROOT / "data" / "models" / "scaler.pkl"
_STATS   = _ROOT / "data" / "models" / "target_stats.json"
_OUT_DIR = _ROOT / "data"


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(_TEST)
    print(f"Loaded test set: {len(test_df)} rows")

    # add derived input features
    test_df = F.derive_input_features(test_df)

    predictor = SurrogatePredictor(
        checkpoint=str(_CKPT),
        scaler_path=str(_SCALER),
        stats_path=str(_STATS),
    )
    preds_df = predictor.predict(test_df)  # gives pass_fail_pred, pass_prob, and regression targets

    # ground truth regression targets (NaN for fail rows)
    gt_reg_cols = []
    for t in F.REGRESSION_TARGETS:
        col = t + "_gt"
        test_df[col] = test_df[t] if t in test_df.columns else float("nan")
        gt_reg_cols.append(col)

    # predicted regression targets
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

    full_path = _OUT_DIR / "appendix_test_predictions.csv"
    appendix.to_csv(full_path, index=False)
    print(f"Written: {full_path}  ({len(appendix)} rows x {len(appendix.columns)} cols)")

    # round to 4 sig figs for a cleaner print table
    rounded = appendix.copy()
    float_cols = rounded.select_dtypes(include="float").columns
    rounded[float_cols] = rounded[float_cols].apply(
        lambda col: col.map(lambda v: float(f"{v:.4g}") if pd.notna(v) else v)
    )
    rounded_path = _OUT_DIR / "appendix_test_predictions_rounded.csv"
    rounded.to_csv(rounded_path, index=False)
    print(f"Written: {rounded_path}  (rounded to 4 sig figs)")

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
