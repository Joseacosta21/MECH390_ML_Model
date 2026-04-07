# ML Pipeline — Step-by-Step Guide

**Scope:** Full pipeline from dataset preparation through surrogate training, optimizer configuration, result interpretation, and mandatory physics verification.  
**Audience:** Developer, teammates, and AI agents.  
**Project constraint:** The dataset is professor-limited to 200 rows (40 × 5 variants). This is below the ~1 000-row threshold for robust ML. All metrics should be treated as directional — they indicate the right design region, not a production-grade prediction. Do not interpret a good F1 score on 30 val rows as certification.

---

## Prerequisites

Before starting, confirm:

| Requirement | How to check |
|---|---|
| Pipeline has been run | `data/preview/passed_configs.csv` and `data/preview/failed_configs.csv` exist |
| Both files are non-empty | `passed_configs.csv` ≥ 1 row, `failed_configs.csv` ≥ 1 row |
| Virtual environment active | `.venv/bin/python --version` returns Python 3.x |
| PyTorch installed | `.venv/bin/python -c "import torch; print(torch.__version__)"` |
| Optuna installed | `.venv/bin/python -c "import optuna; print(optuna.__version__)"` |

If `data/preview/` is missing or empty, run the data generation pipeline first:

```bash
.venv/bin/python scripts/generate_dataset.py \
    --config  configs/generate/baseline.yaml \
    --seed    42 \
    --out-dir data/preview
```

---

## Step 1 — Prepare the dataset

### 1a. Fix the data paths in `configs/train/surrogate.yaml`

`surrogate.yaml` currently points to `data/raw/`, which does not exist. Update it to match the actual output directory:

Open [configs/train/surrogate.yaml](configs/train/surrogate.yaml) and change:

```yaml
# BEFORE (wrong — data/raw/ does not exist)
data:
  csv_pass: data/raw/passed_configs.csv
  csv_fail: data/raw/failed_configs.csv

# AFTER (correct — points to the generated preview data)
data:
  csv_pass: data/preview/passed_configs.csv
  csv_fail: data/preview/failed_configs.csv
```

If you ever move to a production run (e.g., `data/runs/v1/`), update these paths to match.

### 1b. Pre-process: clip unphysical stress values

A known open bug ([stresses.py:218](src/mech390/physics/stresses.py#L218)) produces ~TPa stress values in ~7–27 rows of `failed_configs.csv` when a pin diameter nearly equals the link width. These rows are correctly labeled `pass_fail = 0`, but the extreme magnitudes will distort the regression heads if left raw.

**Before training, clip `sigma_max` and `tau_max` in the failed_configs CSV:**

```bash
.venv/bin/python - <<'EOF'
import pandas as pd

S_ut = 483e6  # Pa — ultimate tensile strength, Al 2024-T3
clip = 10 * S_ut  # 4.83 GPa — 10× S_ut as the ceiling

for fname in ['data/preview/passed_configs.csv', 'data/preview/failed_configs.csv']:
    df = pd.read_csv(fname)
    for col in ['sigma_max', 'tau_max']:
        if col in df.columns:
            df[col] = df[col].clip(upper=clip)
    df.to_csv(fname, index=False)
    print(f"Clipped {fname}")
EOF
```

This is a one-time step. Re-run it whenever you regenerate the dataset.

---

## Step 2 — Configure training (`configs/train/surrogate.yaml`)

Read this file before running. Key parameters to understand and possibly change:

### Data split

```yaml
data:
  train_frac:  0.70   # 140 rows train
  val_frac:    0.15   # 30 rows val
  # test_frac:  0.15  # 30 rows test (inferred)
  random_seed: 42
```

With 200 rows, the val and test sets each have ~30 rows. Metrics computed on 30 rows have high variance — a single misclassified sample changes F1 by ~3 percentage points.

### Features (do not change without also changing `src/mech390/ml/features.py`)

```yaml
features:
  inputs:                     # 10 independent design variables fed to the NN
    - r, l, e                 # kinematic geometry (meters)
    - width_r, thickness_r    # crank cross-section (meters)
    - width_l, thickness_l    # rod cross-section (meters)
    - pin_diameter_A/B/C      # pin diameters (meters)
  classification_target: pass_fail
  regression_targets:         # 7 regression heads
    - total_mass              # kg
    - volume_envelope         # m³
    - tau_A_max               # N·m (peak motor torque)
    - E_rev                   # J/rev (energy per revolution)
    - min_n_static            # derived: min(n_static_rod, crank, pin)
    - utilization             # max(σ/σ_limit, τ/τ_limit) — 0 to 1+
    - n_buck                  # Euler buckling safety factor
```

These 10 inputs + 8 targets (1 clf + 7 reg) are fixed by the feature contract in [src/mech390/ml/features.py](src/mech390/ml/features.py). Do not add or remove columns here without updating that file and re-generating the dataset.

### Architecture search space

```yaml
model:
  hidden_sizes_options:       # Optuna picks one per trial
    - [64, 64]                # small — fast, may underfit 200 rows
    - [128, 128]
    - [256, 128]
    - [256, 128, 64]
    - [512, 256, 128]         # large — may overfit 200 rows
  dropout_range: [0.0, 0.3]   # regularization — important given small dataset
  use_batch_norm: true        # fixed on — stabilizes training
```

For 200 rows, `[64, 64]` or `[128, 128]` will likely win. Larger architectures overfit.

### Training budget

```yaml
training:
  max_epochs: 300
  patience:   25    # stop if val loss does not improve for 25 consecutive epochs

optuna:
  n_trials:  50     # 50 architecture/hyperparameter combinations tried
```

**Estimated wall time:** ~5–15 minutes on CPU (50 trials × avg ~100 epochs each).  
A smoke test with `n_trials: 3` and `max_epochs: 20` runs in ~30 seconds and can be used to verify the pipeline works before committing to the full run.

### Loss weights

```yaml
loss_weights:
  bce: 1.0    # weight on classification (pass/fail) loss
  mse: 0.5    # weight on regression (7 targets) loss
```

The primary metric is `val_f1` (classification). If regression predictions are the priority (e.g., for the optimizer's mass/torque estimates), increase `mse` to `1.0` and re-train.

---

## Step 3 — Run training

```bash
.venv/bin/python scripts/train_model.py \
    --config configs/train/surrogate.yaml
```

**What you will see during the run:**
- One log line per trial showing trial number and val_f1
- A progress bar (Optuna tqdm)
- Final line: `Training complete. Best val F1: X.XXXX` and the best architecture

**What gets saved to `data/models/` after the run:**

| File | Contents |
|---|---|
| `surrogate_best.pt` | PyTorch checkpoint: model weights + hparams + val_f1 |
| `scaler.pkl` | `StandardScaler` fitted on the training split (required at inference) |
| `target_stats.json` | min/max of each regression target from train set (used by optimizer) |

All three files must be present before running the optimizer.

---

## Step 4 — Evaluate training quality

After training, inspect the best val_f1 printed to console.

### Acceptance criteria (guidelines, not hard gates)

Given the 200-row dataset, these thresholds are appropriate directional targets:

| Metric | Target | Rationale |
|---|---|---|
| `val_f1` (classification) | ≥ 0.80 | On 30 val rows, ≥ 0.80 F1 means ~24/30 correctly classified. Below 0.70 is too noisy to trust. |
| Regression R² | ≥ 0.70 for `total_mass`, `volume_envelope`, `tau_A_max` | Not directly reported in training output — evaluate manually (see below). Below 0.50 means the optimizer's objective estimates are unreliable. |

> These are **not** production thresholds. A full dataset (1 000+ rows) would require val F1 > 0.90 and R² > 0.85. With 200 rows, treat results as directional guidance for choosing a design region, then verify with full physics.

### Manually evaluate regression quality (optional but recommended)

```bash
.venv/bin/python - <<'EOF'
import torch, pickle, json
import pandas as pd
import numpy as np
from pathlib import Path
import sys; sys.path.insert(0, 'src')

from mech390.ml.models import build_model_from_hparams, load_checkpoint
from mech390.ml import features as F

# Load artifacts
ckpt   = load_checkpoint('data/models/surrogate_best.pt', device='cpu')
model  = build_model_from_hparams(ckpt['hparams'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
scaler = F.load_scaler('data/models/scaler.pkl')

# Load test split
df = F.load_dataset('data/preview/passed_configs.csv', 'data/preview/failed_configs.csv')
_, _, test_df = F.split_dataset(df, train_frac=0.70, val_frac=0.15, random_seed=42)
X, y_clf, y_reg = F.get_arrays(test_df, scaler)

import torch
with torch.no_grad():
    logit, pred_reg = model(torch.from_numpy(X))

pred_reg = pred_reg.numpy()
for i, name in enumerate(F.REGRESSION_TARGETS):
    ss_res = np.sum((y_reg[:, i] - pred_reg[:, i])**2)
    ss_tot = np.sum((y_reg[:, i] - y_reg[:, i].mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    print(f"  {name:20s}  R² = {r2:.3f}")
EOF
```

### If training quality is insufficient

| Symptom | Likely cause | Action |
|---|---|---|
| val_f1 < 0.70 | Dataset too small or architecture overfitting | Reduce architecture to `[64, 64]`; increase dropout to 0.3; regenerate more data |
| val_f1 oscillates across trials | High variance from 30-sample val set | Normal with 200 rows; accept the best trial result |
| All regression R² < 0.50 | Regression targets have too wide a range | Check for unclipped stress outliers (Step 1b); increase `mse` loss weight to 1.0 |
| Training crashes on `KeyError: min_n_static` | Dataset generated before `min_n_static` derived column was added | Re-run data generation |

---

## Step 5 — Configure the optimizer (`configs/optimize/search.yaml`)

This is the file that controls **what you are optimizing for**. Edit it to match your design priorities.

### Full annotated walkthrough

```yaml
objectives:
  total_mass:
    weight:    0.30       # 30% of score — mass is the primary concern (3D printing cost)
    direction: minimize   # lighter is better

  volume_envelope:
    weight:    0.25       # 25% of score — footprint is second priority
    direction: minimize   # smaller bounding box fits better on the test bench

  tau_A_max:
    weight:    0.15       # 15% of score — worst-case instantaneous motor load
    direction: minimize   # lower peak torque → smaller/cheaper motor required

  E_rev:
    weight:    0.15       # 15% of score — average energy efficiency over a full cycle
    direction: minimize   # lower energy/rev → more efficient motor selection

  min_n_static:
    weight:    0.15       # 15% of score — structural safety margin
    direction: maximize   # higher static FOS → more robustness to material defects
```

> `tau_A_max` and `E_rev` are complementary motor metrics: `tau_A_max` captures the worst-case instantaneous torque spike (determines motor sizing); `E_rev` captures the average energy consumed per cycle (determines running cost and motor efficiency class). Keeping both at equal weight (0.15 each) balances peak and average motor performance.

**Rules for editing objectives:**
- Weights must sum to 1.0. The optimizer normalises each objective to [0, 1] from the training-set range, then applies the weights.
- `direction: minimize` → the score contribution is `weight × (1 − normalized_value)`, so lower raw values score higher.
- `direction: maximize` → the score contribution is `weight × normalized_value`, so higher raw values score higher.
- Available objective names: `total_mass`, `volume_envelope`, `tau_A_max`, `E_rev`, `min_n_static`, `utilization`, `n_buck`. These map exactly to the regression targets.

**Example: prioritise motor power over mass** (e.g., if motor cost is the binding constraint):

```yaml
objectives:
  tau_A_max:
    weight:    0.50
    direction: minimize
  total_mass:
    weight:    0.25
    direction: minimize
  volume_envelope:
    weight:    0.15
    direction: minimize
  min_n_static:
    weight:    0.10
    direction: maximize
```

### Constraint

```yaml
constraints:
  pass_fail_prob_min: 0.90   # reject any geometry where predicted P(pass) < 90%
```

This is a hard gate applied as a penalty in the score function. Geometries with `pass_prob < 0.90` are penalized out of the top results. Lower this to `0.70` if the optimizer returns too few candidates; raise to `0.95` to be more conservative.

### Optimizer settings

```yaml
optimizer:
  method:        differential_evolution  # global optimizer — do not change
  n_top_results: 10       # number of distinct top geometries to report
  seed:          42       # for reproducibility
  maxiter:       500      # max iterations (each iteration = popsize × n_vars evaluations)
  popsize:       15       # population = 15 × 10 dims = 150 candidates per generation
```

Increase `maxiter` to 1000 if the optimizer reports `converged: False`. Increase `popsize` to 20 if results cluster too tightly. Both slow the run proportionally.

The geometry bounds (min/max for all 10 design variables) are **read automatically from `configs/generate/baseline.yaml`** — do not duplicate them here.

---

## Step 6 — Run the optimizer

```bash
.venv/bin/python scripts/optimize_design.py \
    --generate-config configs/generate/baseline.yaml \
    --optimize-config configs/optimize/search.yaml \
    --model           data/models/surrogate_best.pt
```

The `--scaler` and `--stats` arguments default to `data/models/scaler.pkl` and `data/models/target_stats.json`. Only specify them explicitly if you saved to a non-default location.

**Estimated wall time:** 10–60 seconds on CPU (500 iterations × 150 candidates = 75 000 surrogate evaluations; each is a single forward pass).

---

## Step 7 — Interpret the output

The optimizer prints a ranked table of top-N geometries:

```
────────────────────────────────────────────────────────────────────────
  TOP-10 OPTIMISED GEOMETRIES
────────────────────────────────────────────────────────────────────────

  Rank 1  |  score=0.8234  |  pass_prob=97.3%
  Geometry:
    r                   :   62.000 mm
    l                   :  155.000 mm
    e                   :   18.000 mm
    width_r             :    4.000 mm
    thickness_r         :    3.000 mm
    width_l             :    5.000 mm
    thickness_l         :    4.000 mm
    pin_diameter_A      :    2.000 mm
    pin_diameter_B      :    1.500 mm
    pin_diameter_C      :    2.000 mm
  Predicted performance:
    total_mass          :   42.3 g
    volume_envelope     : 1234.56 cm³
    tau_A_max           :  0.3512 N·m
    E_rev               :  0.0721 J/rev
    min_n_static        :   8.45
    utilization         :  0.312
    n_buck              :  12.30
```

**How to read this:**
- `score` — weighted objective score, higher is better (max theoretical = 1.0 if all objectives simultaneously minimized/maximized). Values above 0.70 are good; above 0.85 is excellent.
- `pass_prob` — predicted probability the design passes all checks. Must be ≥ 90% (the configured constraint). Values ≥ 95% are preferred for physics verification.
- Geometry values are the 10 design variables in millimeters.
- Predicted performance values are in SI base units (g, cm³, N·m, J/rev are converted for display).

**What to do with the results:**
1. Select the Rank 1 candidate (or whichever best fits your design trade-offs).
2. Note all 10 geometry values.
3. Proceed to Step 8 — **do not go to 3D printing without physics verification**.

---

## Step 8 — Physics verification (mandatory)

The surrogate is an approximation trained on 200 data points. Even a high `pass_prob` is not a guarantee. Before committing to a 3D-printed prototype, verify the top candidate through the exact physics pipeline.

### Option A: run a single-design verification (recommended)

Create a minimal override YAML that pins the optimizer's geometry and runs the full pipeline:

```bash
# 1. Edit configs/generate/baseline.yaml to fix all 10 geometry values.
#    Set min == max for each variable so Stage 1 and Stage 2 produce exactly one design.
#    (Or create a separate verification config — do not permanently modify baseline.yaml.)

# 2. Run the full pipeline with seed 42 and n_samples=1
.venv/bin/python scripts/generate_dataset.py \
    --config  configs/generate/baseline.yaml \
    --seed    42 \
    --out-dir data/preview/verification

# 3. Check whether the design appears in passed_configs.csv
python - <<'EOF'
import pandas as pd
df = pd.read_csv('data/preview/verification/passed_configs.csv')
print(f"Passed: {len(df)} design(s)")
if len(df):
    cols = ['r','l','e','n_static_rod','n_static_crank','n_static_pin','n_buck',
            'n_fatigue_rod','n_fatigue_crank','pass_fail','total_mass','tau_A_max']
    print(df[cols].to_string(index=False))
EOF
```

### Verification checklist

When the top candidate is run through the physics pipeline, confirm:

| Check | Column | Minimum | Meaning |
|---|---|---|---|
| Static safety — rod | `n_static_rod` | ≥ 1.0 | Rod does not yield under peak load |
| Static safety — crank | `n_static_crank` | ≥ 1.0 | Crank does not yield |
| Static safety — pin | `n_static_pin` | ≥ 1.0 | Pins do not yield |
| Fatigue — rod | `n_fatigue_rod` | ≥ 1.0 | Rod survives design life |
| Fatigue — crank | `n_fatigue_crank` | ≥ 1.0 | Crank survives design life |
| Fatigue — pin | `n_fatigue_pin` | ≥ 1.0 | Pins survive design life |
| Miner's damage — rod | `D_miner_rod` | < 1.0 | Cumulative fatigue damage below threshold |
| Buckling | `n_buck` | ≥ 3.0 | Rod does not buckle under max compressive load |
| Utilization | `utilization` | ≤ 1.0 | Peak stress within allowable limit |
| Overall | `pass_fail` | 1 | Design passes all criteria simultaneously |

### If the candidate fails physics verification

The surrogate made an incorrect prediction. This is expected given the small dataset. Actions:

1. Try Rank 2 or Rank 3 from the optimizer output.
2. Lower `pass_fail_prob_min` in `search.yaml` to 0.80 and re-run to get more candidates.
3. If all top-N candidates fail: the surrogate is not reliable enough — see Step 9.

---

## Step 9 — Iterate and improve

### Improving prediction quality

| Problem | Action |
|---|---|
| All top-N candidates fail physics | Regenerate dataset with more aggressive geometry sampling; re-train |
| val F1 < 0.70 across all trials | Small dataset is the root cause; accept limitation or negotiate more samples |
| Optimizer keeps returning the same geometry | Increase `n_top_results` to 20; increase `popsize` to 20 |
| Regression R² < 0.50 for tau_A_max | Check stress clipping (Step 1b); increase `mse` loss weight; re-train |

### Changing design priorities

To shift toward a different trade-off (e.g., favour lower motor torque over mass):
1. Edit `configs/optimize/search.yaml` — adjust weights and/or add/remove objectives.
2. Re-run the optimizer (Step 6). No re-training needed — the model is unchanged.
3. Verify the new top candidate (Step 8).

### Expanding the dataset

If the surrogate is consistently unreliable, the only real fix is more data:

```bash
# Increase n_samples in baseline.yaml, then re-run:
.venv/bin/python scripts/generate_dataset.py \
    --config  configs/generate/baseline.yaml \
    --seed    42 \
    --out-dir data/preview

# Update paths in surrogate.yaml, re-run Steps 1–8.
```

---

## Quick-reference command table

| Task | Command |
|---|---|
| Generate dataset | `.venv/bin/python scripts/generate_dataset.py --config configs/generate/baseline.yaml --seed 42 --out-dir data/preview` |
| Smoke test training (fast) | Edit `surrogate.yaml`: set `n_trials: 3`, `max_epochs: 20`; then run train command below |
| Full training | `.venv/bin/python scripts/train_model.py --config configs/train/surrogate.yaml` |
| Run optimizer | `.venv/bin/python scripts/optimize_design.py --generate-config configs/generate/baseline.yaml --optimize-config configs/optimize/search.yaml --model data/models/surrogate_best.pt` |
| Verify top candidate | Run `generate_dataset.py` with fixed geometry; check `passed_configs.csv` |

---

## File map

| File | Purpose |
|---|---|
| [configs/train/surrogate.yaml](configs/train/surrogate.yaml) | Training config: data paths, architecture search space, hyperparameter ranges, budget |
| [configs/optimize/search.yaml](configs/optimize/search.yaml) | Optimizer config: objective weights, directions, pass_prob constraint, solver settings |
| [configs/generate/baseline.yaml](configs/generate/baseline.yaml) | Geometry bounds (auto-read by optimizer — do not duplicate here) |
| [src/mech390/ml/features.py](src/mech390/ml/features.py) | Feature contract: INPUT_FEATURES, REGRESSION_TARGETS — source of truth for column names |
| [src/mech390/ml/train.py](src/mech390/ml/train.py) | Optuna sweep + training loop |
| [src/mech390/ml/optimize.py](src/mech390/ml/optimize.py) | Score function + differential evolution |
| [data/models/surrogate_best.pt](data/models/surrogate_best.pt) | Trained model checkpoint (created by training run) |
| [data/models/scaler.pkl](data/models/scaler.pkl) | Input feature scaler (created by training run) |
| [data/models/target_stats.json](data/models/target_stats.json) | Regression target min/max (created by training run) |
