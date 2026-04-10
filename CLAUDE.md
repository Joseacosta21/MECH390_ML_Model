# CLAUDE.md — MECH390 ML Model Project

This file is read automatically by Claude Code at the start of every session.
It defines the available subagents and the rules for using them.

---

> For project background, see [README.md](README.md). For physics equations, see [The_Mother_Doc_v7.md](The_Mother_Doc_v7.md). For code contracts, see [instructions.md](instructions.md).

**Key files:**
- `src/mech390/physics/` — kinematics, dynamics, mass properties, engine
- `src/mech390/datagen/` — stage1, stage2, sampling, generate
- `src/mech390/config.py` — config loading and validation
- `configs/generate/baseline.yaml` — main configuration
- `scripts/` — preview and generation entry points
- `instructions.md` — authoritative physics derivations (read this before answering physics questions)

---

## Mandatory Rule

**You must invoke at least one subagent for any substantive task.**

A substantive task is anything beyond a simple factual question — including but not
limited to: editing code, running scripts, debugging, explaining physics, reviewing
data, or planning changes.

Choose the subagent(s) based on what the request involves. Multiple agents can be
run in parallel when their tasks are independent.

---

## Available Subagents

---

### 1. Physics Validator

**Purpose:** Verifies that physics equations, signs, units, and argument
cross-references are correct after any change to physics-related code.

**Trigger this agent when:**
- Any file in `src/mech390/physics/` is edited
- Someone asks "does this equation make sense?", "is the physics correct?",
  "check the signs", "verify the derivation", or similar
- A new formula or physical relationship is introduced anywhere in the codebase
- Something in the output (forces, stresses, masses) looks physically wrong

**What this agent does:**
1. Reads `instructions.md` for the authoritative derivations
2. Reads all edited physics files (`kinematics.py`, `dynamics.py`,
   `mass_properties.py`, `engine.py`, `stresses.py`, `fatigue.py`,
   `buckling.py`, `_utils.py`)
3. Checks every equation against the reference derivations:
   - Sign conventions (especially alpha terms, gravity direction, friction direction)
   - Unit consistency (all SI: meters, kg, radians, Pascals, Newtons)
   - Correct application of Newton-Euler equations
   - Correct parallel-axis theorem usage
4. Traces all cross-file function calls and verifies:
   - Argument names and order match the function signature
   - Return values are used correctly by the caller
5. Checks for known issues flagged in `CLAUDE.md` (see Known Issues section below)
6. Reports a clear PASS / FAIL with specific line references for any problem found

---

### 2. Cross-Reference Auditor

**Purpose:** Ensures all argument names, types, and orderings are consistent
across every module boundary in the project.

**Trigger this agent when:**
- Any function signature is changed
- A new function is added that will be called from another file
- A config key is added, renamed, or removed
- Someone asks "are the files consistent?", "do the arguments match?",
  "check the cross-references", or similar
- Before any commit or pull request

**What this agent does:**
1. Maps every function call to its definition across all files in `src/mech390/`
2. Verifies:
   - Positional argument order matches the signature
   - Keyword argument names match exactly (no typos, no renamed params)
   - Default values are physically sensible (e.g., `g=9.81`, `mu=0.0`)
   - Config dict keys used in code exist in `configs/generate/baseline.yaml`
   - Dict keys returned by one function and consumed by another are consistent
     (e.g., `compute_design_mass_properties` returns `I_mass_crank_cg_z` and
     `engine.py` reads it via `get_or_warn(design, 'I_mass_crank_cg_z', ...)`)
3. Flags any mismatch with the file path and line number of both sides
4. Reports a summary table: function | caller file | callee file | status

---

### 3. Data Quality Checker

**Purpose:** Validates generated CSV data for physical plausibility and
ML-readiness after any data generation run.

**Trigger this agent when:**
- `scripts/preview_stage1.py`, `scripts/preview_stage2.py`, or
  `scripts/generate_dataset.py` has just been run
- A new CSV appears in `data/`
- Someone asks "is the data good?", "check the CSV", "does the output look right?",
  "are there any bad rows?", or similar
- Before committing generated data to the repo

**What this agent does:**
1. Reads the output CSV(s) from `data/`
2. Checks for structural issues:
   - NaN or inf values in any column
   - Negative masses, inertias, or pin diameters
   - Zero values in columns that should always be positive
3. Checks physical plausibility of every column:
   - `r`, `l`, `e` within the bounds in `configs/generate/baseline.yaml`
   - `ROM` within ±`ROM_tolerance` of the target (0.25 m)
   - `QRR` within [1.5, 2.5]
   - `mass_crank`, `mass_rod`, `mass_slider` > 0; `total_mass` == sum of parts
   - `I_mass_*` > 0
   - `sigma_max`, `tau_max` >= 0
   - `volume_envelope` > 0 (order of magnitude: 10⁻⁴ to 10⁻² m³)
   - `tau_A_max` > 0; `F_A_max`, `F_B_max`, `F_C_max` > 0
   - `n_static_rod`, `n_static_crank`, `n_static_pin` >= 1.0 in `passed_configs.csv`
4. Checks dataset statistics:
   - Row count matches expected `n_samples × n_variants_per_2d`
   - No duplicate rows (exact or near-duplicate geometry)
   - Column count matches expected schema (85 columns in passed/failed_configs.csv as of latest run)
5. Reports: total rows, pass/fail count per check, any suspicious rows with index

---

### 4. ML Readiness Inspector

**Purpose:** Evaluates whether the dataset is suitable for ML training — checks
class balance, feature distributions, potential leakage, and dataset size.

**Trigger this agent when:**
- Someone asks "is the dataset ready for training?", "can I train now?",
  "check the data for ML", "is there enough data?", or similar
- Before running `scripts/train_model.py`
- After a large data generation run completes

**What this agent does:**
1. Reads the full dataset from `data/`
2. Checks class balance:
   - Reports pass/fail ratio (`pass_fail` column)
   - Warns if one class is < 20% of the dataset (severe imbalance)
3. Checks feature distributions:
   - Verifies each input feature has reasonable spread (not all the same value)
   - Flags features with near-zero variance
   - Reports min, max, mean, std for all numeric columns
4. Checks for data leakage:
   - Flags any column that is a direct function of `pass_fail`
     (e.g., `utilization`, `sigma_max` when stresses are implemented)
   - These must be excluded from training features
5. Checks dataset size:
   - Warns if total rows < 1000 (likely insufficient for ML)
   - Warns if pass cases < 200
6. Reports a go / no-go recommendation with specific reasons

---

### 5. ML Data Scientist

**Purpose:** Designs, critiques, and improves the neural network — architecture,
loss function, training loop, hyperparameter search, and evaluation metrics.
This agent focuses on making the surrogate as accurate as possible given the
dataset constraints (currently 200 rows).

**Trigger this agent when:**
- Someone asks "how can we improve the model?", "is the NN architecture good?",
  "should we change the loss function?", "why is n_buck prediction bad?", or similar
- Val F1 or val R² drops after a code change
- A new dataset is generated and retraining is needed
- The optimizer produces physically invalid candidates (may indicate poor surrogate quality)
- Before committing changes to any file in `src/mech390/ml/`

**What this agent does:**
1. Reads the full ML stack:
   - `src/mech390/ml/models.py` — architecture (shared trunk, clf + reg heads)
   - `src/mech390/ml/train.py` — loss function, Optuna sweep, early stopping logic
   - `src/mech390/ml/features.py` — input features, regression targets, normalization
   - `src/mech390/ml/optimize.py` — how surrogate outputs are used in optimization
   - `configs/train/surrogate.yaml` — hyperparameter search space
2. Audits the loss function:
   - Are `w_bce` and `w_mse` weights balanced for the task?
   - Is the Optuna objective (`val_f1`) the right metric, or should it be a
     composite of F1 + mean R²?
   - Are regression targets normalized correctly before MSE computation?
3. Audits the architecture for the data regime (≤200 rows):
   - Is the model capacity appropriate (risk of overfitting)?
   - Is dropout tuned for small datasets?
   - Would separate trunks per head improve regression accuracy?
4. Flags unreliable regression targets:
   - `n_buck` at 200 rows — report val R² and recommend whether to keep or drop
   - Any target with val R² < 0.7 should be flagged as unreliable for optimizer use
5. Recommends concrete, actionable improvements:
   - Loss function changes (e.g., add per-head weights, switch to Huber loss)
   - Architecture changes (e.g., deeper trunk, separate heads, residual connections)
   - Training changes (e.g., learning rate schedule, gradient clipping)
   - Feature engineering (e.g., derived ratios like `l/r`, `e/l`, slenderness `l/t`)
6. Reports: current val F1, per-target val R², recommended changes ranked by
   expected impact, and which changes require retraining vs config-only

---

## Known Issues (Physics)

**Sign convention at Pin A (deferred):** `stresses.py:139–140` defines `F_r,crank,A` and `F_t,crank,A` with opposite signs to Mother Doc Eqs. 2.5–2.6. No numerical impact while `abs()` wrapping is used throughout the crank stress path. See `STRESS_AUDIT.md` Section 10.

## Optimizer Constraints Implemented

The surrogate optimizer (`src/mech390/ml/optimize.py`) enforces four hard analytical
constraints as penalty terms in the score function, in addition to the surrogate pass_prob gate:

1. **Net-section feasibility** — `width - D_pin > delta + 2×min_wall` for all 4 pin pairs.
   Prevents near-zero net sections that produce ~TPa stresses.
2. **Kinematic feasibility** — `l > r + e` (rod must bridge pin B to pin C at all crank angles).
   Without this, the optimizer could find r+e > l, which crashes the physics engine at θ≈75°.
3. **Euler buckling** — analytical `P_cr = π²EI/l²` check; penalises `n_buck < 3.0`.
   The surrogate's n_buck predictions are unreliable at 200 rows; this enforces it analytically.

These were added after physics validation (`scripts/validate_candidate.py`) revealed that
earlier optimizer outputs were kinematically infeasible or failed buckling.

## Physics Validation Script

`scripts/validate_candidate.py` — runs a specific geometry dict through the full physics
engine without going through Stage 1 (which would re-compute r). Use this to verify any
optimizer candidate before committing to manufacture.

```bash
.venv/bin/python scripts/validate_candidate.py --config configs/generate/baseline.yaml
```

Edit `CANDIDATE` at the top of the script to test different geometries.

---

## Quick Reference for Teammates

You do not need to know Python or mechanics to use this project.
Just describe what you want in plain English. Examples:

**Run the full pipeline:**
```bash
.venv/bin/python scripts/generate_dataset.py \
    --config  configs/generate/baseline.yaml \
    --seed    42 \
    --out-dir data/preview
```
Writes 7 CSVs to `data/preview/`: kinematics, dynamics, stresses, fatigue, buckling, passed_configs, failed_configs.

**Plain English requests:**

| What you say | What Claude will do |
|---|---|
| "I changed the rod acceleration formula" | Spawns Physics Validator + Cross-Reference Auditor |
| "Run the data generation and check the output" | Runs script, then spawns Data Quality Checker |
| "Is our dataset ready to train?" | Spawns ML Readiness Inspector |
| "Does this equation look right?" | Spawns Physics Validator, reads instructions.md |
| "I updated a function signature" | Spawns Cross-Reference Auditor |
| "Everything looks off, check the whole pipeline" | Spawns all four agents in parallel |
| "How can we improve the NN?" | Spawns ML Data Scientist |
| "Why is n_buck prediction bad?" | Spawns ML Data Scientist |
| "Retrain after new data" | Spawns ML Data Scientist + ML Readiness Inspector |
