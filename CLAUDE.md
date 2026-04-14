# CLAUDE.md — MECH390 ML Model Project

Read automatically by Claude Code each session.
Defines subagents and rules for using them.

---

> For project background, see [README.md](README.md). For physics equations, see [The_Mother_Doc_v7.md](The_Mother_Doc_v7.md). For code contracts, see [instructions.md](instructions.md).

**Key files:**
- `src/mech390/physics/` — kinematics, dynamics, mass properties, engine
- `src/mech390/datagen/` — stage1, stage2, sampling, generate
- `src/mech390/config.py` — config loading and validation
- `configs/generate/baseline.yaml` — main configuration
- `scripts/` — preview and generation entry points
- `instructions.md` — authoritative physics derivations (read before answering physics questions)

---

## Mandatory Rule

**Invoke at least one subagent for any substantive task.**

Substantive = anything beyond simple factual question — editing code, running scripts, debugging, explaining physics, reviewing data, planning changes.

Pick subagent(s) based on request. Run multiple agents in parallel when independent.

---

## Available Subagents

---

### 1. Physics Validator

**Purpose:** Verifies physics equations, signs, units, argument cross-references after any physics-related code change.

**Trigger when:**
- Any file in `src/mech390/physics/` edited
- "does this equation make sense?", "is the physics correct?", "check the signs", "verify the derivation", or similar
- New formula or physical relationship introduced anywhere
- Output (forces, stresses, masses) looks physically wrong

**Actions:**
1. Reads `instructions.md` for authoritative derivations
2. Reads all edited physics files (`kinematics.py`, `dynamics.py`, `mass_properties.py`, `engine.py`, `stresses.py`, `fatigue.py`, `buckling.py`, `_utils.py`)
3. Checks every equation against reference derivations:
   - Sign conventions (especially alpha terms, gravity direction, friction direction)
   - Unit consistency (all SI: meters, kg, radians, Pascals, Newtons)
   - Correct application of Newton-Euler equations
   - Correct parallel-axis theorem usage
4. Traces all cross-file function calls, verifies:
   - Argument names and order match function signature
   - Return values used correctly by caller
5. Checks known issues flagged in `CLAUDE.md` (see Known Issues below)
6. Reports PASS / FAIL with specific line references for any problem

---

### 2. Cross-Reference Auditor

**Purpose:** Ensures argument names, types, orderings consistent across every module boundary.

**Trigger when:**
- Any function signature changed
- New function added called from another file
- Config key added, renamed, or removed
- "are the files consistent?", "do the arguments match?", "check the cross-references", or similar
- Before any commit or pull request

**Actions:**
1. Maps every function call to its definition across all files in `src/mech390/`
2. Verifies:
   - Positional argument order matches signature
   - Keyword argument names match exactly (no typos, no renamed params)
   - Default values physically sensible (e.g., `g=9.81`, `mu=0.0`)
   - Config dict keys used in code exist in `configs/generate/baseline.yaml`
   - Dict keys returned by one function and consumed by another are consistent
     (e.g., `compute_design_mass_properties` returns `I_mass_crank_cg_z` and
     `engine.py` reads it via `get_or_warn(design, 'I_mass_crank_cg_z', ...)`)
3. Flags any mismatch with file path and line number of both sides
4. Reports summary table: function | caller file | callee file | status

---

### 3. Data Quality Checker

**Purpose:** Validates generated CSV data for physical plausibility and ML-readiness after any data generation run.

**Trigger when:**
- `scripts/preview_stage1.py`, `scripts/preview_stage2.py`, or `scripts/generate_dataset.py` just ran
- New CSV appears in `data/`
- "is the data good?", "check the CSV", "does the output look right?", "are there any bad rows?", or similar
- Before committing generated data to repo

**Actions:**
1. Reads output CSV(s) from `data/`
2. Checks structural issues:
   - NaN or inf values in any column
   - Negative masses, inertias, or pin diameters
   - Zero values in columns that should always be positive
3. Checks physical plausibility of every column:
   - `r`, `l`, `e` within bounds in `configs/generate/baseline.yaml`
   - `ROM` within ±`ROM_tolerance` of target (0.25 m)
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
5. Reports: total rows, pass/fail count per check, suspicious rows with index

---

### 4. ML Readiness Inspector

**Purpose:** Evaluates dataset suitability for ML training — class balance, feature distributions, potential leakage, dataset size.

**Trigger when:**
- "is the dataset ready for training?", "can I train now?", "check the data for ML", "is there enough data?", or similar
- Before running `scripts/train_model.py`
- After large data generation run completes

**Actions:**
1. Reads full dataset from `data/`
2. Checks class balance:
   - Reports pass/fail ratio (`pass_fail` column)
   - Warns if one class < 20% of dataset (severe imbalance)
3. Checks feature distributions:
   - Verifies each input feature has reasonable spread (not all same value)
   - Flags features with near-zero variance
   - Reports min, max, mean, std for all numeric columns
4. Checks for data leakage:
   - Flags any column that is direct function of `pass_fail`
     (e.g., `utilization`, `sigma_max` when stresses implemented)
   - These must be excluded from training features
5. Checks dataset size:
   - Warns if total rows < 1000 (likely insufficient for ML)
   - Warns if pass cases < 200
6. Reports go / no-go recommendation with specific reasons

---

### 5. ML Data Scientist

**Purpose:** Designs, critiques, improves the neural network — architecture, loss function, training loop, hyperparameter search, evaluation metrics. Focuses on maximizing surrogate accuracy given dataset constraints (currently 200 rows).

**Trigger when:**
- "how can we improve the model?", "is the NN architecture good?", "should we change the loss function?", "why is n_buck prediction bad?", or similar
- Val F1 or val R² drops after code change
- New dataset generated and retraining needed
- Optimizer produces physically invalid candidates (may indicate poor surrogate quality)
- Before committing changes to any file in `src/mech390/ml/`

**Actions:**
1. Reads full ML stack:
   - `src/mech390/ml/models.py` — architecture (shared trunk, clf + reg heads)
   - `src/mech390/ml/train.py` — loss function, Optuna sweep, early stopping logic
   - `src/mech390/ml/features.py` — input features, regression targets, normalization
   - `src/mech390/ml/optimize.py` — how surrogate outputs used in optimization
   - `configs/train/surrogate.yaml` — hyperparameter search space
2. Audits loss function:
   - Are `w_bce` and `w_mse` weights balanced for the task?
   - Is Optuna objective (`val_f1`) the right metric, or should it be composite of F1 + mean R²?
   - Are regression targets normalized correctly before MSE computation?
3. Audits architecture for data regime (≤200 rows):
   - Is model capacity appropriate (risk of overfitting)?
   - Is dropout tuned for small datasets?
   - Would separate trunks per head improve regression accuracy?
4. Flags unreliable regression targets:
   - `n_buck` at 200 rows — report val R² and recommend whether to keep or drop
   - Any target with val R² < 0.7 flagged as unreliable for optimizer use
5. Recommends concrete, actionable improvements:
   - Loss function changes (e.g., add per-head weights, switch to Huber loss)
   - Architecture changes (e.g., deeper trunk, separate heads, residual connections)
   - Training changes (e.g., learning rate schedule, gradient clipping)
   - Feature engineering (e.g., derived ratios like `l/r`, `e/l`, slenderness `l/t`)
6. Reports: current val F1, per-target val R², recommended changes ranked by expected impact, which changes require retraining vs config-only

---

## Known Issues (Physics)

**Sign convention at Pin A (deferred):** `stresses.py:139–140` defines `F_r,crank,A` and `F_t,crank,A` with opposite signs to Mother Doc Eqs. 2.5–2.6. No numerical impact while `abs()` wrapping used throughout crank stress path.

**OOP bending model (settled):** `M_eta = F_r_B · i_offset` is **constant along rod and crank** — no `(1 − ξ/L)` decay. System fully planar (Newton-Euler 2D), no ζ pin reactions → no lateral force couple → no moment gradient. Any code or doc showing linear decay is wrong.

**Gravity bending removed from rod and crank:** self-weight enters F_A, F_B, F_C via Newton-Euler. Separate UDL gravity bending double-counted it — removed from both `_rod_stresses` and `_crank_stresses`.

## Optimizer Constraints Implemented

Surrogate optimizer (`src/mech390/ml/optimize.py`) enforces four hard analytical constraints as penalty terms in score function, in addition to surrogate pass_prob gate:

1. **Net-section feasibility** — `width - D_pin > delta + 2×min_wall` for all 4 pin pairs.
   Prevents near-zero net sections that produce ~TPa stresses.
2. **Kinematic feasibility** — `l > r + e` (rod must bridge pin B to pin C at all crank angles).
   Without this, optimizer could find r+e > l, crashing physics engine at θ≈75°.
3. **Euler buckling** — analytical `P_cr = π²EI/l²` check; penalises `n_buck < 3.0`.
   Surrogate's n_buck predictions unreliable at 200 rows; this enforces it analytically.
4. **ROM constraint** — `|kinematics.calculate_metrics(r,l,e)['ROM'] - 0.25| > ROM_tolerance` penalised via `penalty_rom_scale=200`. All training data satisfies ROM=250mm±0.5mm; without this, optimizer picks arbitrary r → wrong stroke.
5. **QRR constraint** — `QRR ∉ [1.5, 2.5]` penalised via `penalty_qrr_scale=20`. Same reasoning — training data entirely within this band. Both ROM and QRR computed via `kinematics.calculate_metrics()` (same formula as Stage 1).

Added after physics validation (`scripts/validate_candidate.py`) revealed earlier optimizer outputs were kinematically infeasible or failed buckling. ROM/QRR added after optimizer was found to ignore these kinematic specs entirely.

## Current ML Model State

**Dataset:** last run — 10,000 rows, 7,526 pass (75.3%) / 2,474 fail (24.7%). Size will change; see `baseline.yaml`. Retraining after dataset change requires only re-running `scripts/train_model.py` — no code changes needed.

**Best trained surrogate** (`data/models/surrogate_best.pt`):
- Val F1: **0.9713** | Architecture: `[256, 128, 64]` | Dropout: 0.299 | LR: 9.17e-3 | Batch: 128
- 50 Optuna trials × 300 epochs max, patience=25

**Architecture decisions already implemented:**
- Early stopping saves on `val_f1 > best_val_f1` (not `val_loss`) — ensures best-classified epoch saved, not best-regressed
- `input_dim` and `n_reg_targets` derived from `len(F.INPUT_FEATURES)` / `len(F.REGRESSION_TARGETS)` in `train.py`; no hardcoded integers in training path
- `infer.py` asserts `ckpt['hparams']['n_reg_targets'] == len(F.REGRESSION_TARGETS)` at load time to catch stale checkpoints

**Optimizer OOD penalty (ML-P2 — implemented):** Regression predictions outside training range by more than `ood_tolerance` (default 10%) penalised as `weight × ood_excess × ood_penalty_scale`. Prediction 3× training max produces deduction that completely dominates score, eliminating that candidate. Both parameters configurable in `configs/optimize/search.yaml` under `constraints`. Previously, OOD predictions silently clamped to [0, 1] — root cause of surrogate 0.97 → physics 0.48 gap on Rank 1 candidates.

**Next recommended ML improvement (ML-P8 — Split regression training):**
Train classification head on all data (pass + fail); train regression heads only on passing rows. Regression targets (mass, torque, safety factors) physically meaningful only for passing designs; training on fail configs adds noise to regression gradient. pass_prob gate in optimizer still requires classifier to have seen both classes.

---

## Pipeline Scripts

**Full pipeline (recommended):**
```bash
.venv/bin/python scripts/run_pipeline.py
# flags: --skip-datagen, --skip-training, --top N, --seed INT, --log-level LEVEL
```
Runs 4 steps in sequence: data generation → surrogate training → optimization → manufacturing report.

**Step 4 — Manufacturing report** (`scripts/summarize_results.py`):
Reads `data/results/candidates.json` (written by `optimize_design.py --out-json`), runs full physics validation on top-N candidates, prints ROM/QRR/FoS/mass/verdict for each. Standalone:
```bash
.venv/bin/python scripts/summarize_results.py --top 3
```

**Physics validation script** (`scripts/validate_candidate.py`):
Runs specific geometry dict through full physics engine without Stage 1. Use to verify any optimizer candidate before committing to manufacture.
```bash
.venv/bin/python scripts/validate_candidate.py --config configs/generate/baseline.yaml
```
Edit `CANDIDATE` at top of script to test different geometries.

---

## Quick Reference for Teammates

No Python or mechanics knowledge needed.
Describe what you want in plain English. Examples:

**Run full pipeline:**
```bash
.venv/bin/python scripts/run_pipeline.py
```
Runs all 4 steps: data gen → train → optimize → manufacturing report (top 3 candidates with physics validation).

**Run only data generation:**
```bash
.venv/bin/python scripts/generate_dataset.py \
    --config  configs/generate/baseline.yaml \
    --seed    42 \
    --out-dir data/preview
```
Writes 7 CSVs to `data/preview/`.

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