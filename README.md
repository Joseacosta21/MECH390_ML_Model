# Data-Driven Design of an Offset Crank–Slider Mechanism

**MECH 390 – Machine Learning–Assisted Mechanical Design**

---

## For agents and teammates — read this first

**Setup (one time):** Install [Claude Code](https://claude.ai/code), open this repo, start Claude — it automatically reads `CLAUDE.md` and loads all agent rules.

**Key files:**

| File | Purpose |
|---|---|
| `CLAUDE.md` | Agent definitions and mandatory rules — read before anything else |
| `instructions.md` | Authoritative physics derivations and full technical spec |
| `configs/generate/baseline.yaml` | Controls geometry ranges, sampling, and pass/fail limits |
| `data/preview/` | All pipeline and preview-script outputs (default output directory) |

**Making requests (plain English works):**

| What you say | What Claude does |
|---|---|
| "I changed the rod formula" | Physics Validator + Cross-Reference Auditor |
| "Run the data generation and check it" | Runs pipeline → Data Quality Checker |
| "Is the dataset ready to train?" | ML Readiness Inspector |
| "Everything looks wrong" | All four agents in parallel |

> **For AI agents:** The To-Do list in this file is the authoritative backlog. It is intentionally large — do not attempt to complete it in one session. Pick one task, validate it with the appropriate subagent(s), commit, and stop.

---

## 1. What this project is

A **physics-first data generation and ML pipeline** for the offset crank–slider mechanism, built for the MECH 390 Winter 2026 design project at Concordia University.

Physics generates the data. ML learns pass/fail patterns from it. No ML shortcuts replace the physics.

**Design specifications (fixed — not design variables):**

| Spec | Value |
|---|---|
| Reaction force (slider load) | 500 g (~4.905 N) |
| Range of motion (ROM target) | 250 mm ± 0.5 mm |
| Input speed | 30 RPM |
| Quick return ratio (QRR) | 1.5 – 2.5 |
| Link material | Al 2024-T3 (ρ=2780 kg/m³, E=73.1 GPa, S_ut=483 MPa, S_y=345 MPa) |
| Link geometry | Rectangular cross-section |
| Slider–guide friction (μ) | 0.47 — dry machined Al–Al (Shigley's) |

**What the ML model must predict:**
- Pass/fail classification for a given design configuration
- Minimum safety factors (static and fatigue)
- Optimal QRR to minimize crank torque and motor power

**Optimization objectives (Week 8):**
- Minimize mechanism envelope (size and weight)
- Minimize required motor power while satisfying all design targets


> CAD modeling, 3D-printed prototype, and the written report are handled outside this repository.

---

## 2. Pipeline

```
configs/generate/baseline.yaml
         │
         ▼
┌─────────────────────────┐
│  Stage 1 – 2D Kinematic │  stage1_kinematic.py
│  Screening              │  Samples (r, l, e), solves ROM, checks QRR
└──────────┬──────────────┘
           │ valid (r, l, e)
           ▼
┌─────────────────────────┐
│  Stage 2 – 3D Embodiment│  stage2_embodiment.py
│  Expansion              │  Adds cross-sections and pin diameters
└──────────┬──────────────┘
           │ 3D design dicts
           ▼
┌─────────────────────────┐
│  Mass Properties        │  mass_properties.py
│                         │  Masses, COGs, MOIs for each link
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Physics Engine         │  engine.py + dynamics.py
│  (15° sweep, 24 angles) │  Newton-Euler 8×8 → F_A, F_B, F_C, N, F_f, tau_A
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Stress + Fatigue +     │  stresses.py, fatigue.py, buckling.py
│  Buckling Evaluation    │  σ, τ per angle; Goodman/Miner; Euler buckling
└──────────┬──────────────┘
           │ sigma_max, tau_max + tau_A_max, E_rev, F_A/B/C_max, σ_peak per link
           ▼
┌─────────────────────────┐
│  Pass / Fail Label      │  generate.py
│  + ML Feature Assembly  │  utilization, n_buck, fatigue FOSs
│                         │  total_mass, volume_envelope
│                         │  tau_A_max, E_rev, F_A/B/C_max
│                         │  n_static_rod/crank/pin
└──────────┬──────────────┘
           │ labeled dataset (85 cols)
           ▼
┌─────────────────────────┐
│  ML Training            │  ml/
│  PyTorch multi-task NN  │  ReLU trunk + classification & regression heads
│  Optuna hyperpar. sweep │  Inputs: 10 design vars → pass_fail + 8 targets
└─────────────────────────┘
```

**Hard constraints (enforced by Stage 1):**

| Constraint | Value |
|---|---|
| ROM target | 250 mm ± 0.5 mm |
| QRR | [1.5, 2.5] |
| RPM | 30 (fixed, not a design variable) |

**Key variable names** (full physics derivations in `instructions.md`):

| Code key | Meaning | Unit |
|---|---|---|
| `r`, `l`, `e` | Crank radius, rod length, offset | m |
| `theta`, `omega` | Crank angle, angular speed (= RPM × 2π/60) | rad, rad/s |
| `ROM`, `QRR` | Slider stroke, quick-return ratio | m, — |
| `width_r/l`, `thickness_r/l` | Link cross-sections | m |
| `d_shaft_A` | Motor output shaft diameter at joint A | m |
| `pin_diameter_B`, `pin_diameter_C` | Lug pin diameters at joints B and C | m |
| `F_A`, `F_B`, `F_C` | Joint reaction vectors [Fx, Fy] | N |
| `N`, `F_f`, `tau_A` | Slider normal, friction, drive torque | N, N, N·m |
| `sigma_max`, `tau_max` | Peak normal/shear stress over cycle | Pa |
| `n_static_rod/crank/pin` | Static FOS per link: sigma_limit / peak σ_link, with `sigma_limit = yield_stress/safety_factor` | — |
| `total_mass` | mass_crank + mass_rod + mass_slider | kg |
| `volume_envelope` | Bounding-box volume of assembled mechanism (T × H × L) | m³ |
| `tau_A_max` | Peak motor torque over full cycle | N·m |
| `E_rev` | Energy per revolution ∫τ dθ (discrete sum × 2π/24) | J |
| `F_A_max`, `F_B_max`, `F_C_max` | Peak resultant force at each pin over cycle | N |
| `pass_fail` | 1 = pass, 0 = fail | — |

---

## 3. Repository structure

```
MECH390_ML_Model/
├── CLAUDE.md
├── instructions.md
├── configs/
│   ├── generate/
│   │   ├── baseline.yaml        # ✅ Main config (40 samples, LHS, 5 variants/2D)
│   │   ├── test_small.yaml      # Fast test config
│   │   └── aggressive.yaml      # 🔲 EMPTY — wider ranges for diverse training
│   ├── train/
│   │   └── surrogate.yaml       # ✅ Optuna sweep config (arch, dropout, lr, batch)
│   └── optimize/
│       └── search.yaml          # ✅ Weight table + optimizer settings
├── src/mech390/
│   ├── config.py                # ✅ Config loading, normalization, validation
│   ├── physics/
│   │   ├── _utils.py            # ✅ Shared utilities (get_or_warn fallback logging)
│   │   ├── kinematics.py        # ✅ Positions, velocities, accelerations, ROM/QRR
│   │   ├── dynamics.py          # ✅ Newton-Euler 8×8 solver
│   │   ├── mass_properties.py   # ✅ Mass, MOI, COG helpers + design aggregator
│   │   ├── engine.py            # ✅ 15° sweep orchestrator
│   │   ├── stresses.py          # ✅ σ, τ per crank angle (axial, bending, torsion, shear)
│   │   ├── fatigue.py           # ✅ Marin factors, Modified Goodman, Basquin, Miner's rule
│   │   └── buckling.py          # ✅ Euler buckling check (pin-pin, weak axis)
│   ├── datagen/
│   │   ├── sampling.py          # ✅ LHS + random sampler
│   │   ├── stage1_kinematic.py  # ✅ Full Stage 1 (streaming iterator)
│   │   ├── stage2_embodiment.py # ✅ 3D expansion
│   │   └── generate.py          # ✅ Orchestrator with pass/fail labeling (85 cols)
│   └── ml/
│       ├── __init__.py          # ✅ Package init
│       ├── features.py          # ✅ Scaler, feature split, min_n_static, target stats
│       ├── models.py            # ✅ CrankSliderSurrogate (ReLU trunk + clf + reg heads)
│       ├── train.py             # ✅ Optuna sweep, early stopping, checkpoint save
│       ├── infer.py             # ✅ SurrogatePredictor (dict / DataFrame inference)
│       └── optimize.py          # ✅ Surrogate optimizer (differential_evolution, top-N)
├── scripts/
│   ├── preview_stage1.py        # ✅ Stage 1 → CSV
│   ├── preview_stage2.py        # ✅ Stage 1 + Stage 2 + mass props → CSV
│   ├── preview_forces.py        # ✅ Full pipeline → force sweep (4800 rows)
│   ├── generate_dataset.py      # ✅ Full pipeline CLI → 7 CSVs
│   ├── train_model.py           # ✅ Surrogate training CLI
│   └── optimize_design.py       # ✅ Surrogate optimizer CLI
├── data/
│   ├── preview/                     # All quick-run outputs (preview_*.py + generate_dataset.py default)
│   │   ├── stage1_geometries.csv    # 40 rows × 7 cols
│   │   ├── stage2_designs.csv       # 200 rows × 27 cols
│   │   ├── forces_sweep.csv         # 4800 rows × 15 cols (200 designs × 24 angles)
│   │   ├── kinematics.csv           # 4800 rows — per (design, angle): positions, velocities, accels
│   │   ├── dynamics.csv             # 4800 rows — per (design, angle): joint forces, torque
│   │   ├── stresses.csv             # 4800 rows — per (design, angle): per-component σ, τ
│   │   ├── fatigue.csv              # 200 rows  — per design: Goodman / Miner metrics
│   │   ├── buckling.csv             # 200 rows  — per design: Euler buckling metrics
│   │   ├── passed_configs.csv       # N rows    — passing designs with all check columns
│   │   └── failed_configs.csv       # N rows    — failing designs with all check columns
│   ├── runs/       # Named production runs: --out-dir data/runs/<name>
│   ├── processed/
│   ├── splits/
│   └── models/
└── tests/
    └── test_datagen_units.py    # Kinematics, datagen, import tests
```

---

## 4. How to run

```bash
# One-time setup
python3 -m venv .venv
.venv/bin/pip install pandas scipy numpy pyyaml

# Full pipeline — generates all 7 CSVs (kinematics, dynamics, stresses,
# fatigue, buckling, passed_configs, failed_configs) in data/preview/
.venv/bin/python scripts/generate_dataset.py

# With explicit config, seed, and output directory
.venv/bin/python scripts/generate_dataset.py \
    --config  configs/generate/baseline.yaml \
    --seed    42 \
    --out-dir data/preview

# Preview scripts (individual pipeline stages)
.venv/bin/python3 scripts/preview_stage1.py --out-dir data/preview
.venv/bin/python3 scripts/preview_stage2.py --out-dir data/preview
.venv/bin/python3 scripts/preview_forces.py --out-dir data/preview

# All scripts accept --config, --seed, --out-dir (and --max-2d for stage2/forces)
```

---

## 5. Known bugs

No open bugs.

---

## 6. To-Do list

> **For agents and contributors:** This is the full project backlog. It spans many sessions.
> Complete one or two items per session — validate, commit, push, then stop.
> Physics changes → **Physics Validator**. Signature changes → **Cross-Reference Auditor**. New data → **Data Quality Checker**.
>
> Academic schedule reference: Weeks 4–5 = dataset generation, Weeks 6–7 = ML training + validation, Week 8 = optimization + visualization.

### High priority (Weeks 4–5 — dataset generation)

- [x] **`generate_dataset.py` CLI** — fully implemented; writes 7 CSVs to `data/preview/`. Pass/fail considers static stress, buckling, fatigue Goodman, and Miner's rule.
- [x] **Fix post-rounding duplicates** — post-generation dedup pass on geometry columns; duplicate design_ids removed from all 7 DataFrames before CSV write. Count reported in summary.
- [ ] **Fill `configs/generate/aggressive.yaml`** — wider ranges, higher n_samples (reference: `baseline.yaml`)
- [ ] **Populate `data/splits/`** — train/val/test split CSVs from the full dataset

### ML pipeline (Weeks 6–7)

- [x] **`ml/features.py`** — StandardScaler, feature/target split, `min_n_static` derived column, train/val/test stratified split, `compute_target_stats()` for score normalization
- [x] **`ml/models.py`** — `CrankSliderSurrogate`: shared ReLU FC trunk + classification head (pass_fail, sigmoid+BCE) + regression head (7 targets, linear+MSE); `save/load_checkpoint`, `build_model_from_hparams`
- [x] **`ml/train.py`** — Optuna hyperparameter sweep (50 trials × 300 epochs), early stopping, saves best checkpoint + scaler + target_stats to `data/models/`
- [x] **`ml/infer.py`** — `SurrogatePredictor`: loads checkpoint + scaler, `predict()` accepts dict / list / DataFrame, returns pass_prob + all 7 regression targets
- [x] **`ml/optimize.py`** — `run_optimization()`: reads geometry bounds from `baseline.yaml`, weighted score function with pass_prob penalty, `scipy differential_evolution`, returns top-N distinct candidates
- [x] **`scripts/train_model.py`** — CLI: `python scripts/train_model.py --config configs/train/surrogate.yaml`
- [x] **`scripts/optimize_design.py`** — CLI: `python scripts/optimize_design.py --generate-config ... --optimize-config ... --model ...`
- [x] **`configs/train/surrogate.yaml`** — full Optuna sweep config (architecture options, dropout, lr, batch size, weight decay ranges)
- [x] **`configs/optimize/search.yaml`** — user-editable weight table (objectives + directions + constraint on pass_prob); bounds auto-read from `baseline.yaml`

### ML To-Do (surrogate optimizer)

- [x] **Run full training** — 50 Optuna trials × 300 epochs. Best val F1 = 0.9714, architecture [256, 128, 64]
- [x] **Cross-validate top-1 candidate** — `scripts/validate_candidate.py` bypasses Stage 1 and feeds geometry directly to physics engine; Rank 1 passes all checks (n_buck=3.03, n_shaft=2.89, all Miner D=0)
- [ ] **Early stopping on val_f1 (ML-P1)** — `train.py:127` saves checkpoint when `val_loss` improves, not `val_f1`; change condition to `val_f1 > best_val_f1` so the best-classified model is saved, not the best-regressed one
- [ ] **OOD penalty in score function (ML-P2)** — `optimize.py:151` clips `norm` to `[0, 1]`, giving out-of-distribution designs a free maximum score; replace with: penalize `|norm - clip(norm, 0, 1)| × 10` when `norm < -0.1` or `norm > 1.1`; this was the primary cause of the surrogate 0.759 → physics 0.48 gap
- [ ] **Remove `min_n_static` from optimizer objectives (ML-P3)** — `search.yaml` still lists `min_n_static`; designs with thin fatigue-failing cranks show paradoxically high static FOS, inverting the optimization direction; remove it and redistribute its weight to `total_mass` and `volume_envelope`
- [ ] **Refactor features: drop 3 noisy targets, add 3 derived inputs (ML-P4)** — `features.py`: drop `n_buck`, `utilization`, `min_n_static` from `REGRESSION_TARGETS` (noisy or circular at 200 rows; n_buck is enforced analytically in optimizer); add `slenderness_r = r/thickness_r`, `slenderness_l = l/thickness_l`, `net_section_r = width_r - d_shaft_A` to `INPUT_FEATURES`; update `input_dim` to 13, `n_reg_targets` to 5
- [ ] **Update model defaults to match new feature contract (ML-P5)** — `models.py`: update `input_dim` default from 10 to 13 and `n_reg_targets` default from 8 to 5 after ML-P4 is applied
- [ ] **Restrict architecture search space (ML-P6)** — `surrogate.yaml`: remove `[256,128,64]`, `[512,256,128]` (460k+ params at 140 training rows = memorization); add `[32,32]`, `[64,32]`; set `batch_size_options: [16, 32]`; `dropout_range: [0.2, 0.5]`; `use_batch_norm: false`; `weight_decay_range: [1e-5, 5e-2]`; `loss_weights.mse: 0.8`
- [ ] **Add learning rate schedule (ML-P7)** — `train.py`: add `CosineAnnealingLR(optimizer, T_max=150)` after Adam definition; small datasets plateau early without a scheduler
- [ ] **Validate regression quality** — val R² not yet confirmed for `total_mass`, `volume_envelope`, `tau_A_max`; n_buck and utilization known unreliable at 200 rows (enforced analytically in optimizer)
- [ ] **Fix target normalisation (8.1)** — target stats computed on train set then applied to val/test; values outside training range are silently clipped; use explicit out-of-range warning or switch to StandardScaler
- [ ] **Stratify regression targets in split (8.2)** — train/val split only stratifies on pass_fail; add warning if any regression target distribution is severely skewed across splits
- [ ] **Remove hardcoded input_dim (8.3)** — input_dim=10 appears in both features.py and models.py default; pass len(INPUT_FEATURES) explicitly from train.py so adding a feature doesn't require updating models.py
- [ ] **Add checkpoint version field (8.5)** — checkpoint has no schema version; add a version key to save_checkpoint() and validate it in build_model_from_hparams() so schema changes are caught at load time

### Optimization and visualization (Week 8)

- [x] **Surrogate optimizer** — `scripts/optimize_design.py` runs differential_evolution over 10D space (75k surrogate evals), returns top-10 candidates ranked by weighted score
- [x] **Physics validation script** — `scripts/validate_candidate.py` validates any optimizer output through the full physics engine; edit `CANDIDATE` dict at top of file
- [x] **Dataset vs optimizer comparison** — best design in training data (design 126, score=0.85) outperforms optimizer Rank 1 (score=0.48 on actual physics) due to surrogate extrapolation at 200 rows
- [ ] **`scripts/visualize_design.py`** — 2D mechanism drawing from design_id: crank, rod, slider with cross-section widths, pin circles at A/B/C, guide rail, annotated metrics
- [ ] **Sensitivity plots** — each input feature vs `utilization` / `pass_fail`
- [ ] **Parameter correlation map** — heatmap of feature–target correlations

### Physics and testing

- [x] **Size factor formula check** — replaced Shigley piecewise with Mott Table 5-3; split `C_s` (size) and `C_sur` (manufacturing method) per Mott naming
- [x] **Sn verified** — 133 MPa at design life 18.72×10⁶ cycles (Al 2024-T3, ASM)
- [x] **Fix Basquin exponent denominator** — replaced synthetic steel-origin formula with experimental AA2024-T3 constants: `σa = 924·N^(−0.086)` from anchors (10⁷, 230 MPa) + (10⁹, 155 MPa)
- [ ] **`preview_stresses.py`** — like `preview_forces.py` but outputs σ/τ per angle
- [ ] **Expand `test_datagen_units.py`** — add dynamics, mass properties, and stress tests
- [ ] **Add regression tests** — fixed-seed full pipeline run vs reference snapshot
- [ ] **Physics module tests** — add unit tests for dynamics.py (Newton-Euler solver), stresses.py (rod/crank/pin formulas), fatigue.py (Goodman + Miner), buckling.py (critical load), mass_properties.py (mass/inertia formulas), engine.py (full sweep integration) — currently 0 tests
- [ ] **ML stack tests** — add tests for features.py (normalisation round-trip), models.py (forward pass shapes), infer.py (load + predict cycle), optimize.py (constraint satisfaction)
- [ ] **Test infrastructure** — add pytest.ini or pyproject.toml test config; add shared fixtures
- [ ] **Integration / regression test** — fixed-seed config → dataset → at least N passing designs; run as CI smoke test

### Code hygiene

- [ ] **Unified logging (9.1)** — replace remaining `print()` calls in `generate_dataset.py:136–137` with `logger.info()`; ensure all entry-point scripts configure a root logger
- [ ] **Type annotations (9.2)** — add type hints to physics modules (dynamics.py, stresses.py, fatigue.py, buckling.py); configure mypy or pyright in pyproject.toml
- [ ] **Symbol naming consistency (9.3)** — standardize rod dimension names: `generate.py` uses `w_rod`/`t_rod`; `stresses.py` uses `width`/`thickness`; `buckling.py` uses `w`/`t`; pick one convention across all physics files
- [ ] **Remaining magic numbers (9.4)** — document or name: `1e-12` epsilon in `dynamics.py` (singular-matrix guard), `0.01` dedup tolerance in `optimize.py`, `0.808` in `fatigue.py` (Marin reliability factor)
- [ ] **Docstring format (9.5)** — enforce a consistent docstring style (Google or NumPy) across all modules in `src/mech390/`; physics modules have inconsistent or missing parameter docs
- [ ] **`min_wall_mm` unit clarity (6.2)** — `stress_analysis.min_wall_mm` is stored in mm in YAML but used internally in metres (multiplied by `1e-3` in stage2_embodiment.py and validate_candidate.py); rename to `min_wall_m` in YAML and remove the conversion, or add explicit `_mm` suffix comments everywhere it is read
- [ ] **Scaler path assumption in infer.py (8.6)** — `SurrogatePredictor` infers scaler path from checkpoint path using string replacement; brittle if directory layout changes; accept explicit `scaler_path` argument instead
- [ ] **`design_eval` god object (2.1 / 3.1)** — deferred by design: `design_eval` is a flat dict assembled in `generate.py` and consumed by the full physics stack; splitting it into typed sub-dicts (geometry, operating, material, etc.) would improve IDE support and catch key-name bugs at import time; tracked here for future refactor when the schema stabilises

### Postponed

- [ ] **Sign convention at Pin A** — `stresses.py:139–140` defines `F_r,crank,A` and `F_t,crank,A` with opposite signs to Mother Doc Eqs. 2.5–2.6. No numerical impact on current results because `abs()` wrapping is applied before all stress magnitude calculations in the crank stress path. Deferred until `abs()` wrapping is removed from the crank stress path; at that point the signs must be corrected to match the Mother Doc.
