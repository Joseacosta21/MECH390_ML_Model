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

### One-time setup

```bash
python3 -m venv .venv
.venv/bin/pip install pandas scipy numpy pyyaml torch optuna scikit-learn
```

---

### Full pipeline (recommended)

Runs all three steps in sequence: data generation → surrogate training → design optimization.

```bash
# All defaults — reads baseline.yaml, writes to data/preview/, saves model to data/models/
.venv/bin/python scripts/run_pipeline.py

# With explicit options
.venv/bin/python scripts/run_pipeline.py \
    --generate-config configs/generate/baseline.yaml \
    --train-config    configs/train/surrogate.yaml \
    --optimize-config configs/optimize/search.yaml \
    --seed            42 \
    --out-dir         data/preview

# Skip steps if outputs already exist
.venv/bin/python scripts/run_pipeline.py --skip-datagen           # skip data generation
.venv/bin/python scripts/run_pipeline.py --skip-datagen --skip-training  # optimize only
```

---

### Step by step

**Step 1 — Data generation** (physics simulation → 7 CSVs)

```bash
# All defaults (reads baseline.yaml, seed from config, writes to data/preview/)
.venv/bin/python scripts/generate_dataset.py

# With explicit options
.venv/bin/python scripts/generate_dataset.py \
    --config  configs/generate/baseline.yaml \
    --seed    42 \
    --out-dir data/preview
```

Writes to `--out-dir`: `kinematics.csv`, `dynamics.csv`, `stresses.csv`, `fatigue.csv`,
`buckling.csv`, `passed_configs.csv`, `failed_configs.csv`.

**Step 2 — Surrogate training** (Optuna sweep → checkpoint)

```bash
# All defaults (reads surrogate.yaml, saves checkpoint to data/models/)
.venv/bin/python scripts/train_model.py

# With explicit options
.venv/bin/python scripts/train_model.py \
    --config configs/train/surrogate.yaml \
    --seed   42
```

Saves to `data/models/`: `surrogate_best.pt`, `scaler.pkl`, `target_stats.json`.

**Step 3 — Design optimization** (differential evolution → top-N candidates)

```bash
# All defaults
.venv/bin/python scripts/optimize_design.py

# With explicit options
.venv/bin/python scripts/optimize_design.py \
    --generate-config configs/generate/baseline.yaml \
    --optimize-config configs/optimize/search.yaml \
    --model           data/models/surrogate_best.pt
```

**Validate a specific candidate geometry** (bypasses Stage 1, runs full physics)

```bash
# Edit CANDIDATE dict at the top of the script first, then:
.venv/bin/python scripts/validate_candidate.py --config configs/generate/baseline.yaml
```

---

### Preview scripts (individual pipeline stages)

```bash
.venv/bin/python scripts/preview_stage1.py --out-dir data/preview   # 2D kinematics only
.venv/bin/python scripts/preview_stage2.py --out-dir data/preview   # + 3D embodiment
.venv/bin/python scripts/preview_forces.py --out-dir data/preview   # + force sweep
```

All preview scripts accept `--config`, `--seed`, `--out-dir` (and `--max-2d` for stage2/forces).

---

## 5. Known bugs

No open bugs.

---

## 6. To-Do list

> **For agents and contributors:** Open items only. Physics changes → **Physics Validator**. Signature changes → **Cross-Reference Auditor**. New data → **Data Quality Checker**.

### Dataset generation

- [ ] **Fill `configs/generate/aggressive.yaml`** — wider ranges, higher n_samples (reference: `baseline.yaml`)
- [ ] **Populate `data/splits/`** — train/val/test split CSVs from the full dataset

### ML pipeline

- [ ] **Split regression training (ML-P8)** — classification head trains on all data (pass + fail); regression heads should train only on passing rows. Regression targets are only physically meaningful for passing designs — training on fail configs adds noise to the regression heads. Implementation: add a second filtered DataLoader (`pass_fail == 1`) and feed it to the regression MSE step only. The pass_prob gate in the optimizer still relies on the classification head seeing both classes.


- [ ] **Remove `min_n_static` from optimizer objectives (ML-P3)** — `search.yaml` still lists `min_n_static`; designs with thin fatigue-failing cranks show paradoxically high static FOS, inverting the optimization direction; remove it and redistribute its weight to `total_mass` and `volume_envelope`

- [ ] **Refactor features (ML-P4)** — *originally written for 200-row dataset; re-evaluate at 10k rows.* Consider dropping `n_buck`, `utilization`, `min_n_static` from `REGRESSION_TARGETS` (n_buck enforced analytically in optimizer); consider adding `slenderness_r = r/thickness_r`, `slenderness_l = l/thickness_l`, `net_section_r = width_r - d_shaft_A` to `INPUT_FEATURES`

- [ ] **Update model defaults after ML-P4 (ML-P5)** — after ML-P4 is applied, update `input_dim` and `n_reg_targets` defaults in `models.py`; note `train.py` already derives these dynamically from `features.py` so defaults are only fallbacks

- [ ] **Restrict architecture search space (ML-P6)** — *originally written for 200-row dataset; re-evaluate at 10k rows.* `[256, 128, 64]` is the current best-performing architecture and may remain appropriate

- [ ] **Add learning rate schedule (ML-P7)** — `train.py`: add `CosineAnnealingLR(optimizer, T_max=150)` after Adam definition

- [ ] **Validate regression quality** — val R² not yet confirmed for `total_mass`, `volume_envelope`, `tau_A_max`, `n_buck`, `utilization`

- [ ] **Fix target normalisation (8.1)** — target stats computed on train set; values outside training range are silently clipped; add explicit out-of-range warning or switch to StandardScaler

- [ ] **Stratify regression targets in split (8.2)** — train/val split only stratifies on `pass_fail`; add warning if any regression target distribution is severely skewed across splits

- [ ] **Add checkpoint version field (8.5)** — add a `version` key to `save_checkpoint()` and validate it in `build_model_from_hparams()` so schema changes are caught at load time

### Optimization and visualization

- [ ] **`scripts/visualize_design.py`** — 2D mechanism drawing from design_id: crank, rod, slider with cross-section widths, pin circles at A/B/C, guide rail, annotated metrics
- [ ] **Sensitivity plots** — each input feature vs `utilization` / `pass_fail`
- [ ] **Parameter correlation map** — heatmap of feature–target correlations

### Physics and testing

- [ ] **`preview_stresses.py`** — like `preview_forces.py` but outputs σ/τ per angle
- [ ] **Expand `test_datagen_units.py`** — add dynamics, mass properties, and stress tests
- [ ] **Add regression tests** — fixed-seed full pipeline run vs reference snapshot
- [ ] **Physics module tests** — unit tests for dynamics.py (Newton-Euler solver), stresses.py (rod/crank/pin formulas), fatigue.py (Goodman + Miner), buckling.py (critical load), mass_properties.py (mass/inertia formulas), engine.py (full sweep integration) — currently 0 tests
- [ ] **ML stack tests** — tests for features.py (normalisation round-trip), models.py (forward pass shapes), infer.py (load + predict cycle), optimize.py (constraint satisfaction)
- [ ] **Test infrastructure** — add pytest.ini or pyproject.toml test config; add shared fixtures
- [ ] **Integration / regression test** — fixed-seed config → dataset → at least N passing designs; run as CI smoke test

### Code hygiene

- [ ] **Unified logging (9.1)** — replace remaining `print()` calls in `generate_dataset.py:136–137` with `logger.info()`; ensure all entry-point scripts configure a root logger
- [ ] **Type annotations (9.2)** — add type hints to physics modules (dynamics.py, stresses.py, fatigue.py, buckling.py); configure mypy or pyright in pyproject.toml
- [ ] **Symbol naming consistency (9.3)** — standardize rod dimension names: `generate.py` uses `w_rod`/`t_rod`; `stresses.py` uses `width`/`thickness`; `buckling.py` uses `w`/`t`; pick one convention across all physics files
- [ ] **Remaining magic numbers (9.4)** — document or name: `1e-12` epsilon in `dynamics.py` (singular-matrix guard), `0.01` dedup tolerance in `optimize.py`, `0.808` in `fatigue.py` (Marin reliability factor)
- [ ] **Docstring format (9.5)** — enforce a consistent docstring style (Google or NumPy) across all modules in `src/mech390/`; physics modules have inconsistent or missing parameter docs
- [ ] **`min_wall_mm` unit clarity (6.2)** — `stress_analysis.min_wall_mm` is stored in mm in YAML but used internally in metres; rename to `min_wall_m` in YAML and remove the conversion, or add explicit `_mm` suffix comments everywhere it is read
- [ ] **Scaler path assumption in infer.py (8.6)** — `SurrogatePredictor` infers scaler path from checkpoint path using string replacement; brittle if directory layout changes; accept explicit `scaler_path` argument instead
- [ ] **`design_eval` god object (2.1 / 3.1)** — deferred: `design_eval` is a flat dict assembled in `generate.py` and consumed by the full physics stack; splitting it into typed sub-dicts would improve IDE support; tracked for future refactor when schema stabilises

### Postponed

- [ ] **Sign convention at Pin A** — `stresses.py:139–140` defines `F_r,crank,A` and `F_t,crank,A` with opposite signs to Mother Doc Eqs. 2.5–2.6. No numerical impact while `abs()` wrapping is applied throughout the crank stress path. Deferred until `abs()` wrapping is removed.

---

## 7. Current Status

> **Note:** Dataset size is intentionally variable. The pipeline is designed so that changing `n_samples` and `n_variants_per_2d` in `configs/generate/baseline.yaml` and re-running `generate_dataset.py` is all that is required to produce a larger or smaller dataset. The ML stack reads `len(INPUT_FEATURES)` and `len(REGRESSION_TARGETS)` dynamically — no hardcoded counts anywhere in the training path. Retrain after any dataset change.

### Dataset (latest run snapshot)

| Item | Value |
|---|---|
| Total rows | 10,000 (will change with config) |
| Pass | 7,526 (75.3%) |
| Fail | 2,474 (24.7%) |
| Source config | `configs/generate/baseline.yaml` |
| Post-rounding dedup | Applied in `generate.py` before CSV write; count reported in summary |

### Trained surrogate model (`data/models/`) — snapshot from last training run

| Item | Value |
|---|---|
| Best val F1 | **0.9713** |
| Architecture | `[256, 128, 64]` shared ReLU trunk + classification head + 8-target regression head |
| Dropout | 0.299 |
| Learning rate | 9.17 × 10⁻³ |
| Batch size | 128 |
| Optuna sweep | 50 trials × 300 epochs max, patience=25 |
| Early stopping | On `val_f1` (not `val_loss`) |
| Checkpoint | `data/models/surrogate_best.pt` |
| Scaler | `data/models/scaler.pkl` |
| Target stats | `data/models/target_stats.json` |

**Permanent architecture decisions (not dataset-size-dependent):**
- Early stopping on `val_f1` — saves the best-classified model; regression loss can continue falling after classification peaks, so stopping on `val_loss` discards the optimal pass/fail epoch
- `input_dim` and `n_reg_targets` derived from `len(F.INPUT_FEATURES)` and `len(F.REGRESSION_TARGETS)` in `train.py` — adding or removing features never requires touching `train.py` or `models.py`
- `infer.py` asserts `ckpt['hparams']['n_reg_targets'] == len(F.REGRESSION_TARGETS)` at load — stale checkpoints are caught immediately, not silently

### Optimizer findings

- Historical gap (pre ML-P2): Rank 1 optimizer candidate scored 0.97 on surrogate but 0.48 on physics — root cause was OOD predictions being silently clamped to [0, 1], giving free maximum scores
- **ML-P2 implemented:** OOD predictions now penalised proportionally to weight × excess × `ood_penalty_scale`. A prediction 3× the training max produces a score deduction that dominates the entire objective, eliminating that candidate. Parameters configurable in `search.yaml` (`ood_tolerance: 0.1`, `ood_penalty_scale: 10.0`)
- Best design found in training data (design 126, score=0.85) outperforms previous optimizer Rank 1 in physics validation
- Three analytical constraints in `optimize.py` mitigate worst cases (see CLAUDE.md)

### Physics verifications confirmed

| Item | Confirmed value | Source |
|---|---|---|
| Fatigue strength `Sn` | 133 MPa at 18.72×10⁶ cycles | Al 2024-T3, ASM |
| Basquin S-N constants | σa = 924 · N^(−0.086) | AA2024-T3 experimental anchors: (10⁷, 230 MPa) + (10⁹, 155 MPa) |
| Size factor `C_s` | Mott Table 5-3 (split from `C_sur`) | `C_sur` = 0.88 as-machined |

> Full derivations and code contracts for all physics quantities: see `instructions.md`.
