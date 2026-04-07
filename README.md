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
│  Optuna hyperpar. sweep │  Inputs: 10 design vars → pass_fail + 4 targets
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
| `pin_diameter_A/B/C` | Pin diameters at joints A, B, C | m |
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
│   ├── debug_stage1.py          # ✅ Quick debug runner
│   ├── test_datagen.py          # ✅ Inline generation test
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

| Bug | Location | Impact |
|---|---|---|
| Net-section fallback (`1e-9`) produces unphysical stresses (~TPa) when `pin_diameter + clearance ≥ link_width` | `stresses.py:218, 256, 358, 399` | Affects ~7–27 rows in `failed_configs` only. Labels correct (pass_fail = 0). Stress magnitudes meaningless — clip or log-transform stress features before ML training. Fix: reject degenerate geometry upstream in Stage 2. |

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

- [ ] **Run full training** — `python scripts/train_model.py --config configs/train/surrogate.yaml` (50 Optuna trials × up to 300 epochs). Smoke test used only 3 trials × 20 epochs — regression predictions are currently unreliable.
- [ ] **Validate regression quality** — after full training, confirm val R² > 0.85 for `total_mass`, `volume_envelope`, `tau_A_max`; val F1 > 0.90 for `pass_fail`
- [ ] **Cross-validate top-1 candidate** — run the best optimizer result through the full physics pipeline to confirm it physically passes all checks
- [ ] **Tune weight table** — edit `configs/optimize/search.yaml` objectives/weights based on design trade-offs for the final prototype

### Optimization and visualization (Week 8)

- [ ] **`scripts/visualize_design.py`** — CLI: reads a CSV (passed/failed_configs), renders a 2D mechanism drawing for a given `design_id`. Shows crank, rod, slider with correct cross-section widths, pin circles at A/B/C, guide rail, and annotates pass/fail + key metrics (ROM, QRR, n_f, n_buck). Args: `--csv`, `--id`, `--angle` (default: extended position). ~130 lines.
- [ ] **Sensitivity plots** — plot each input feature vs `utilization` / `pass_fail`
- [ ] **Parameter correlation map** — heatmap of feature–target correlations

### Physics and testing

- [x] **Size factor formula check** — replaced Shigley piecewise with Mott Table 5-3; split `C_s` (size) and `C_sur` (manufacturing method) per Mott naming
- [x] **Sn verified** — 133 MPa at design life 18.72×10⁶ cycles (Al 2024-T3, ASM)
- [x] **Fix Basquin exponent denominator** — replaced synthetic steel-origin formula with experimental AA2024-T3 constants: `σa = 924·N^(−0.086)` from anchors (10⁷, 230 MPa) + (10⁹, 155 MPa)
- [ ] **`preview_stresses.py`** — like `preview_forces.py` but outputs σ/τ per angle
- [ ] **Expand `test_datagen_units.py`** — add dynamics, mass properties, and stress tests
- [ ] **Add regression tests** — fixed-seed full pipeline run vs reference snapshot
