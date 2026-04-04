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
| `data/preview/` | Three validated CSVs — quick sanity check of the pipeline |

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
| Link material | Aluminum |
| Link geometry | Rectangular cross-section |

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
           │ labeled dataset (71 cols)
           ▼
┌─────────────────────────┐
│  ML Training            │  ml/  ← STUBS (architecture decided)
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
| `n_static_rod/crank/pin` | Static FOS per link: σ_allow / peak σ_link | — |
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
│   │   ├── baseline.yaml        # Main config (40 samples, LHS, 5 variants/2D)
│   │   ├── test_small.yaml      # Fast test config
│   │   └── aggressive.yaml      # 🔲 EMPTY
│   ├── train/
│   │   ├── classifier.yaml      # 🔲 EMPTY
│   │   └── regression.yaml      # 🔲 EMPTY
│   └── optimize/
│       └── search.yaml          # 🔲 EMPTY
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
│   │   ├── stage2_embodiment.py # ✅ 3D expansion (stress calls are TODOs)
│   │   └── generate.py          # ✅ Orchestrator with pass/fail labeling
│   └── ml/
│       ├── features.py          # 🔲 EMPTY
│       ├── models.py            # 🔲 EMPTY
│       ├── train.py             # 🔲 EMPTY
│       └── infer.py             # 🔲 EMPTY
├── scripts/
│   ├── preview_stage1.py        # ✅ Stage 1 → CSV
│   ├── preview_stage2.py        # ✅ Stage 1 + Stage 2 + mass props → CSV
│   ├── preview_forces.py        # ✅ Full pipeline → force sweep (4800 rows)
│   ├── debug_stage1.py          # ✅ Quick debug runner
│   ├── test_datagen.py          # ✅ Inline generation test
│   ├── generate_dataset.py      # ✅ Full pipeline CLI → 7 CSVs
│   ├── train_model.py           # 🔲 STUB
│   └── optimize_config.py       # 🔲 STUB
├── data/
│   ├── preview/
│   │   ├── stage1_geometries.csv  # 40 rows × 7 cols
│   │   ├── stage2_designs.csv     # 200 rows × 27 cols
│   │   ├── forces_sweep.csv       # 4800 rows × 15 cols (200 designs × 24 angles)
│   │   ├── kinematics.csv         # 4800 rows — per (design, angle): positions, velocities, accels
│   │   ├── dynamics.csv           # 4800 rows — per (design, angle): joint forces, torque
│   │   ├── stresses.csv           # 4800 rows — per (design, angle): per-component σ, τ
│   │   ├── fatigue.csv            # 200 rows  — per design: Goodman / Miner metrics
│   │   ├── buckling.csv           # 200 rows  — per design: Euler buckling metrics
│   │   ├── passed_configs.csv     # N rows    — passing designs with all check columns
│   │   └── failed_configs.csv     # N rows    — failing designs with all check columns
│   ├── raw/        # Full generation runs (not yet populated)
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

- [ ] **`ml/features.py`** — drop zero-variance columns (`rho`, `I_area_slider_*`), scale features, flag leakage columns (`utilization`, `sigma_max`, `tau_max`)
- [ ] **`ml/models.py`** — NN architecture (see ML Architecture section below)
- [ ] **`ml/train.py`** — load dataset, apply feature pipeline, train/val/test split, hyperparameter tuning, evaluate; save to `data/models/`
- [ ] **`ml/infer.py`** — load saved model, accept weight vector + constraints, return optimal geometry
- [ ] **`train_model.py` CLI** — argparse wrapper around `ml/train.py`
- [ ] **Fill `configs/train/classifier.yaml` and `regression.yaml`**

### Optimization and visualization (Week 8)

- [ ] **`optimize_config.py` CLI** — use trained ML model as surrogate to sweep design space; rank candidates by pass probability, minimize envelope and motor power. Cross-check top candidates against physics engine.
- [ ] **Fill `configs/optimize/search.yaml`** — define search bounds and optimization objectives (minimize size/weight and required motor power)
- [ ] **Sensitivity plots** — plot each input feature vs `utilization` / `pass_fail` to show which design parameters matter most
- [ ] **Parameter correlation map** — heatmap of feature–target correlations across the full dataset

### Physics and testing

- [ ] **`preview_stresses.py`** — like `preview_forces.py` but outputs σ/τ per angle
- [ ] **Expand `test_datagen_units.py`** — add dynamics, mass properties, and stress tests
- [ ] **Add regression tests** — fixed-seed full pipeline run vs reference snapshot
