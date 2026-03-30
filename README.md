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

A **physics-first data generation and ML pipeline** for the offset crank–slider mechanism.

Physics generates the data. ML learns pass/fail patterns from it. No ML shortcuts replace the physics.

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
           │ sigma_max, tau_max
           ▼
┌─────────────────────────┐
│  Pass / Fail Label      │  generate.py
│                         │  utilization = max(σ/σ_allow, τ/τ_allow)
│                         │  pass_fail = 1 if utilization ≤ 1.0
└──────────┬──────────────┘
           │ labeled dataset
           ▼
┌─────────────────────────┐
│  ML Training            │  ml/  ← ALL STUBS
│                         │  Classifier / regressor on pass/fail
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
│   ├── generate_dataset.py      # 🔲 STUB
│   ├── train_model.py           # 🔲 STUB
│   └── optimize_config.py       # 🔲 STUB
├── data/
│   ├── preview/
│   │   ├── stage1_geometries.csv  # 40 rows × 7 cols
│   │   ├── stage2_designs.csv     # 200 rows × 27 cols
│   │   └── forces_sweep.csv       # 4800 rows × 15 cols (200 designs × 24 angles)
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

# Preview scripts (all output to data/preview/ by default)
.venv/bin/python3 scripts/preview_stage1.py --out-dir data/preview
.venv/bin/python3 scripts/preview_stage2.py --out-dir data/preview
.venv/bin/python3 scripts/preview_forces.py --out-dir data/preview

# All scripts accept --config, --seed, --out-dir (and --max-2d for stage2/forces)
```

---

## 5. Known bugs

| # | File | Issue | Status |
|---|---|---|---|
| 1 | `generate.py` | `omega` and mass properties not injected before physics eval | ✅ Fixed — `bugfix/physics_corrections` |
| 2 | `kinematics.py:290` | Sign error on `alpha2` in rod angular acceleration | ✅ Fixed — `bugfix/physics_corrections` |
| 3 | `mass_properties.py:208` | Pin hole MOI offsets only exact for equal pin diameters | ✅ Fixed — `bugfix/physics_corrections` |
| 4 | `stage2_embodiment.py` | No post-rounding uniqueness check — duplicates possible | ⚠️ Open |

---

## 6. To-Do list

> **For agents and contributors:** This is the full project backlog. It spans many sessions.
> Complete one or two items per session — validate, commit, push, then stop.
> Physics changes → **Physics Validator**. Signature changes → **Cross-Reference Auditor**. New data → **Data Quality Checker**.

### High priority

- [ ] **Implement `generate_dataset.py` CLI** — argparse + `generate.generate_dataset()` + write `all_cases.csv` / `train_pass.csv`. Run Data Quality Checker after.
- [ ] **Fix post-rounding duplicates in Stage 2** — add seen-set check after rounding to prevent identical geometry rows.
- [ ] **Fill `configs/generate/aggressive.yaml`** — wider ranges, higher n_samples (reference: `baseline.yaml`)

### ML pipeline (not started)

- [ ] **`ml/features.py`** — drop zero-variance columns (`rho`, `I_area_slider_*`), scale features, flag leakage columns (`utilization`, `sigma_max`, `tau_max`)
- [ ] **`ml/models.py`** — binary classifier (Random Forest or XGBoost) for pass/fail + regressor for `utilization`. Use scikit-learn.
- [ ] **`ml/train.py`** — load dataset, apply feature pipeline, fit, evaluate, save to `data/models/`
- [ ] **`ml/infer.py`** — load saved model, accept design dict, return prediction + confidence
- [ ] **`train_model.py` CLI** — argparse wrapper around `ml/train.py`
- [ ] **Fill `configs/train/classifier.yaml` and `regression.yaml`**

### Physics — future work

- [ ] **`preview_stresses.py`** — like `preview_forces.py` but outputs σ/τ per angle
- [ ] **`optimize_config.py` CLI** — ML-based design space search; rank candidates by pass probability
- [ ] **Fill `configs/optimize/search.yaml`**

### Data and testing

- [ ] **Generate large dataset** — once stresses done, run with `n_samples ≥ 10000`, save to `data/raw/`
- [ ] **Run ML Readiness Inspector** before any training run
- [ ] **Expand `test_datagen_units.py`** — add dynamics, mass properties, and stress tests
- [ ] **Add regression tests** — fixed-seed full pipeline run vs reference snapshot
- [ ] **Populate `data/splits/`** — train/val/test split CSVs from full dataset

### Documentation

- [ ] **Add pipeline architecture diagram** to `assets/`
