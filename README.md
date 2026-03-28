# Data-Driven Design of an Offset Crank–Slider Mechanism

**MECH 390 – Machine Learning–Assisted Mechanical Design**

---

## For AI agents and teammates — read this first

This project uses **Claude Code** as its AI assistant (via `CLAUDE.md`).
Claude is pre-configured with four subagents that run automatically based on your request.
You do not need to know Python or mechanics to contribute.

**Setup (one time):**
1. Install [Claude Code](https://claude.ai/code)
2. Open this repository in VS Code or terminal
3. Run Claude Code — it automatically reads `CLAUDE.md` and loads all agent rules

**Making a request (just use plain English):**

| What you say | What Claude does automatically |
|---|---|
| "I changed the rod formula" | Physics Validator + Cross-Reference Auditor |
| "Run the data generation and check it" | Runs pipeline, then Data Quality Checker |
| "Is the dataset ready to train?" | ML Readiness Inspector |
| "Everything looks wrong, check the pipeline" | All four agents in parallel |
| "Add a new feature to stage 2" | Plan agent → implement → Cross-Reference Auditor |

**Key files every agent/person should know:**

| File | What it is |
|---|---|
| `CLAUDE.md` | Defines the four subagents and mandatory rules — read first |
| `instructions.md` | Authoritative physics derivations and full technical spec |
| `configs/generate/baseline.yaml` | Controls how data is generated (geometry, sampling, limits) |
| `data/preview/` | Preview CSV outputs — quick sanity check of the pipeline |

> **Note for AI agents:** The To-Do list at the bottom of this file is the authoritative task backlog.
> It is intentionally large. Do NOT attempt to complete the full list in one session.
> Pick a specific task, complete it, validate it with the appropriate subagent(s), and stop.

---

## 1. What this project is

This repository implements a **physics-first data generation and ML workflow** for the design
and evaluation of an **offset crank–slider mechanism**.

The goal is to automate what is traditionally done by hand in mechanical design:

1. Select a mechanism geometry
2. Verify it satisfies motion requirements (ROM and QRR)
3. Evaluate forces and stresses over a full crank cycle
4. Determine whether the design is structurally acceptable (pass/fail)
5. Train a machine learning model to rapidly predict pass/fail for new designs

Physics governs the system. ML learns patterns from physically valid data.
No ML shortcuts replace the physics.

---

## 2. System architecture

```
configs/generate/baseline.yaml
         │
         ▼
┌─────────────────────────┐
│  Stage 1 – 2D Kinematic │  ← stage1_kinematic.py
│  Screening              │    Samples (r, l, e), solves ROM constraint,
│                         │    checks QRR, rejects invalid designs
└──────────┬──────────────┘
           │ valid (r, l, e) tuples
           ▼
┌─────────────────────────┐
│  Stage 2 – 3D Embodiment│  ← stage2_embodiment.py
│  Expansion              │    Adds widths, thicknesses, pin diameters
│                         │    (multiple variants per 2D design)
└──────────┬──────────────┘
           │ 3D design dicts
           ▼
┌─────────────────────────┐
│  Mass Properties        │  ← mass_properties.py
│                         │    Computes masses, COGs, MOIs for each link
└──────────┬──────────────┘
           │ design + mass props
           ▼
┌─────────────────────────┐
│  Physics Engine         │  ← engine.py + dynamics.py
│  (15° sweep)            │    Solves Newton-Euler 8×8 system at each angle
│                         │    Returns F_A, F_B, F_C, N, F_f, tau_A
└──────────┬──────────────┘
           │ force history
           ▼
┌─────────────────────────┐
│  Stress Evaluation      │  ← stresses.py  ← STUB (not yet implemented)
│                         │    Normal + shear stress from joint forces
└──────────┬──────────────┘
           │ sigma_max, tau_max
           ▼
┌─────────────────────────┐
│  Pass / Fail Label      │  ← generate.py
│                         │    utilization = max(σ/σ_allow, τ/τ_allow)
│                         │    pass_fail = 1 if utilization ≤ 1.0
└──────────┬──────────────┘
           │ labeled dataset
           ▼
┌─────────────────────────┐
│  ML Training            │  ← ml/  ← ALL STUBS (not yet implemented)
│                         │    Classifier or regressor on pass/fail
└─────────────────────────┘
```

---

## 3. Stage 1 — 2D kinematic synthesis

Only geometry and motion are evaluated here. No forces, masses, or stresses.

### Hard constraints enforced

| Constraint | Value | Meaning |
|---|---|---|
| ROM target | 250 mm | Slider stroke must equal this |
| ROM tolerance | ±0.5 mm | Acceptance band |
| QRR bounds | [1.5, 2.5] | Quick-return ratio range |
| RPM | 30 | Fixed — not a design variable |

### Process

1. Pre-filter feasible `(l, e)` domain (avoid wasted samples)
2. Sample `(l, e)` via Latin Hypercube Sampling (or random)
3. Solve `r` analytically from ROM constraint:
   ```
   r = (S/2) · √( (4(l²−e²) − S²) / (4l² − S²) )
   ```
4. Branch-feasibility checks: `l > r + |e|`, residual ROM check
5. Find dead-center angles via Brent root-finding
6. Compute QRR from crank-angle spans
7. Accept design if ROM tolerance and QRR bounds both satisfied

Output: streaming iterator of `(r, l, e, ROM, QRR, theta_min, theta_max)` dicts.

---

## 4. Stage 2 — 3D embodiment, dynamics, pass/fail

Only designs that passed Stage 1 reach here.

### Process

1. Generate `n_variants_per_2d` 3D geometry variants per 2D design
   (widths, thicknesses, pin diameters sampled from config ranges)
2. Enforce geometric feasibility: `width_r > pin_diameter_A/B`, `width_l > pin_diameter_B/C`
3. Compute mass properties: link/slider masses, COGs, mass MOIs, area MOIs
4. Dynamics sweep at 15° increments (24 angles per design):
   - Solve Newton–Euler 8×8 linear system
   - Outputs: `F_A`, `F_B`, `F_C`, `N`, `F_f`, `tau_A`
5. *(Not yet)* Compute stresses from joint forces
6. *(Not yet)* Apply pass/fail: `utilization = max(σ/σ_allow, τ/τ_allow)`

---

## 5. Physics model summary

### Kinematic variables

| Symbol | Code key | Meaning | Unit |
|---|---|---|---|
| r | `r` | Crank radius | m |
| l | `l` | Connecting rod length | m |
| D | `e` | Offset (slider line to crank pivot) | m |
| θ | `theta` | Crank angle | rad |
| ω | `omega` | Crank angular speed = RPM × 2π/60 | rad/s |
| φ | `phi` | Rod angle | rad |
| ROM / S | `ROM` | Slider stroke | m |
| QRR | `QRR` | Quick-return ratio = Δθ_forward / Δθ_return | — |

### 3D embodiment variables

| Code key | Meaning | Unit |
|---|---|---|
| `width_r`, `thickness_r` | Crank link cross-section | m |
| `width_l`, `thickness_l` | Rod link cross-section | m |
| `pin_diameter_A/B/C` | Pin diameters at joints A, B, C | m |
| `rho` | Material density (steel: 7800 kg/m³) | kg/m³ |

### Newton–Euler 8×8 system

Unknown vector at each crank angle:

```
x = [F_Ax, F_Ay, F_Bx, F_By, F_Cx, F_Cy, N, tau_A]
```

Eight equations from:
1. Crank Fx balance
2. Crank Fy balance (gravity)
3. Crank moment about CG (includes tau_A)
4. Rod Fx balance
5. Rod Fy balance (gravity)
6. Rod moment about CG
7. Slider Fx (Coulomb friction: `−μ·sign(v_sx)·N`)
8. Slider Fy (guide normal N)

### Pass/fail criteria

```
utilization = max(sigma_max / sigma_allow,  tau_max / tau_allow)
pass_fail   = 1 if utilization ≤ safety_factor, else 0
```

Current limits (from `baseline.yaml`): σ_allow = 180 MPa, τ_allow = 100 MPa, safety_factor = 1.0.

---

## 6. Repository structure

```
MECH390_ML_Model/
│
├── CLAUDE.md                    # Agent definitions + mandatory rules (read first)
├── instructions.md              # Full technical spec and authoritative physics derivations
├── README.md                    # This file
│
├── configs/
│   ├── generate/
│   │   ├── baseline.yaml        # Main config — 40 samples, LHS, 5 variants/2D
│   │   ├── test_small.yaml      # Small test config
│   │   └── aggressive.yaml      # Wide-range config — EMPTY STUB
│   ├── train/
│   │   ├── classifier.yaml      # ML classifier config — EMPTY STUB
│   │   └── regression.yaml      # ML regression config — EMPTY STUB
│   └── optimize/
│       └── search.yaml          # Design optimization config — EMPTY STUB
│
├── src/mech390/
│   ├── config.py                # Config loading, normalization, validation
│   ├── physics/
│   │   ├── kinematics.py        # ✅ Position/velocity/acceleration, dead centers, ROM/QRR
│   │   ├── dynamics.py          # ✅ Newton-Euler 8×8 solver, joint reactions
│   │   ├── mass_properties.py   # ✅ Link/slider mass, MOI, COG helpers
│   │   ├── engine.py            # ✅ 15° sweep orchestrator (stresses placeholder)
│   │   ├── stresses.py          # 🔲 STUB — stress formulas not implemented
│   │   ├── fatigue.py           # 🔲 EMPTY — reserved for future
│   │   └── buckling.py          # 🔲 EMPTY — reserved for future
│   ├── datagen/
│   │   ├── sampling.py          # ✅ LHS sampler, scalar sampler, factory
│   │   ├── stage1_kinematic.py  # ✅ Full Stage 1 pipeline (streaming iterator)
│   │   ├── stage2_embodiment.py # ✅ 3D expansion (mass/stress calls are TODOs)
│   │   └── generate.py          # ✅ Dataset generation orchestrator
│   └── ml/
│       ├── features.py          # 🔲 EMPTY — feature selection and scaling
│       ├── models.py            # 🔲 EMPTY — ML model architectures
│       ├── train.py             # 🔲 EMPTY — training loop
│       └── infer.py             # 🔲 EMPTY — inference / prediction
│
├── scripts/
│   ├── preview_stage1.py        # ✅ CLI: Stage 1 → CSV in data/preview/
│   ├── preview_stage2.py        # ✅ CLI: Stage 1 → Stage 2 + mass props → CSV
│   ├── preview_forces.py        # ✅ CLI: Full pipeline → force sweep CSV (4800 rows)
│   ├── debug_stage1.py          # ✅ Quick debug runner for Stage 1
│   ├── test_datagen.py          # ✅ Inline config data generation test
│   ├── generate_dataset.py      # 🔲 STUB — needs argparse + generate_dataset() call
│   ├── train_model.py           # 🔲 STUB — empty
│   └── optimize_config.py       # 🔲 STUB — empty
│
├── data/
│   ├── preview/                 # ← All preview outputs go here
│   │   ├── stage1_geometries.csv    # 40 rows × 7 cols
│   │   ├── stage2_designs.csv       # 200 rows × 27 cols
│   │   └── forces_sweep.csv         # 4800 rows × 15 cols (200 designs × 24 angles)
│   ├── raw/                     # Full generation runs (not yet populated)
│   ├── processed/               # Post-processed datasets
│   ├── splits/                  # Train/val/test splits
│   └── models/                  # Trained model artifacts
│
└── tests/
    └── test_datagen_units.py    # Unit tests: kinematics, datagen, imports
```

---

## 7. How to run the preview scripts

```bash
# Activate the virtual environment (create it first if needed)
python3 -m venv .venv
.venv/bin/pip install pandas scipy numpy pyyaml

# Stage 1: 2D kinematic screening
.venv/bin/python3 scripts/preview_stage1.py --out-dir data/preview

# Stage 2: 3D embodiment + mass properties
.venv/bin/python3 scripts/preview_stage2.py --out-dir data/preview

# Forces: full pipeline + joint reaction forces at every 15°
.venv/bin/python3 scripts/preview_forces.py --out-dir data/preview

# Optional overrides
.venv/bin/python3 scripts/preview_forces.py \
    --config configs/generate/baseline.yaml \
    --seed 123 \
    --out-dir data/preview \
    --max-2d 10
```

---

## 8. Current implementation status

| Module | Status | Notes |
|---|---|---|
| `config.py` | ✅ Complete | Loads YAML, normalizes numerics, validates ranges |
| `kinematics.py` | ✅ Complete | All position/velocity/acceleration functions; dead centers; ROM & QRR |
| `dynamics.py` | ✅ Complete | Newton-Euler 8×8 solver; returns F_A/B/C, N, F_f, tau_A; condition guard |
| `mass_properties.py` | ✅ Complete | Link/slider mass, COG, mass MOI, area MOI, design-level aggregator |
| `engine.py` | ✅ Complete | 15° sweep; stresses plugged as 0.0 placeholder |
| `stresses.py` | 🔲 Stub | Only a comment; stress formulas not written |
| `fatigue.py` | 🔲 Empty | Reserved for future work |
| `buckling.py` | 🔲 Empty | Reserved for future work |
| `sampling.py` | ✅ Complete | LHS via scipy.stats.qmc; random; factory |
| `stage1_kinematic.py` | ✅ Complete | Constrained (l,e) sampling; closed-form r solver; streaming iterator |
| `stage2_embodiment.py` | ✅ Complete | Streaming 3D expansion; width/pin constraints; mass/stress calls are TODOs |
| `generate.py` | ✅ Complete | Orchestrator; omega + mass props injected before physics; pass/fail labeling |
| `ml/` (all 4 files) | 🔲 Empty | Not started |
| `preview_stage1.py` | ✅ Complete | CLI with --config, --seed, --out-dir |
| `preview_stage2.py` | ✅ Complete | CLI with --config, --seed, --out-dir, --max-2d |
| `preview_forces.py` | ✅ Complete | CLI; one row per (design, angle); all 9 force/torque outputs |
| `generate_dataset.py` | 🔲 Stub | Imports only |
| `train_model.py` | 🔲 Stub | Empty |
| `optimize_config.py` | 🔲 Stub | Empty |
| `configs/generate/aggressive.yaml` | 🔲 Empty | Config stub only |
| `configs/train/*.yaml` | 🔲 Empty | Config stubs only |
| `configs/optimize/search.yaml` | 🔲 Empty | Config stub only |

---

## 9. Known bugs and fixes

| # | File | Issue | Status |
|---|---|---|---|
| 1 | `generate.py` | `omega` and mass properties not injected before physics evaluation → forces ~10× wrong | ✅ Fixed in `bugfix/physics_corrections` |
| 2 | `kinematics.py:290` | Sign error: `a_By = -alpha2·r·cos(θ)` should be `+alpha2·r·cos(θ)` | ✅ Fixed in `bugfix/physics_corrections` |
| 3 | `mass_properties.py:208-209` | Pin hole MOI offsets assumed `±c/2`; only exact when `d_left == d_right` | ✅ Fixed in `bugfix/physics_corrections` |
| 4 | `stage2_embodiment.py` | No post-rounding uniqueness check; duplicate geometry rows possible after mm rounding | ⚠️ Known, not yet fixed |

---

## 10. Data preview outputs

Three preview CSVs in `data/preview/` (all validated by Data Quality Checker — PASS):

### `stage1_geometries.csv` — 40 rows × 7 columns

Kinematic screening output. One row per valid 2D design.

| Column | Description | Unit |
|---|---|---|
| r | Crank radius | m |
| l | Connecting rod length | m |
| e | Offset | m |
| ROM | Computed slider stroke (target: 0.25 m ± 0.5 mm) | m |
| QRR | Quick-return ratio (bounds: [1.5, 2.5]) | — |
| theta_min | Crank angle at retracted position | rad |
| theta_max | Crank angle at extended position | rad |

### `stage2_designs.csv` — 200 rows × 27 columns

3D embodiment output. One row per design variant (40 × 5 variants).
All Stage 1 columns plus: `width_r, width_l, thickness_r, thickness_l, pin_diameter_A/B/C, rho, mass_crank, mass_rod, mass_slider, I_mass_*_cg_z (×3), I_area_*_yy/zz (×6)`.

### `forces_sweep.csv` — 4800 rows × 15 columns

Dynamics sweep output. One row per (design, crank angle) — 200 designs × 24 angles.

| Column | Description | Unit |
|---|---|---|
| design_index | Design identifier (1–200) | — |
| r, l, e | 2D geometry (inherited) | m |
| theta_deg | Crank angle | ° |
| theta_rad | Crank angle | rad |
| F_Ax, F_Ay | Joint A reaction (at crank pivot) | N |
| F_Bx, F_By | Joint B reaction (at crank-rod pin) | N |
| F_Cx, F_Cy | Joint C reaction (at rod-slider pin) | N |
| N | Slider guide normal force | N |
| F_f | Slider friction force (Coulomb) | N |
| tau_A | Required crank drive torque | N·m |

---

## 11. To-Do list

> **For agents and contributors:** This list represents the full backlog of outstanding work.
> It is intentionally long and spans multiple sessions. Do not attempt to complete more than
> one or two items per session. Each item should be completed, validated with the appropriate
> subagent(s), committed, and pushed before moving to the next.
> Physics items require the **Physics Validator**. Signature changes require the
> **Cross-Reference Auditor**. New data outputs require the **Data Quality Checker**.

### High priority — unblocks everything else

- [ ] **Merge `bugfix/physics_corrections` into `main`** — three confirmed physics bugs are fixed on this branch; main is currently running broken physics
- [ ] **Implement `stresses.py`** — normal stress σ = F/A and bending stress σ = M·c/I for crank and rod links; shear stress τ = VQ/Ib at pin locations. This is the single biggest blocker for real pass/fail labels. Read `instructions.md` §6 and §7 before implementing. Run Physics Validator after.
- [ ] **Wire stress calls in `engine.py`** — replace the `sigma, tau = 0.0, 0.0` placeholder with `stresses.evaluate(design, F_B, F_C, ...)`. Run Physics Validator + Cross-Reference Auditor after.
- [ ] **Wire stress calls in `stage2_embodiment.py`** — the TODO stubs for mass properties and stress in the expansion loop need to call `compute_design_mass_properties` and the stress evaluator. Run Cross-Reference Auditor after.

### Medium priority — core pipeline completion

- [ ] **Implement `generate_dataset.py` CLI** — add argparse with `--config`, `--seed`, `--out-dir`, `--n-samples`; call `generate.generate_dataset(config, seed)`; write `all_cases.csv` and `train_pass.csv`. Run Data Quality Checker after.
- [ ] **Fix post-rounding duplicate issue in Stage 2** — `stage2_embodiment.py` can produce identical cross-section geometry after 1 mm rounding for some designs (confirmed, rows 171–172 in previous run). Add a seen-set check after rounding to enforce uniqueness.
- [ ] **Fill `configs/generate/aggressive.yaml`** — wider geometry ranges and higher n_samples for a full-scale production run (reference: `baseline.yaml`)
- [ ] **Update `instructions.md` Known Bugs table** — bugs 1–3 are now fixed but the table still shows them as open. Update to reflect the fixed status.

### ML pipeline — not started

- [ ] **Implement `ml/features.py`** — feature selection (drop `rho`, `I_area_slider_*` which have near-zero variance), scaling (StandardScaler or MinMaxScaler), train/val/test split logic. Flag potential data leakage columns (`utilization`, `sigma_max`, `tau_max`).
- [ ] **Implement `ml/models.py`** — define at least two model architectures: (1) binary classifier (Random Forest or XGBoost) for pass/fail; (2) regressor for `utilization`. Use scikit-learn as the base library.
- [ ] **Implement `ml/train.py`** — training loop: load dataset, apply features pipeline, fit model, evaluate on validation set, save model artifact to `data/models/`.
- [ ] **Implement `ml/infer.py`** — load saved model, accept a design dict, return predicted pass/fail + confidence.
- [ ] **Implement `train_model.py` CLI** — argparse wrapper around `ml/train.py`. Args: `--dataset`, `--model-type`, `--config`, `--out-dir`.
- [ ] **Fill `configs/train/classifier.yaml`** — hyperparameters for classifier (e.g. n_estimators, max_depth, learning_rate)
- [ ] **Fill `configs/train/regression.yaml`** — hyperparameters for utilization regressor

### Optimization — future work

- [ ] **Implement `optimize_config.py` CLI** — use trained ML model for rapid design space search; generate candidate designs and rank by predicted pass probability.
- [ ] **Fill `configs/optimize/search.yaml`** — define search bounds and objective

### Physics — future work

- [ ] **Implement `fatigue.py`** — Goodman or Miner's rule fatigue analysis using `TotalCycles` from config and alternating/mean stress from the sweep
- [ ] **Implement `buckling.py`** — Euler column buckling check for the connecting rod under compressive loads
- [ ] **Add `preview_stresses.py` script** — similar to `preview_forces.py` but outputs `sigma`, `tau` at every 15° once `stresses.py` is implemented

### Data and testing

- [ ] **Generate a large dataset** — once stresses.py is done, run `generate_dataset.py` with `n_samples ≥ 10000` and save to `data/raw/`
- [ ] **Run ML Readiness Inspector** on the large dataset before training
- [ ] **Expand unit tests in `test_datagen_units.py`** — add tests for dynamics (forces at known angles), mass properties (known geometry), and stress evaluation once implemented
- [ ] **Add regression tests** — run full pipeline with fixed seed and assert CSV outputs match a reference snapshot to catch regressions
- [ ] **Populate `data/splits/`** — create train/val/test split CSVs from the full dataset

### Documentation

- [ ] **Update `instructions.md` with stress formulas** — once stresses.py is implemented, add the exact equations used (with signs, units, cross-section conventions) to maintain the authoritative spec
- [ ] **Add assets/architecture diagram** — a visual diagram of the pipeline would help onboarding

---

## 12. Active branches

| Branch | Purpose | Status |
|---|---|---|
| `main` | Production-stable code | Has open physics bugs (see Known Bugs) |
| `bugfix/physics_corrections` | Fixes for bugs 1–3 | Ready to merge into main |
| `bugfix/decrease_sample_size` | Reduced n_samples for fast testing | Merged into current work |
| `feature/example` | Example feature branch | Status unknown |
