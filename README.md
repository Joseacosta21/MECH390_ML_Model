# Data-Driven Design of an Offset Crank-Slider Mechanism

**MECH 390 - Machine Learning-Assisted Mechanical Design**

---

## 1. What this project is

A physics-first data generation and ML pipeline for the offset crank-slider mechanism, built for the MECH 390 Winter 2026 design project at Concordia University.

Physics generates the data. ML learns pass/fail patterns from it.

**Design specifications (fixed):**

| Spec                         | Value                                                             |
| ---------------------------- | ----------------------------------------------------------------- |
| Reaction force (slider load) | 500 g (~4.905 N)                                                  |
| Range of motion (ROM target) | 250 mm +- 0.5 mm                                                  |
| Input speed                  | 30 RPM                                                            |
| Quick return ratio (QRR)     | 1.5 - 2.5                                                         |
| Link material                | Al 2024-T3 (rho=2780 kg/m^3, E=73.1 GPa, S_ut=483 MPa, S_y=345 MPa) |
| Link geometry                | Rectangular cross-section                                         |
| Slider-guide friction (mu)   | 0.47 - dry machined Al-Al                                         |

**ML model outputs:**

- Pass/fail classification for a given design
- 9 regression targets: total_mass, volume_envelope, tau_A_max, E_rev, min_n_static, utilization, n_buck, n_shaft, min_n_fatigue

**Optimization objectives:**

- Minimize mechanism envelope (size and weight)
- Minimize required motor power while satisfying all design targets

> CAD modeling, 3D-printed prototype, and written report are handled outside this repository.

---

## 2. Pipeline

```
configs/generate/baseline.yaml
         |
         v
+-----------------------+
| Stage 1 - 2D Kinematic|  stage1_kinematic.py
| Screening             |  Samples (r, l, e), solves ROM, checks QRR
+-----------+-----------+
            |  valid (r, l, e)
            v
+-----------------------+
| Stage 2 - 3D Embodiment  stage2_embodiment.py
| Expansion             |  Adds cross-sections and pin diameters
+-----------+-----------+
            |  3D design dicts
            v
+-----------------------+
| Mass Properties       |  mass_properties.py
|                       |  Masses, COGs, MOIs for each link
+-----------+-----------+
            |
            v
+-----------------------+
| Physics Engine        |  engine.py + dynamics.py
| (15 deg sweep)        |  Newton-Euler 8x8 -> F_A, F_B, F_C, N, F_f, tau_A
+-----------+-----------+
            |
            v
+-----------------------+
| Stress + Fatigue +    |  stresses.py, fatigue.py, buckling.py
| Buckling Evaluation   |  sigma, tau per angle; Goodman/Miner; Euler buckling
+-----------+-----------+
            |  sigma_max, tau_max, tau_A_max, E_rev, F_A/B/C_max, sigma_peak per link
            v
+-----------------------+
| Pass / Fail Label     |  generate.py
| + ML Feature Assembly |  utilization, n_buck, fatigue FoSs
|                       |  total_mass, volume_envelope
|                       |  tau_A_max, E_rev, F_A/B/C_max
|                       |  n_static_rod/crank/pin
+-----------+-----------+
            |  labeled dataset (85 cols)
            v
+-----------------------+
| ML Training           |  ml/
| PyTorch multi-task NN |  ReLU trunk + classification & regression heads
| Optuna hyperpar. sweep|  Inputs: 12 features -> pass_fail + 9 targets
+-----------------------+
```

**Hard constraints (enforced by Stage 1):**

| Constraint | Value                             |
| ---------- | --------------------------------- |
| ROM target | 250 mm +- 0.5 mm                  |
| QRR        | [1.5, 2.5]                        |
| RPM        | 30 (fixed)                        |

**Key variable names** (full physics derivations in `instructions.md`):

| Code key                           | Meaning                                                                                          | Unit      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ | --------- |
| `r`, `l`, `e`                      | Crank radius, rod length, offset                                                                 | m         |
| `theta`, `omega`                   | Crank angle, angular speed (= RPM x 2pi/60)                                                     | rad, rad/s |
| `ROM`, `QRR`                       | Slider stroke, quick-return ratio                                                                | m, -      |
| `width_r/l`, `thickness_r/l`       | Link cross-sections                                                                              | m         |
| `d_shaft_A`                        | Motor output shaft diameter at joint A                                                           | m         |
| `pin_diameter_B`, `pin_diameter_C` | Lug pin diameters at joints B and C                                                              | m         |
| `F_A`, `F_B`, `F_C`                | Joint reaction vectors [Fx, Fy]                                                                  | N         |
| `N`, `F_f`, `tau_A`                | Slider normal, friction, drive torque                                                            | N, N, N*m |
| `sigma_max`, `tau_max`             | Peak normal/shear stress over cycle                                                              | Pa        |
| `n_static_rod/crank/pin`           | Static FoS per link: sigma_limit / peak sigma_link                                               | -         |
| `total_mass`                       | mass_crank + mass_rod + mass_slider                                                              | kg        |
| `volume_envelope`                  | Bounding-box volume of assembled mechanism (T x H x L)                                           | m^3       |
| `tau_A_max`                        | Peak motor torque over full cycle                                                                | N*m       |
| `E_rev`                            | Energy per revolution: sum(tau_A) * delta_theta                                                  | J         |
| `F_A_max`, `F_B_max`, `F_C_max`    | Peak resultant force at each pin over cycle                                                      | N         |
| `pass_fail`                        | 1 = pass, 0 = fail                                                                               | -         |

---

## 3. Repository structure

```
MECH390_ML_Model/
+-- instructions.md
+-- configs/
|   +-- generate/
|   |   +-- baseline.yaml        # main config (n_samples, LHS, n_variants_per_2d)
|   |   +-- test_small.yaml      # fast test config
|   |   +-- aggressive.yaml      # EMPTY - placeholder for wider-range config
|   +-- train/
|   |   +-- surrogate.yaml       # Optuna sweep config (arch, dropout, lr, batch)
|   +-- optimize/
|       +-- search.yaml          # weight table + optimizer settings + OOD penalty
+-- src/mech390/
|   +-- config.py                # config loading, normalization, validation
|   +-- physics/
|   |   +-- _utils.py            # shared utilities (get_or_warn fallback logging)
|   |   +-- kinematics.py        # positions, velocities, accelerations, ROM/QRR
|   |   +-- dynamics.py          # Newton-Euler 8x8 solver
|   |   +-- mass_properties.py   # mass, MOI, COG helpers + design aggregator
|   |   +-- engine.py            # 15-degree sweep orchestrator
|   |   +-- stresses.py          # sigma, tau per crank angle (axial, bending, torsion, shear)
|   |   +-- fatigue.py           # Marin factors, modified Goodman, Basquin, Miner's rule
|   |   +-- buckling.py          # Euler buckling check (pin-pin, weak axis)
|   +-- datagen/
|   |   +-- sampling.py          # LHS + random sampler
|   |   +-- stage1_kinematic.py  # Stage 1 (streaming iterator)
|   |   +-- stage2_embodiment.py # 3D expansion
|   |   +-- generate.py          # orchestrator with pass/fail labeling (85 cols)
|   +-- ml/
|       +-- __init__.py
|       +-- features.py          # scaler, feature split, min_n_static, target stats
|       +-- models.py            # CrankSliderSurrogate (ReLU trunk + clf + reg heads)
|       +-- train.py             # Optuna sweep, early stopping, checkpoint save
|       +-- infer.py             # SurrogatePredictor (dict / DataFrame inference)
|       +-- optimize.py          # surrogate optimizer (differential_evolution, top-N)
+-- scripts/
|   +-- run_pipeline.py          # full 4-step pipeline (datagen -> train -> optimize -> report)
|   +-- generate_dataset.py      # data generation -> 7 CSVs
|   +-- train_model.py           # surrogate training
|   +-- optimize_design.py       # surrogate optimizer
|   +-- summarize_results.py     # manufacturing report for top-N candidates
|   +-- validate_candidate.py    # full physics check on a specific geometry dict
|   +-- export_test_appendix.py  # exports test-set inputs and predictions for report
|   +-- preview_stage1.py        # Stage 1 only -> CSV
|   +-- preview_stage2.py        # Stage 1 + Stage 2 + mass props -> CSV
|   +-- preview_forces.py        # full pipeline -> force sweep (n_designs x 24 angles)
|   +-- generate_report.py       # plots: heatmap, learning curves, parity, convergence, sensitivity
+-- data/
|   +-- models/
|   |   +-- surrogate_best.pt        # best network state dict
|   |   +-- scaler.pkl               # fitted StandardScaler
|   |   +-- target_stats.json        # per-target min/max for normalization
|   |   +-- optuna_history.json      # trial loss/metric records over epochs
|   |   +-- validation_preds.npz     # val set predictions for parity scatter
|   +-- results/
|   |   +-- candidates.json          # top-N compiled design dicts
|   |   +-- convergence_log.json     # DE optimizer progression history
|   +-- preview/                     # output of preview_*.py and generate_dataset.py
|   |   +-- stage1_geometries.csv    # 2D kinematic results
|   |   +-- stage2_designs.csv       # 3D embodiment results
|   |   +-- forces_sweep.csv         # per (design, angle): joint forces, torque
|   |   +-- kinematics.csv           # per (design, angle): positions, velocities, accels
|   |   +-- dynamics.csv             # per (design, angle): joint forces, torque
|   |   +-- stresses.csv             # per (design, angle): per-component sigma, tau
|   |   +-- fatigue.csv              # per design: Goodman / Miner metrics
|   |   +-- buckling.csv             # per design: Euler buckling metrics
|   |   +-- passed_configs.csv       # passing designs with all check columns
|   |   +-- failed_configs.csv       # failing designs with all check columns
|   +-- runs/                        # named production runs (--out-dir data/runs/<name>)
|   +-- splits/                      # train.csv, validate.csv, test.csv (written by train_model.py)
+-- reports/
|   +-- optimization/                # convergence curves and heatmap reports
|   +-- training/                    # learning curves and parity validation plots
+-- tests/
    +-- test_datagen_units.py        # kinematics, datagen, import tests
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

```bash
# all defaults
.venv/bin/python scripts/run_pipeline.py

# with options
.venv/bin/python scripts/run_pipeline.py \
    --generate-config configs/generate/baseline.yaml \
    --train-config    configs/train/surrogate.yaml \
    --optimize-config configs/optimize/search.yaml \
    --seed            42 \
    --out-dir         data/preview

# skip completed steps
.venv/bin/python scripts/run_pipeline.py --skip-datagen
.venv/bin/python scripts/run_pipeline.py --skip-datagen --skip-training
```

Runs 4 steps: data generation -> surrogate training -> design optimization -> manufacturing report.

---

### Step by step

**Step 1 - Data generation**

```bash
.venv/bin/python scripts/generate_dataset.py
.venv/bin/python scripts/generate_dataset.py --config configs/generate/baseline.yaml --seed 42 --out-dir data/preview
```

Writes to `--out-dir`: `kinematics.csv`, `dynamics.csv`, `stresses.csv`, `fatigue.csv`, `buckling.csv`, `passed_configs.csv`, `failed_configs.csv`.

**Step 2 - Surrogate training**

```bash
.venv/bin/python scripts/train_model.py
.venv/bin/python scripts/train_model.py --config configs/train/surrogate.yaml --seed 42
```

Saves to `data/models/`: `surrogate_best.pt`, `scaler.pkl`, `target_stats.json`. Also writes `data/splits/train.csv`, `validate.csv`, `test.csv`.

**Step 3 - Design optimization**

```bash
.venv/bin/python scripts/optimize_design.py
.venv/bin/python scripts/optimize_design.py \
    --generate-config configs/generate/baseline.yaml \
    --optimize-config configs/optimize/search.yaml \
    --model           data/models/surrogate_best.pt
```

**Step 4 - Manufacturing report**

```bash
.venv/bin/python scripts/summarize_results.py --top 3
```

Reads `data/results/candidates.json`, runs full physics validation, prints ROM/QRR/FoS/mass/verdict for each candidate.

**Validate a specific geometry** (bypasses Stage 1, runs full physics)

```bash
# Edit CANDIDATE dict at top of script first
.venv/bin/python scripts/validate_candidate.py --config configs/generate/baseline.yaml
```

---

### Preview scripts

```bash
.venv/bin/python scripts/preview_stage1.py --out-dir data/preview   # 2D kinematics only
.venv/bin/python scripts/preview_stage2.py --out-dir data/preview   # + 3D embodiment
.venv/bin/python scripts/preview_forces.py --out-dir data/preview   # + force sweep
```

All accept `--config`, `--seed`, `--out-dir` (and `--max-2d` for stage2/forces).

---

## 5. Open work

**Config:**
- Fill `configs/generate/aggressive.yaml` - wider parameter ranges, higher n_samples

**Scripts:**
- `scripts/visualize_design.py` - 2D mechanism drawing from a design dict (crank, rod, slider, pin circles, guide rail, annotated metrics)
- `scripts/preview_stresses.py` - per-angle sigma/tau output, like preview_forces.py

**Tests:**
- Expand `test_datagen_units.py` - currently covers kinematics and imports only
- Physics module unit tests: dynamics.py (Newton-Euler solver), stresses.py (rod/crank/pin formulas), fatigue.py (Goodman + Miner), buckling.py, mass_properties.py, engine.py (full sweep integration)
- ML stack tests: features.py (normalization round-trip), models.py (forward pass shapes), infer.py (load + predict), optimize.py (constraint satisfaction)
- Regression test: fixed-seed config -> dataset -> at least N passing designs; run as CI smoke test
- Add pytest.ini or pyproject.toml test config with shared fixtures

**Code:**
- Type annotations in physics modules (dynamics.py, stresses.py, fatigue.py, buckling.py)
- `design_eval` flat dict in generate.py - refactor to typed sub-dicts deferred until schema stabilises

**Postponed:**
- Sign convention at Pin A: `_crank_frame_forces` defines F_r,crank,A and F_t,crank,A with opposite signs to the Mother Doc derivation. No numerical impact while abs() wrapping is applied throughout the crank stress path. Deferred until abs() wrapping is removed.

---

## 6. Current status

Dataset size is controlled by `n_samples` and `n_variants_per_2d` in `configs/generate/baseline.yaml`. Re-running `generate_dataset.py` is all that is needed to produce a new dataset. The ML stack reads `len(INPUT_FEATURES)` and `len(REGRESSION_TARGETS)` dynamically - no hardcoded counts in the training path.

### Dataset (latest run)

| Item          | Value                          |
| ------------- | ------------------------------ |
| Total rows    | 10,000                         |
| Pass          | 7,526 (75.3%)                  |
| Fail          | 2,474 (24.7%)                  |
| Source config | `configs/generate/baseline.yaml` |

### Trained surrogate (`data/models/surrogate_best.pt`)

| Item           | Value                                    |
| -------------- | ---------------------------------------- |
| Best val F1    | 0.9704                                   |
| Architecture   | [512, 256, 128] shared ReLU trunk + clf head + 9-target reg head |
| Dropout        | 0.269                                    |
| Learning rate  | 8.26e-3                                  |
| Batch size     | 256                                      |
| Optuna sweep   | 50 trials x 300 epochs max, patience=25  |
| Early stopping | on val_f1                                |

Val R^2 (pass rows only, normalized space):

| Target          | R^2  |
| --------------- | ---- |
| E_rev           | 0.90 |
| n_shaft         | 0.89 |
| total_mass      | 0.83 |
| volume_envelope | 0.82 |
| min_n_fatigue   | 0.75 |
| n_buck          | 0.72 |
| utilization     | 0.68 |
| tau_A_max       | 0.63 |
| min_n_static    | 0.63 |

### Optimizer

Five analytical constraints are enforced as penalty terms in `optimize.py`, in addition to the surrogate pass_prob gate:

1. Net-section feasibility: `width - D_pin > delta + 2*min_wall` for all 4 pin pairs
2. Kinematic feasibility: `l > r + e`
3. Euler buckling: analytical `P_cr = pi^2 * E * I / l^2`; penalizes n_buck < 3.0
4. ROM: `|ROM - 0.25| > ROM_tolerance` penalized via `penalty_rom_scale`
5. QRR: `QRR not in [1.5, 2.5]` penalized via `penalty_qrr_scale`

OOD penalty: regression predictions outside training range by more than `ood_tolerance` (default 10%) are penalized proportionally. Both parameters are configurable in `search.yaml`.

### Physics constants confirmed

| Item                  | Value                              |
| --------------------- | ---------------------------------- |
| Fatigue strength Sn   | 133 MPa at 18.72e6 cycles (Al 2024-T3, ASM) |
| Basquin S-N constants | sigma_a = 924 * N^(-0.086); anchors: (10^7 cycles, 230 MPa), (10^9 cycles, 155 MPa) |
| Size factor C_s       | per Mott Table 5-3; C_sur = 0.88 as-machined |

> Full derivations and code contracts: `instructions.md`.
