# FULL TECHNICAL SPECIFICATION AND REPOSITORY CONTRACT  

## Offset Crank–Slider Data Generation, Simulation, and ML System  

**Audience:** Coding AI agents, automation systems, advanced developers  
**Goal:** Enable full implementation with zero ambiguity  

---

## 1. Scope of the project (authoritative)

This repository defines a **complete physics-first pipeline** for:

1. Synthesizing offset crank–slider mechanisms that satisfy **hard kinematic constraints**
2. Expanding those mechanisms into **3D parametric geometries**
3. Computing **dynamics and stresses** deterministically
4. Labeling designs as **pass/fail**
5. Training **machine learning models** on the resulting dataset
6. Using trained models to **rapidly evaluate or optimize new designs**

All results must be **reproducible**, **physically consistent**, and **traceable to configuration files**.

This project is NOT exploratory scripting.  
It is a structured simulation + ML system.

---

## 2. Hard constraints (non-negotiable)

These constraints are assumed known and are enforced by design:

- **Constant RPM**
  - Angular speed is fixed
  - RPM is NOT a design variable
- **Range of Motion (ROM)**
  - Target slider stroke: `ROM_target = 250 mm`
  - Treated as a hard acceptance target with configured tolerance:

    ```
    |ROM_computed - ROM_target| <= ROM_tolerance
    ```
- **Quick Return Ratio (QRR)**
  - Forward-to-return time ratio must satisfy:

    ```
    QRR_min ≤ QRR ≤ QRR_max
    ```

  - Since RPM is constant, time ratios are crank-angle ratios

Any design that violates these constraints is invalid and must never reach stress evaluation.

---

## 3. Fundamental variables

### 3.1 2D kinematic variables

- `r` : crank radius
- `l` : connecting rod length
- `D` (code key `e`) : offset (vertical distance from crank pivot to slider line)
- `θ` (code key `theta`) : crank angle
- `ω` (code key `omega`) : angular speed (constant)

### 3.2 3D embodiment variables (examples, extensible)

- `w_r` (code key `width_r`) : crank width
- `t_r` (code key `thickness_r`) : crank thickness
- `w_l` (code key `width_l`) : rod width
- `t_l` (code key `thickness_l`) : rod thickness
- `d_A` (code key `pin_diameter_A`) : pin diameter at A
- `d_B` (code key `pin_diameter_B`) : pin diameter at B
- `d_C` (code key `pin_diameter_C`) : pin diameter at C
- `ρ` (code key `material.rho`) : material density

### 3.3 Derived physical quantities

- mass of each body
- center of gravity of each body — returned as `np.ndarray([x, y])`
- mass moment of inertia of each body
- area moment of inertia of link/slider sections (bending properties)
- joint reaction forces — returned as `np.ndarray([Fx, Fy])`
- crank drive torque `τ_A` (also denoted `T` in equations; code key `tau_A`)
- normal and shear stresses

---

## 4. Two-stage pipeline (mandatory design)

### Stage 1 — 2D kinematic synthesis and filtering

- Purpose: eliminate invalid mechanisms cheaply
- Inputs: sampled geometry variables
- Outputs: kinematically valid `(r, l, D)` tuples

Operations:

1. Sample two of `{r, l, D}` (current implementation samples `l` and `D`)
2. Solve for the third variable (`r`) using the closed-form ROM relation
3. Apply closed-form feasibility checks (real radicals, branch feasibility, full-rotation geometry)
4. Find dead-center crank angles via root-finding
5. Compute ROM and QRR from kinematics
6. Accept only if both ROM tolerance and QRR bounds are satisfied

NO dynamics. NO stresses.

---

### Stage 2 — 3D embodiment, dynamics, and stress

- Purpose: evaluate structural feasibility
- Inputs: valid 2D mechanisms
- Outputs: stress metrics + pass/fail labels

Operations:

1. Generate multiple 3D geometry variants for each valid 2D mechanism
2. Sample widths, thicknesses, and pin diameters from config-defined ranges
3. Enforce Stage-2 geometric constraints:
   - `width_r > pin_diameter_A`
   - `width_r > pin_diameter_B`
   - `width_l > pin_diameter_B`
   - `width_l > pin_diameter_C`
4. Compute mass and inertia properties
5. Evaluate dynamics every 15° using a Newton–Euler linear solve that returns:
   - `F_A`, `F_B`, `F_C` (joint reactions)
   - `N`, `F_f` (slider normal + kinetic Coulomb friction)
   - `tau_A` (required crank torque)
   - compatibility alias `F_O = F_A`
6. Compute stresses
7. Track maximum stress values
8. Apply pass/fail criteria

---

## 5. Authoritative kinematic equations (summary)

### Vector convention

All position, velocity, and acceleration quantities are `np.ndarray` of shape `(2,)` representing `[x, y]`.

- **Slider C** is constrained to the x-axis: `pos_C = [x_C, 0]`, `vel_C = [v_Cx, 0]`, `acc_C = [a_Cx, 0]`
- **Crank pin B** moves in a full circle: `pos_B = [r·cosθ, r·sinθ]`, with non-zero x and y components

### Slider position

    x_C(θ) = r cos θ + √(l² − (r sin θ + D)²)
    pos_C   = np.array([x_C, 0.0])

### Crank pin position

    pos_B = np.array([r·cos θ, r·sin θ])

### Dead centers

    dx_C / dθ = 0    (solved numerically via root-finding)

### ROM

    ROM = x_C(θ_max) − x_C(θ_min)

At the dead centres the triangle OBC is collinear, giving the exact positions:

    x_max = √((r + l)² − D²)    (extended)
    x_min = √((l − r)² − D²)    (retracted)

### Closed-form solution for r given l, D, and ROM

Setting S = ROM and solving algebraically:

    r = (S / 2) · √( (4(l² − D²) − S²) / (4l² − S²) )

Feasibility conditions that must hold before applying the formula:

| Condition | Meaning |
|---|---|
| `D < l` | A valid triangle must exist |
| `S < 2l` | Keeps the denominator positive |
| `S < 2·√(l²−D²)` | Keeps the numerator positive; physical maximum stroke |

Additional branch-feasibility conditions that must hold after computing `r`:

| Condition | Meaning |
|---|---|
| `l > r + |D|` | Full-rotation geometry validity (no lockup over one cycle) |
| `l² + r² − D² − S²/2 >= 0` | Squared-equation branch consistency |
| `|S_original(r,l,D) − S_target| <= ROM_tolerance` | Rejects extraneous roots introduced by squaring |

### Quick return ratio

    Δθ_forward = (θ_max − θ_min) mod 2π
    Δθ_return  = 2π − Δθ_forward
    QRR = Δθ_forward / Δθ_return

All dead-center detection must be done via robust root-finding.

---

## 6. Dataset definition (authoritative schema)

Each dataset row MUST include:

### Inputs

- `r, l, D`
- all 3D geometry parameters
- mass properties
- mass moments of inertia
- area moments of inertia used in stress calculations

### Outputs

- `sigma_max` (maximum normal stress over cycle)
- `tau_max` (maximum shear stress over cycle)
- `utilization = max(sigma_max/σ_allow, tau_max/τ_allow)`
- `pass_fail` (binary)
- optional: crank angle at which maxima occur

---

## 7. Repository structure and file contracts

mech390-crank-slider-ml/

---

### `configs/`

**Purpose:** Define experiments. No code logic here.

configs/
├─ generate/
│  ├─ baseline.yaml
│  └─ aggressive.yaml
├─ train/
│  ├─ regression.yaml
│  └─ classifier.yaml
└─ optimize/
└─ search.yaml

Each config file:

- defines sampling ranges
- defines constraints
- can define Stage-2 embodiment controls (`n_variants_per_2d`, optional retry cap)
- defines output paths
- contains NO physics logic

Configuration loading is responsible for numeric normalization and range validation.

---

### `src/`

**Purpose:** All reusable, deterministic logic.

src/
└─ mech390/
├─ physics/
│  ├─ kinematics.py
│  ├─ dynamics.py
│  ├─ stresses.py
│  └─ engine.py

- `kinematics.py`
  - Implements all position, velocity, acceleration equations
  - **All quantities returned as `np.ndarray([x, y])`**
  - Slider C (x-axis constrained): `slider_position`, `slider_velocity`, `slider_acceleration`
  - Crank pin B (full 2D circle): `crank_pin_position`, `crank_pin_velocity`, `crank_pin_acceleration`
  - Dead-center detection via root-finding
  - ROM and QRR evaluation
  - NO randomness

- `dynamics.py`
  - Newton–Euler 8x8 linear system solve per crank angle
  - Returns joint reactions `F_A`, `F_B`, `F_C` as `np.ndarray([Fx, Fy])`
  - Also returns `N`, `F_f`, and `tau_A` with compatibility alias `F_O`
  - Uses outputs from kinematics

- `stresses.py`
  - Stress formulas
  - Section properties
  - Returns σ(θ), τ(θ)

- `engine.py`
  - Orchestrates 15° sweep
  - Tracks maxima
  - Returns summary metrics

- `mass_properties.py`
  - Center-of-gravity positions — returned as `np.ndarray([x, y])`
  - Modular mass/inertia helpers for link and slider bodies with pin holes
  - Design-level aggregator `compute_design_mass_properties(...)`
  - Link area moments reported with `Iyy` and `Izz`

---

├─ datagen/
│  ├─ sampling.py
│  ├─ stage1_kinematic.py
│  ├─ stage2_embodiment.py
│  └─ generate.py

- `sampling.py`
  - Random / Latin hypercube sampling
  - Seed control
  - NO physics

- `stage1_kinematic.py`
  - Implements Stage 1 algorithm
  - Solves for unknown geometry using closed-form `r(l,D,S)`
  - Applies branch-feasibility checks to reject extraneous analytical roots
  - Applies ROM and QRR constraints

- `stage2_embodiment.py`
  - Expands each valid 2D mechanism into multiple 3D variants
  - Supports streaming iterator-style generation for large runs
  - Enforces width/pin feasibility constraints during sampling

- `generate.py`
  - High-level data generation pipeline
  - Consumes Stage-2 designs incrementally (streaming-capable)
  - Writes CSV outputs

---

├─ ml/
│  ├─ features.py
│  ├─ models.py
│  ├─ train.py
│  └─ infer.py

- `features.py`
  - Feature selection
  - Scaling
- `models.py`
  - ML architectures
- `train.py`
  - Training loops
- `infer.py`
  - Prediction utilities

---

### `scripts/`

**Purpose:** Entry points only (thin wrappers).

scripts/
├─ generate_dataset.py
├─ train_model.py
└─ optimize_config.py

Each script:

- loads a config file
- calls library code
- creates a run directory
- logs metadata

---

### `data/`

**Purpose:** Generated artifacts only.

data/
├─ raw/
│  └─ <run_id>/
├─ processed/
├─ models/
└─ splits/

Never hard-code paths to this directory.

---

### `reports/`

**Purpose:** Diagnostics and summaries.

reports/
├─ data_generation/
├─ training/
└─ optimization/

---

## 8. Expected usage pattern

1. Define a generation config
2. Run dataset generation
3. Inspect summary reports
4. Train ML model
5. Use ML for rapid evaluation or optimization

All steps are repeatable and configuration-driven.

---

## 9. Determinism and reproducibility rules

- All physics functions must be deterministic
- Randomness only allowed in sampling modules
- All runs must be seed-controlled
- Outputs must be traceable to configs

---

## 10. Final contract statement

This document defines:

- all constraints
- all variables
- all equations
- all file responsibilities
- all expected behaviors

Any implementation that follows this document is considered correct.

No assumptions beyond what is written here are allowed.
