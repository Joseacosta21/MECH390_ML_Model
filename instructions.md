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
  - Treated as a target with tolerance (e.g. ±1–2 mm)
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
- `D` : offset (vertical distance from crank pivot to slider line)
- `θ` : crank angle
- `ω` : angular speed (constant)

### 3.2 3D embodiment variables (examples, extensible)
- link thickness
- link width
- link height
- pin diameter
- material density

### 3.3 Derived physical quantities
- mass of each body
- center of gravity of each body
- mass moment of inertia of each body
- joint reaction forces
- normal and shear stresses

---

## 4. Two-stage pipeline (mandatory design)

### Stage 1 — 2D kinematic synthesis and filtering
- Purpose: eliminate invalid mechanisms cheaply
- Inputs: sampled geometry variables
- Outputs: kinematically valid `(r, l, D)` tuples

Operations:
1. Sample two of `{r, l, D}`
2. Solve for the third variable so that `ROM ≈ 250 mm`
3. Find dead-center crank angles
4. Compute QRR
5. Accept or reject geometry

NO dynamics. NO stresses.

---

### Stage 2 — 3D embodiment, dynamics, and stress
- Purpose: evaluate structural feasibility
- Inputs: valid 2D mechanisms
- Outputs: stress metrics + pass/fail labels

Operations:
1. Generate 3D geometry variants
2. Compute mass and inertia
3. Evaluate dynamics every 15°
4. Compute stresses
5. Track maximum stress values
6. Apply pass/fail criteria

---

## 5. Authoritative kinematic equations (summary)

Slider position:

x_C(θ) = r cos θ + √(l² − (r sin θ + D)²)

Dead centers:

dx_C / dθ = 0

ROM:

ROM = x_C(θ_max) − x_C(θ_min)

Quick return ratio:

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
- mass and inertia properties

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
- defines output paths
- contains NO physics logic

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
  - Dead-center detection
  - ROM and QRR evaluation
  - NO randomness

- `dynamics.py`
  - Newton–Euler equations
  - Joint reaction forces
  - Uses outputs from kinematics

- `stresses.py`
  - Stress formulas
  - Section properties
  - Returns σ(θ), τ(θ)

- `engine.py`
  - Orchestrates 15° sweep
  - Tracks maxima
  - Returns summary metrics

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
  - Solves for unknown geometry
  - Applies ROM and QRR constraints

- `stage2_embodiment.py`
  - Expands 2D mechanisms into 3D
  - Computes mass/inertia

- `generate.py`
  - High-level data generation pipeline
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