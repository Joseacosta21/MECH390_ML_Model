# Data-Driven Design of an Offset Crank–Slider Mechanism  
**MECH 390 – Machine Learning–Assisted Mechanical Design**

---

## 1. What this repository is about

This repository implements a physics-based data generation and machine learning workflow for the design and evaluation of an offset crank–slider mechanism.

The goal is to automate what is traditionally done by hand in mechanical design:
- select a mechanism geometry,
- verify that it satisfies motion requirements,
- evaluate forces and stresses,
- and determine whether the design is structurally acceptable.

Instead of evaluating a small number of designs manually, this project generates large datasets using exact kinematics and dynamics, then trains a machine learning model to rapidly evaluate new designs.

Physics governs the behavior of the system.  
Machine learning is used only to learn patterns from physically valid data.

---

## 2. High-level workflow

The project is intentionally divided into two stages:

1. **2D kinematic feasibility**
2. **3D embodiment, dynamics, and stress evaluation**

This separation ensures that expensive stress calculations are only performed on mechanisms that are already valid from a motion standpoint.

---

## 3. Stage 1 – 2D kinematic feasibility

In Stage 1, the mechanism is treated as a purely planar offset crank–slider.

Only geometry and motion are considered.  
No forces, masses, or stresses are evaluated at this stage.

### Process
1. Two geometric parameters are randomly sampled  
   (e.g., connecting rod length and offset).
2. The remaining geometric parameter (typically the crank radius) is solved for numerically so that the slider achieves the required range of motion.
3. Dead-center positions are computed by solving for zero slider velocity.
4. The forward and return crank-angle spans are evaluated.
5. The quick return ratio is computed from these angle spans.
6. The geometry is retained only if:
   - the range of motion matches the required value within tolerance, and
   - the quick return ratio lies within the acceptable range.

This stage produces a set of kinematically valid two-dimensional mechanisms.

---

## 4. Stage 2 – 3D embodiment, dynamics, and stress evaluation

Only mechanisms that pass Stage 1 are evaluated further.

In Stage 2, each valid 2D mechanism is expanded into a family of three-dimensional designs.

### Process
1. A standard link topology is assumed for each component.
2. Key geometric dimensions such as thickness, width, height, and pin diameter are varied.
3. Mass, center of gravity, and mass moments of inertia are computed from the 3D geometry.
4. Dynamic forces are evaluated at every 15° of crank rotation.
5. Normal and shear stresses are computed throughout the cycle.
6. The maximum stress values over the full cycle are extracted.
7. Each design is classified as pass or fail based on allowable stress limits.

This stage generates the dataset used for machine learning.

---

## 5. Why the two-stage approach is used

This structure reflects standard mechanical design practice:

1. First ensure the mechanism functions kinematically.
2. Then evaluate structural performance and strength.
3. Finally, optimize or automate design decisions.

From a computational standpoint:
- kinematic checks are inexpensive and eliminate invalid designs early,
- stress analysis is reserved for physically meaningful cases,
- overall data generation becomes scalable and efficient.

---

## 6. Repository structure

mech390-crank-slider-ml/
├─ configs/        # Experiment definitions (YAML files)
├─ src/            # Physics, data generation, and ML code
├─ scripts/        # Executable entry points
├─ data/           # Generated datasets and trained models
├─ reports/        # Plots and run summaries
└─ README.md

All experiments are defined through configuration files.  
The core code does not need to be modified to run new studies.

---

## 7. Configuration files

Configuration files describe how data is generated and how models are trained.

They define:
- geometry sampling ranges,
- kinematic constraints,
- stress limits,
- output locations.

Rather than listing individual designs, configuration files specify how entire families of designs are generated.

---


## 9. Role of machine learning

Machine learning does not replace physical modeling in this project.

Instead, it is used to:
	•	approximate the relationship between geometry and peak stress,
	•	rapidly classify designs as likely pass or fail,
	•	enable fast exploration of large design spaces.

All training data is generated using physics-based equations.

⸻


# Repository File Tree and Responsibilities (Authoritative)

Each file below is listed **once** with its responsibility stated **inline**, so an AI agent or developer can immediately understand scope, ownership, and behavior.

---

mech390-crank-slider-ml/
├─ README.md
├─ TECHNICAL_SPEC.md
├─ pyproject.toml
├─ configs/
│  ├─ generate/
│  ├─ train/
│  └─ optimize/
├─ src/
│  └─ mech390/
├─ scripts/
├─ data/
├─ reports/
└─ tests/

## File and Directory Responsibilities

| Path | Description |
|-----|-------------|
| `README.md` | High-level project overview for mechanical engineering students |
| `TECHNICAL_SPEC.md` | Authoritative technical specification for developers and AI agents |
| `configs/` | YAML experiment definitions (no executable logic) |
| `configs/generate/` | Data generation configurations |
| `configs/train/` | ML training configurations |
| `configs/optimize/` | Optimization and inference configurations |
| `src/mech390/physics/kinematics.py` | All position, velocity, acceleration, ROM, and QRR equations |
| `src/mech390/physics/dynamics.py` | Newton–Euler force and moment equations |
| `src/mech390/physics/stresses.py` | Normal and shear stress calculations |
| `src/mech390/datagen/stage1_kinematic.py` | 2D kinematic synthesis and filtering |
| `src/mech390/datagen/stage2_embodiment.py` | 3D geometry, mass, and inertia generation |
| `scripts/generate_dataset.py` | CLI entry point for dataset generation |
| `scripts/train_model.py` | CLI entry point for ML training |
| `scripts/optimize_config.py` | CLI entry point for ML-based design evaluation |