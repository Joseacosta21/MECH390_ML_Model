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

mech390-crank-slider-ml/                         # Project root

├─ README.md                                     # Human-readable project overview (mechanical focus, minimal code)
├─ TECHNICAL_SPEC.md                             # Authoritative technical contract (equations, constraints, pipeline)
├─ pyproject.toml / requirements.txt             # Python dependencies and environment definition
├─ .gitignore                                    # Excludes generated data, models, caches, reports

├─ configs/                                      # Experiment definitions (NO code logic)
│  ├─ generate/                                 # Data generation experiment recipes
│  │  ├─ baseline.yaml                          # Standard geometry ranges, balanced pass/fail distribution
│  │  └─ aggressive.yaml                        # Edge-of-feasibility sampling to populate failure boundary
│  ├─ train/                                    # ML training configurations
│  │  ├─ regression.yaml                        # Train regression model (stress / utilization prediction)
│  │  └─ classifier.yaml                        # Train classifier (pass/fail prediction)
│  └─ optimize/                                 # ML-based evaluation / optimization configs
│     └─ search.yaml                            # Defines design space and optimization objective

├─ src/                                         # All reusable, deterministic source code
│  └─ mech390/                                  # Top-level Python package
│     ├─ init.py                            # Package initializer
│
│     ├─ common/                                # Shared utilities (no physics)
│     │  ├─ io.py                               # Centralized I/O (CSV/YAML/JSON, run folders, logging)
│     │  ├─ schema.py                           # Dataset column definitions and validation rules
│     │  └─ utils.py                            # General helpers (units, checks, formatting)
│
│     ├─ physics/                               # Physics layer (fully deterministic)
│     │  ├─ kinematics.py                       # All position/velocity/acceleration equations, ROM, QRR, dead centers
│     │  ├─ dynamics.py                         # Newton–Euler equations, joint reaction forces
│     │  ├─ stresses.py                         # Normal and shear stress calculations from forces and geometry
│     │  └─ engine.py                           # 15° crank sweep, aggregation of peak stresses
│
│     ├─ datagen/                               # Data generation pipeline
│     │  ├─ sampling.py                         # Random / LHS sampling, seed control, no physics
│     │  ├─ stage1_kinematic.py                 # Stage 1: 2D kinematic synthesis (ROM + QRR enforcement)
│     │  ├─ stage2_embodiment.py                # Stage 2: 3D geometry, mass, inertia generation
│     │  └─ generate.py                         # End-to-end dataset generation orchestration
│
│     ├─ ml/                                    # Machine learning layer
│     │  ├─ features.py                         # Feature selection, scaling, normalization
│     │  ├─ models.py                           # ML model architectures (NNs, classifiers)
│     │  ├─ train.py                            # Training loops, losses, metrics, checkpoints
│     │  └─ infer.py                            # Model loading and prediction utilities
│
│     └─ optimize/                              # Design space exploration
│        ├─ objective.py                        # Optimization objective using ML predictions
│        └─ search.py                           # Search strategies (random, heuristic, etc.)

├─ scripts/                                     # Thin command-line entry points
│  ├─ generate_dataset.py                       # Runs full data generation from a YAML config
│  ├─ train_model.py                            # Trains ML models from prepared datasets
│  └─ optimize_config.py                        # Uses ML to evaluate or optimize new designs

├─ data/                                        # Generated artifacts only (never hand-edited)
│  ├─ raw/                                     # Raw datasets (all cases, pass-only, metadata)
│  ├─ processed/                               # Cleaned / feature-engineered datasets
│  ├─ models/                                  # Saved ML models, scalers, training metadata
│  └─ splits/                                  # Train/validation/test splits or indices

├─ reports/                                     # Generated summaries and diagnostics
│  ├─ data_generation/                          # ROM/QRR acceptance rates, stress distributions
│  ├─ training/                                 # Learning curves, evaluation metrics
│  └─ optimization/                             # Optimization and search results

└─ tests/                                       # Validation and regression tests
├─ test_kinematics.py                        # Unit tests for kinematics, ROM, QRR
├─ test_stresses.py                          # Unit tests for stress calculations
└─ test_pipeline_smoke.py                    # End-to-end pipeline sanity test

---