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

1. Rod length `l` and offset `e` are sampled from configured ranges.
2. The crank radius `r` is solved **analytically** (exact closed-form) from the target ROM:

   ```
   r = (S/2) * sqrt( (4(l²−e²) − S²) / (4l² − S²) )
   ```

   where `S = ROM`. The following conditions are checked before applying the formula:
   - `e < l` — a valid triangle must exist
   - `S < 2l` — keeps the denominator positive
   - `S < 2√(l²−e²)` — keeps the numerator positive (physical maximum stroke)
   - `l > r + |e|` — full-rotation geometry feasibility

   After solving `r`, Stage 1 also enforces branch feasibility introduced by squared algebra:
   - `l² + r² − e² − S²/2 >= 0`
   - residual check against the original ROM expression:
     `|ROM_computed − ROM_target| <= ROM_tolerance`

3. Dead-center positions are found via robust numerical root-finding on the velocity equation.
4. The forward and return crank-angle spans are evaluated.
5. The quick return ratio is computed from these angle spans.
6. The geometry is retained only if:
   - the ROM target is met within tolerance, and
   - the quick return ratio lies within the acceptable range.

This stage produces a set of kinematically valid two-dimensional mechanisms.

---

## 4. Stage 2 – 3D embodiment, dynamics, and stress evaluation

Only mechanisms that pass Stage 1 are evaluated further.

In Stage 2, each valid 2D mechanism is expanded into a family of three-dimensional designs.

### Process

1. For each valid `(r, l, e)` geometry, many 3D variants are generated (controlled by `sampling.n_variants_per_2d`).
2. Widths, thicknesses, and pin diameters are sampled from configuration ranges.
3. Geometric feasibility constraints are enforced:
   - `width_r > pin_diameter_A`
   - `width_r > pin_diameter_B`
   - `width_l > pin_diameter_B`
   - `width_l > pin_diameter_C`
4. Mass properties are evaluated through the modular `mass_properties` API:
   masses, center-of-gravity vectors, mass moments, and area moments.
5. Dynamic forces are evaluated at every 15° of crank rotation using a planar Newton–Euler solve that returns:
   - joint reactions at A, B, and C (`F_A`, `F_B`, `F_C`)
   - slider normal/friction (`N`, `F_f`, kinetic Coulomb)
   - required crank torque (`tau_A`)
   - compatibility alias `F_O = F_A`
6. Normal and shear stresses are computed throughout the cycle.
7. The maximum stress values over the full cycle are extracted.
8. Each design is classified as pass or fail based on allowable stress limits.

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
- stage-2 variant controls (`n_variants_per_2d`, optional retry cap),
- stress limits,
- output locations.

Configuration loading normalizes numeric values (including scientific notation) and validates `{min,max}` ranges before sampling.

Rather than listing individual designs, configuration files specify how entire families of designs are generated.

---

## 8. Nomenclature

| Variable | Meaning | Units |
|---|---|---|
| `r` | Crank radius (link O-B center distance) | m |
| `l` | Connecting rod length (link B-C center distance) | m |
| `e` (`D`) | Offset between crank centerline and slider axis | m |
| `theta` (`θ`) | Crank angle | rad |
| `omega` (`ω`) | Crank angular speed | rad/s |
| `alpha_r` (`α_r`) | Crank angular acceleration | rad/s² |
| `phi` (`φ`) | Connecting-rod angle | rad |
| `alpha_l` (`α_l`) | Connecting-rod angular acceleration | rad/s² |
| `ROM` / `S` | Slider range of motion (stroke) | m |
| `QRR` | Quick-return ratio (forward/return angle ratio) | dimensionless |
| `x_C` | Slider x-position | m |
| `v_Cx` | Slider x-velocity | m/s |
| `a_Cx` | Slider x-acceleration | m/s² |
| `mass_crank` (`m_r`) | Crank mass | kg |
| `mass_rod` (`m_l`) | Rod mass | kg |
| `mass_slider` (`m_s`) | Slider mass | kg |
| `width_r` (`w_r`) | Crank link width | m |
| `thickness_r` (`t_r`) | Crank link thickness | m |
| `width_l` (`w_l`) | Rod link width | m |
| `thickness_l` (`t_l`) | Rod link thickness | m |
| `pin_diameter_A` (`d_A`) | Pin diameter at joint A | m |
| `pin_diameter_B` (`d_B`) | Pin diameter at joint B | m |
| `pin_diameter_C` (`d_C`) | Pin diameter at joint C | m |
| `I_mass_crank_cg_z` | Crank mass moment of inertia about CG z-axis | kg·m² |
| `I_mass_rod_cg_z` | Rod mass moment of inertia about CG z-axis | kg·m² |
| `I_mass_slider_cg_z` | Slider mass moment of inertia about CG z-axis | kg·m² |
| `Iyy`, `Izz` | Area moments of inertia of cross-section (bending) | m⁴ |
| `F_A`, `F_B`, `F_C` | Joint reaction force vectors at joints A, B, C | N |
| `F_O` | Compatibility alias for `F_A` | N |
| `N` | Slider normal reaction from guide | N |
| `F_f` | Slider friction force (kinetic Coulomb model) | N |
| `tau_A` (`τ_A`, `T`) | Motor torque applied on the crank at A (required drive torque) | N·m |
| `mu` (`μ`) | Coefficient of friction | dimensionless |
| `rho` (`ρ`) | Material density | kg/m³ |
| `g` | Gravitational acceleration | m/s² |
| `sigma_max` | Maximum normal stress over a full cycle | Pa |
| `tau_max` | Maximum shear stress over a full cycle | Pa |
| `sigma_allow` | Allowable normal stress limit | Pa |
| `tau_allow` | Allowable shear stress limit | Pa |
| `utilization` | `max(sigma_max/sigma_allow, tau_max/tau_allow)` | dimensionless |
| `pass_fail` | Design label (1 = pass, 0 = fail) | binary |
| `RPM` | Crank rotational speed (input setting) | rev/min |

---

## 9. Role of machine learning

Machine learning does not replace physical modeling in this project.

Instead, it is used to:
 • approximate the relationship between geometry and peak stress,
 • rapidly classify designs as likely pass or fail,
 • enable fast exploration of large design spaces.

All training data is generated using physics-based equations.

⸻

# Repository File Tree and Responsibilities (Authoritative)

Each file below is listed **once** with its responsibility stated **inline**, so an AI agent or developer can immediately understand scope, ownership, and behavior.

---

![Screenshot](assets/filepath.jpg)

## File and Directory Responsibilities

| Path | Description |
|-----|-------------|
| `README.md` | High-level project overview for mechanical engineering students |
| `TECHNICAL_SPEC.md` | Authoritative technical specification for developers and AI agents |
| `configs/` | YAML experiment definitions (no executable logic) |
| `configs/generate/` | Data generation configurations |
| `configs/train/` | ML training configurations |
| `configs/optimize/` | Optimization and inference configurations |
| `src/mech390/config.py` | Configuration loading, numeric normalization, and range validation helpers |
| `src/mech390/physics/kinematics.py` | Position, velocity, acceleration for slider (x-axis constrained) and crank pin (2D circular motion); all quantities returned as `np.ndarray([x, y])`; ROM and QRR metrics |
| `src/mech390/physics/dynamics.py` | Newton–Euler 8x8 joint-reaction solver (`A/B/C`, `N`, `F_f`, `tau_A`) with compatibility key `F_O` |
| `src/mech390/physics/mass_properties.py` | Center-of-gravity vectors, masses, mass moments, and area moments (`Iyy`, `Izz`) with a design-level aggregator |
| `src/mech390/physics/stresses.py` | Normal and shear stress calculations |
| `src/mech390/physics/engine.py` | 15° crank-angle sweep; orchestrates kinematics → dynamics → stresses; tracks peak sigma and tau |
| `src/mech390/datagen/stage1_kinematic.py` | 2D kinematic synthesis using exact closed-form r formula; feasibility filtering |
| `src/mech390/datagen/stage2_embodiment.py` | Multi-variant 3D embodiment generation with streaming iterator and width/pin feasibility constraints |
| `scripts/generate_dataset.py` | CLI entry point for dataset generation |
| `scripts/train_model.py` | CLI entry point for ML training |
| `scripts/optimize_config.py` | CLI entry point for ML-based design evaluation |
