# FULL TECHNICAL SPECIFICATION AND REPOSITORY CONTRACT  

## Offset Crank–Slider Data Generation, Simulation, and ML System  

**Audience:** Coding AI agents, automation systems, advanced developers
**Goal:** Enable full implementation with zero ambiguity

---

## 0. Agent rules (mandatory — read before anything else)

**You must invoke at least one subagent from `CLAUDE.md` for any substantive task.**

`CLAUDE.md` (project root) defines four subagents:
- **Physics Validator** — for any physics file edit or equation question
- **Cross-Reference Auditor** — for any signature change or cross-file consistency check
- **Data Quality Checker** — after any data generation run
- **ML Readiness Inspector** — before any training run

Read `CLAUDE.md` now if you have not already done so.

### Known bugs (confirmed — do not ignore)

These issues are documented and must be considered whenever touching the affected files:

| # | File | Line | Issue | Impact |
|---|---|---|---|---|
| 1 | `src/mech390/datagen/generate.py` | ~144–150 | `omega` not set in design dict before `engine.evaluate_design()` — defaults to 1.0 rad/s instead of `RPM × 2π/60`. Mass properties also not merged before the physics call. | Forces scale as ω² — currently ~10× wrong when using `generate_dataset()` |
| 2 | `src/mech390/physics/kinematics.py` | 290 | Sign error: `a_By = -alpha2 * r * cos(theta)` should be `+alpha2 * r * cos(theta)` | No current impact (`alpha_r` always 0.0), but formula is wrong |
| 3 | `src/mech390/physics/mass_properties.py` | 208–209 | Hole offsets in `link_mass_moi_cg_z` use `±c/2`, exact only when `d_left == d_right` | Minor MOI error for asymmetric pin diameters |

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
    Current value: **ROM_tolerance = 0.0005 m (±0.5 mm)**
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
- area moment of inertia of link/slider sections (bending properties) — `Iyy`, `Izz`
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

1. Pre-filter feasible `(l, e)` domain before sampling (avoid wasted draws):
   - enforce `l > ROM/2 + eps`
   - enforce `e < sqrt(l² − ROM²/4) − eps` (numerator positivity)
2. Sample `(l, e)` pairs inside the constrained domain using `random` or `latin_hypercube`
3. Solve for `r` using the closed-form ROM relation
4. Apply branch-feasibility checks to reject extraneous analytical roots
5. Evaluate optional user-defined constraint expressions from config
6. Find dead-center crank angles via Brent root-finding
7. Compute ROM and QRR from kinematics
8. Accept only if both ROM tolerance (±0.5 mm) and QRR bounds are satisfied

**`n_samples` = target number of VALID designs to yield.**  
Candidates are generated in batches until exactly `n_samples` valid designs are produced
or the draw budget (`n_attempts`) is exhausted. This guarantees the dataset size
regardless of how many candidates are rejected.

NO dynamics. NO stresses.

---

### Stage 2 — 3D embodiment, dynamics, and stress

- Purpose: evaluate structural feasibility
- Inputs: valid 2D mechanisms
- Outputs: stress metrics + pass/fail labels

Operations:

1. Generate multiple 3D geometry variants for each valid 2D mechanism (controlled by `sampling.n_variants_per_2d`)
2. Sample widths, thicknesses, and pin diameters using the method in `sampling.method`
3. Enforce Stage-2 geometric constraints:
   - `width_r > pin_diameter_A`
   - `width_r > pin_diameter_B`
   - `width_l > pin_diameter_B`
   - `width_l > pin_diameter_C`
4. Compute mass and inertia properties via `mass_properties.compute_design_mass_properties`
5. Evaluate dynamics every 15° using a Newton–Euler 8×8 linear solve that returns:
   - `F_A`, `F_B`, `F_C` (joint reactions as `np.ndarray([Fx, Fy])`)
   - `N`, `F_f` (slider normal + kinetic Coulomb friction)
   - `tau_A` (required crank torque)
   - compatibility alias `F_O = F_A`
6. Compute stresses *(not yet implemented in `stresses.py`; engine uses 0.0 placeholder)*
7. Track maximum stress values over the full crank cycle
8. Apply pass/fail criteria using `sigma_allow`, `tau_allow`, and `safety_factor`

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

### Connecting rod angle (open branch)

    sin(φ) = −(e + r·sin θ) / l
    cos(φ) = +√(1 − sin²(φ))     (positive, open configuration)
    φ = arctan2(sin_φ, cos_φ)

### Connecting rod angular velocity and acceleration

    ω_cb = −V_By / (l·cos φ)
    α_cb = (ω_cb² · l · sin φ − a_By) / (l · cos φ)

### Dead centers

    dx_C / dθ = 0    (solved numerically via Brent root-finding)

### ROM

    ROM = x_C(θ_max) − x_C(θ_min)

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

## 6. Newton–Euler dynamics (8×8 linear system)

The unknown vector at each crank angle θ is:

    x = [F_Ax, F_Ay, F_Bx, F_By, F_Cx, F_Cy, N, tau_A]

Eight equations are assembled from:
1. Crank Fx balance
2. Crank Fy balance (+ gravity term)
3. Crank moment about G_r (includes tau_A)
4. Rod Fx balance
5. Rod Fy balance (+ gravity term)
6. Rod moment about G_l
7. Slider Fx balance (includes kinetic Coulomb friction: `−mu·sign(v_sx)·N`)
8. Slider Fy balance (+ N as slider guide reaction)

Condition number of the system matrix is checked against a limit of `1e12`. Ill-conditioned solves raise `ValueError`.

COG positions used by the force balance:
- Crank CG: midpoint of O–B → `crank_cog = 0.5 * pos_B`
- Rod CG: midpoint of B–C → `rod_cog = 0.5 * (pos_B + pos_C)`
- Slider CG: coincides with slider pin C → `slider_cog = pos_C`

Body accelerations used:
- `a_Gr = 0.5 * a_B`
- `a_Gl = 0.5 * (a_B + a_C)`
- `a_Gs = a_C`

---

## 7. Mass properties model

### Link bodies (crank and rod)

- Body length: `L = center_distance + 0.5*d_left + 0.5*d_right` (minimal tangency)
- Net plan area: `L*w − π/4*(d_left² + d_right²)`
- Net volume: `plan_area_net * t`
- Net mass: `volume * rho`
- Net mass MOI about CG z-axis: rectangular prism minus two circular holes (parallel-axis theorem)
- Gross area moments: `Iyy = w*t³/12`, `Izz = t*w³/12`

### Slider body

- Net plan area: `l*w − π/4*d_C²`
- Net volume: `plan_area_net * h`
- Net mass: `volume * rho`
- Net mass MOI: rectangular box minus centered hole
- Gross area moments: `Iyy = w*h³/12`, `Izz = h*w³/12`

### Aggregator

`compute_design_mass_properties(design, config)` returns a flat dict with keys:
`rho`, `mass_crank`, `mass_rod`, `mass_slider`, `I_mass_crank_cg_z`, `I_mass_rod_cg_z`,
`I_mass_slider_cg_z`, `I_area_crank_yy`, `I_area_crank_zz`, `I_area_rod_yy`, `I_area_rod_zz`,
`I_area_slider_yy`, `I_area_slider_zz`  
(+ legacy aliases `I_area_slider_x`, `I_area_slider_y`).

Slider fixed dimensions (`length`, `width`, `height`) are read from `config.geometry.slider`.
Density is read from `config.material.rho` (must be fixed scalar or `{min: x, max: x}` with equal bounds).

---

## 8. Dataset definition (authoritative schema)

Each dataset row MUST include:

### Inputs

- `r, l, e`
- all 3D geometry parameters (`width_r`, `width_l`, `thickness_r`, `thickness_l`, `pin_diameter_A/B/C`)
- mass properties (`mass_crank`, `mass_rod`, `mass_slider`)
- mass moments of inertia (`I_mass_crank_cg_z`, `I_mass_rod_cg_z`, `I_mass_slider_cg_z`)
- area moments of inertia (`I_area_crank_yy`, `I_area_crank_zz`, `I_area_rod_yy`, `I_area_rod_zz`, `I_area_slider_yy`, `I_area_slider_zz`)
- kinematic metrics (`ROM`, `QRR`, `theta_min`, `theta_max`)

### Outputs

- `sigma_max` (maximum normal stress over cycle)
- `tau_max` (maximum shear stress over cycle)
- `utilization = max(sigma_max/σ_allow, tau_max/τ_allow)`
- `pass_fail` (binary: 1 = pass, 0 = fail)
- optional: `theta_sigma_max`, `theta_tau_max` (crank angle at which maxima occur)

---

## 9. Repository structure and file contracts

```
mech390-crank-slider-ml/
```

---

### `CLAUDE.md`

**Purpose:** Defines available subagents and mandatory usage rules for Claude Code.
Automatically read by Claude Code at the start of every session.
Contains: project context, 4 subagent definitions with trigger conditions,
mandatory subagent rule, known issues reference, and quick-reference table for teammates.
Do not put physics logic here — use `instructions.md` for that.

---

### `configs/`

**Purpose:** Define experiments. No code logic here.

```
configs/
├─ generate/
│  ├─ baseline.yaml     # Full-scale run: 20M samples, LHS, 5 variants/2D
│  ├─ test_small.yaml   # Test run: 1000 samples, LHS, 5 variants/2D
│  └─ aggressive.yaml   # Wide-range generation config
├─ train/
│  ├─ regression.yaml
│  └─ classifier.yaml
└─ optimize/
   └─ search.yaml
```

Each config file:

- defines sampling ranges
- defines constraints
- can define Stage-2 embodiment controls (`n_variants_per_2d`, optional `stage2_max_attempts_per_2d`)
- defines output paths
- contains NO physics logic

Configuration loading is responsible for numeric normalization (including scientific notation) and range validation.

---

### `src/mech390/config.py`

**Implemented functions:**

- `load_config(config_path)` → loads YAML, normalizes all numeric strings, validates `{min,max}` ranges
- `get_baseline_config()` → loads `configs/generate/baseline.yaml` relative to package location
- `normalize_range_def(range_def, name)` → canonicalizes `{min,max}`, `[min,max]`, or scalar to `{'min': float, 'max': float}`
- `get_stage2_param_ranges(config)` → extracts all 7 Stage-2 parameter ranges with precedence: nested groups → legacy flat keys
- `get_stage2_sampling_settings(config)` → extracts `n_variants_per_2d` and `stage2_max_attempts_per_2d` with defaults

---

### `src/mech390/physics/`

**Purpose:** All reusable, deterministic physics logic.

#### `kinematics.py` ✅ Implemented

- `slider_position(theta, r, l, e)` → `np.ndarray([x_C, 0.0])`
- `slider_velocity(theta, omega, r, l, e)` → `np.ndarray([v_Cx, 0.0])`
- `slider_acceleration(theta, omega, r, l, e)` → `np.ndarray([a_Cx, 0.0])`
- `crank_pin_position(theta, r)` → `np.ndarray([r·cosθ, r·sinθ])`
- `crank_pin_velocity(theta, omega, r)` → `np.ndarray([-r·ω·sinθ, r·ω·cosθ])`
- `crank_pin_acceleration(theta, omega, r)` → `np.ndarray([-r·ω²·cosθ, -r·ω²·sinθ])`
- `rod_angle(theta, r, l, e)` → φ (open branch, positive cos)
- `rod_angular_velocity(theta, omega, r, l, e)` → ω_cb
- `rod_angular_acceleration(theta, omega, r, l, e, alpha2=0.0)` → α_cb
- `get_dead_center_angles(r, l, e)` → sorted `np.ndarray` of 2 roots, or empty array
- `calculate_metrics(r, l, e)` → dict with `ROM`, `QRR`, `theta_retracted`, `theta_extended`, `x_min`, `x_max`; or `{'valid': False, 'reason': str}`
- NO randomness

#### `dynamics.py` ✅ Implemented

- `solve_joint_reactions_newton_euler(theta, omega, r, l, e, mass_crank, mass_rod, mass_slider, I_crank, I_rod, mu, g, alpha_r, v_eps)` → dict of `F_A`, `F_B`, `F_C`, `F_O`, `N`, `F_f`, `tau_A`
- `joint_reaction_forces(theta, omega, r, l, e, mass_crank, mass_rod, mass_slider, **kwargs)` → backward-compatible wrapper
- Ill-conditioning guard: `cond(A) > 1e12` raises `ValueError`
- Uses kinematics and mass_properties COG helpers internally

#### `mass_properties.py` ✅ Implemented

- Kinematic COG helpers: `crank_cog`, `rod_cog`, `slider_cog` → `np.ndarray([x, y])`
- Link geometry: `link_body_length`, `link_plan_area_net`, `link_volume_net`, `link_mass`, `link_mass_moi_cg_z`, `link_area_moments_gross`
- Slider geometry: `slider_volume_net`, `slider_mass`, `slider_mass_moi_cg_z`, `slider_area_moments_gross`
- `MassPropertiesResult` frozen dataclass with `.to_dict()` (includes legacy aliases)
- `compute_design_mass_properties(design, config)` → flat dict aggregator

#### `stresses.py` 🔲 Stub

- Currently only imports `dynamics`
- Stress formulas not yet implemented

#### `fatigue.py` 🔲 Empty

- Reserved for future fatigue analysis

#### `engine.py` ✅ Implemented (stresses placeholder)

- `evaluate_design(design)` → 15° sweep, calls kinematics → dynamics → stresses (0.0 placeholder)
- Returns `sigma_max`, `tau_max`, `theta_sigma_max`, `theta_tau_max`, `valid_physics`

---

### `src/mech390/datagen/`

#### `sampling.py` ✅ Implemented

- `sample_scalar(range_def, seed)` — uniform, discrete choice, or constant
- `LatinHypercubeSampler(param_ranges, n_samples, seed).generate()` — uses `scipy.stats.qmc.LatinHypercube`
- `get_sampler(method, param_ranges, n_samples, seed)` — factory for `latin_hypercube` and `random`
- NO physics imports

#### `stage1_kinematic.py` ✅ Implemented

- `solve_for_r_given_rom(l, e, target_rom, r_min, r_max, rom_tolerance)` → float or `None`
- `iter_valid_2d_mechanisms(config, n_attempts)` → streaming iterator; keeps drawing in batches until `n_samples` VALID designs are yielded (or `n_attempts` budget is hit)
- `generate_valid_2d_mechanisms(config, n_attempts)` → list wrapper for compatibility
- Internal: pre-feasibility constrained `(l,e)` sampling; rounding to `resolution_mm` for `l`, `e`, `r`; optional constraint expression evaluation
- `_round_to_res(value, resolution_m)` — rounds to nearest multiple of `resolution_m`, then applies `round(raw, decimal_places)` to eliminate binary float noise

#### `stage2_embodiment.py` ✅ Implemented (stubs for mass props + stresses)

- `iter_expand_to_3d(valid_2d_designs, config)` → streaming iterator of 3D variants
- `expand_to_3d(valid_2d_designs, config)` → list wrapper
- Width/pin feasibility enforced: `width_r > pin_diameter_A/B`, `width_l > pin_diameter_B/C`
- Rounding applied BEFORE constraint check: widths/thicknesses to `resolution_mm`, pin diameters to `pin_resolution_mm`
- `_round_to_res(value, resolution_m)` — same noise-free implementation as Stage 1
- Seed diversification: `design_seed = base_seed + design_idx * 9973`
- Mass properties and stress calls marked as TODO stubs

#### `generate.py` ✅ Implemented

- `generate_dataset(config, seed)` → `DatasetResult(all_cases, pass_cases, summary)`
- Streaming Stage-2 consumption via `iter_expand_to_3d`
- Physics evaluation with fallback mock when engine unavailable
- `_apply_limits` computes `utilization` and `pass_fail` from `sigma_allow`, `tau_allow`, `safety_factor`

---

### `scripts/`

**Purpose:** Entry points only. Each script loads config and calls library code.

| Script | Status | Description |
|---|---|---|
| `preview_stage1.py` | ✅ Complete | CLI: runs Stage 1, streams to CSV. Args: `--config`, `--seed`, `--out-dir` |
| `preview_stage2.py` | ✅ Complete | CLI: runs Stage 1 → Stage 2, computes mass properties, streams 27-column CSV. Args: `--config`, `--seed`, `--out-dir`, `--max-2d` |
| `debug_stage1.py` | ✅ Complete | Quick debug runner using baseline config; prints first 5 designs + stats |
| `generate_dataset.py` | 🔲 Stub | Imports only |
| `train_model.py` | 🔲 Stub | Imports only |
| `optimize_config.py` | 🔲 Stub | Imports only |

---

### `data/`

**Purpose:** Generated artifacts only.

```
data/
├─ stage1_preview/
│  └─ stage1_geometries.csv   # Output of preview_stage1.py
├─ raw/
│  └─ <run_id>/
├─ processed/
├─ models/
└─ splits/
```

Never hard-code paths to this directory.

---

### `tests/`

| File | Description |
|---|---|
| `tests/test_datagen_units.py` | Unit tests for data generation pipeline |

---

### `reports/`

**Purpose:** Diagnostics and summaries.

```
reports/
├─ data_generation/
├─ training/
└─ optimization/
```

---

## 10. Expected usage pattern

1. Define or choose a generation config in `configs/generate/`
2. Run Stage 1 preview: `python scripts/preview_stage1.py --config configs/generate/baseline.yaml`
3. Run full dataset generation: `python scripts/generate_dataset.py` *(stub — implement CLI)*
4. Inspect summary reports
5. Train ML model: `python scripts/train_model.py` *(stub — implement CLI)*
6. Use ML for rapid evaluation or optimization: `python scripts/optimize_config.py` *(stub)*

All steps are repeatable and configuration-driven.

---

## 11. Determinism and reproducibility rules

- All physics functions must be deterministic
- Randomness only allowed in `sampling.py` and `stage1_kinematic.py` (constrained candidate generation)
- All runs must be seed-controlled via `config.random_seed`
- Stage-2 seed diversification: `design_seed = base_seed + design_idx * 9973`
- Outputs must be traceable to configs

---

## 12. Repository rules

- **Empty Directories (`.gitkeep`):** Git does not track empty directories. Placeholder `.gitkeep` files are used locally to enforce folder structures (e.g. `data/models/`). **Rule:** When you add a new file to a directory that only has a `.gitkeep`, you MUST remove the `.gitkeep` file.

---

## 13. Outstanding work (not yet implemented)

### Known bugs (fix these first)

| Bug | File | Fix |
|---|---|---|
| `omega` + mass props not injected before `engine.evaluate_design()` | `generate.py:144–150` | Compute mass props, merge into design dict, set `omega = RPM * 2π/60` before physics call |
| Sign error on `alpha2` in `rod_angular_acceleration` | `kinematics.py:290` | Change `-alpha2 * r * cos(theta)` to `+alpha2 * r * cos(theta)` |
| Hole offset approximation in `link_mass_moi_cg_z` | `mass_properties.py:208–209` | Use exact per-pin offsets from rectangle centroid instead of `±c/2` |

### Unimplemented features

| Item | Location | Notes |
|---|---|---|
| Stress formulas (normal + shear) | `stresses.py` | Required for real pass/fail labels |
| Mass+stress calls in Stage 2 | `stage2_embodiment.py` | Marked as TODO stubs |
| ML feature engineering | `ml/features.py` | Stub |
| ML model definitions | `ml/models.py` | Stub |
| ML training loop | `ml/train.py` | Stub |
| ML inference | `ml/infer.py` | Stub |
| `generate_dataset.py` CLI | `scripts/generate_dataset.py` | Stub — needs argparse + `generate_dataset()` call |
| `preview_stage2.py` mass+stress | `scripts/preview_stage2.py` | Currently computes mass props; stress calls pending `stresses.py` |
| `train_model.py` CLI | `scripts/train_model.py` | Stub |
| `optimize_config.py` CLI | `scripts/optimize_config.py` | Stub |
| Fatigue analysis | `fatigue.py` | Empty — future work |

---

## 14. Final contract statement

This document defines:

- all constraints
- all variables
- all equations
- all file responsibilities
- all expected behaviors
- the current implementation status of every module

Any implementation that follows this document is considered correct.

No assumptions beyond what is written here are allowed.
