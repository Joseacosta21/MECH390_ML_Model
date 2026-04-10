# FULL TECHNICAL SPECIFICATION AND REPOSITORY CONTRACT  

## Offset Crank–Slider Data Generation, Simulation, and ML System  

**Audience:** Coding AI agents, automation systems, advanced developers
**Goal:** Enable full implementation with zero ambiguity

---

> Agent rules are in [CLAUDE.md](CLAUDE.md). Physics derivations are in [The_Mother_Doc_v7.md](The_Mother_Doc_v7.md).

---

## 1. Scope of the project (authoritative)

This repository defines a **complete physics-first pipeline** for synthesizing offset crank–slider mechanisms, expanding them into 3D parametric geometries, computing dynamics and stresses deterministically, labeling designs as pass/fail, and training ML models for rapid design evaluation and optimization. All results must be **reproducible**, **physically consistent**, and **traceable to configuration files**. See [README.md](README.md) for the pipeline diagram.

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

### 3.4 Mechanism geometry and coordinate system

This section is the authoritative reference for how the mechanism is oriented in space and how `volume_envelope` is computed. Read this before touching `generate.py` lines 452–466 or any code that uses `volume_envelope`.

#### Coordinate frame

```
         y (+)
         |
         |   B (crank pin, traces circle of radius r)
         |  /
         | /  r
         |/_________ x (+)   ← slider travel axis
         A (crank pivot, origin)
         |
         |  e  (offset — slider guide is e BELOW the crank pivot)
         |
    ===================================  slider guide (y = −e)
              C ──────────────► slider travels in +x
```

- **Origin**: crank pivot A at `(0, 0)`.
- **x-axis**: slider travel direction (positive = extended).
- **y-axis**: vertical, positive upward.
- **z-axis**: out-of-plane (depth), positive toward viewer.
- **Eccentricity `e`**: the slider guide runs at `y = −e`. The code's kinematic constraint absorbs this as the `(r·sinθ + e)` term in the rod-angle formula; `pos_C` is reported as `[x_C, 0]` with the offset implicit.

#### Assembly cross-section (z-axis view, looking from +z toward the mechanism)

The three bodies stack in z in the following order, centred on the slider guide plane:

```
z  →  more positive (toward viewer)

|← max(t_r, (t_s−t_l)/2) →|←── t_l/2 ──→|←── t_s/2 ──→|
|                           |              |              |
|      R (crank)            | L (rod)      | S (slider)   |
|                           |              |              |
                            ↑
                     contact plane
                  (R and L touch here;
                   L and S share centreline)
```

- **R (crank)** and **L (rod)** touch face-to-face. Their contact plane is the z-reference.
- **L (rod)** and **S (slider block)** share the same z-centreline. Each protrudes `t_l/2` and `t_s/2` respectively to the right of the contact plane.
- **Left of the contact plane**: either R (`t_r`) or the slider overhang `(t_s − t_l)/2` — whichever is larger.

#### Bounding box dimensions

**T — out-of-plane depth (z-axis)**

```
T = (t_l + t_s)/2  +  max(t_r, (t_s − t_l)/2)
```

- Right of contact plane: `(t_l + t_s)/2` — half of L plus half of S (both centred on same line).
- Left of contact plane: `max(t_r, (t_s − t_l)/2)` — crank thickness or slider overhang, whichever protrudes more.
- Note: `pin_diameter_A` does NOT contribute to T — the pin sits inside the crank bore and does not add to the bounding box depth.

Code (`generate.py:453`):
```python
_T = (_tl + _s_h) / 2.0 + max(_tr, (_s_h - _tl) / 2.0)
```

**H — vertical extent (y-axis)**

```
H = r  +  max(r, e + s_h/2)
```

- Top: crank pin B reaches `y = +r` (at θ = 90°).
- Bottom: lower of crank pin at `y = −r` (at θ = 270°) OR slider block bottom at `y = −(e + s_h/2)`.
- `max` selects whichever reaches further below the crank pivot.

Code (`generate.py:454`):
```python
_H = _r + max(_r, _e + _s_h / 2.0)
```

**L — horizontal extent (x-axis)**

```
L = r  +  √((r + l)² − e²)  +  s_l/2
```

- Left: crank pin B reaches `x = −r` (at θ = 180°).
- Right: slider pin C at maximum extension. When the crank and rod are collinear and the slider guide is at perpendicular distance `e` from A, the horizontal reach of C from A is `√((r+l)² − e²)`. The slider block adds `s_l/2` beyond pin C.

Code (`generate.py:455`):
```python
_L = _r + np.sqrt(max((_r + _l)**2 - _e**2, 0.0)) + _s_l / 2.0
```

---

> Pipeline overview: see [README.md §2](README.md). Full constraint derivations: see [The_Mother_Doc_v7.md](The_Mother_Doc_v7.md).

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
7. Slider Fx balance: `−F_Cx − mu·sign(v_sx)·N = (m_s + m_block)·a_Gsx`
8. Slider Fy balance: `−F_Cy + N = (m_s + m_block)·(a_Gsy + g)` — larger m_block → larger N → larger friction in eq 7 (coupled automatically)

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
- `utilization = max(sigma_max/sigma_limit, tau_max/tau_limit)`, where
  `sigma_limit = S_y / safety_factor` and `tau_limit = 0.577 * S_y / safety_factor` (Von Mises shear yield)
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
│  ├─ baseline.yaml     # Main run: 40 samples, LHS, 5 variants/2D → 200 designs
│  ├─ test_small.yaml   # Fast test config
│  └─ aggressive.yaml   # 🔲 EMPTY — wider ranges (not yet populated)
├─ train/
│  └─ surrogate.yaml    # Optuna sweep config (arch, dropout, lr, batch)
└─ optimize/
   └─ search.yaml       # Weight table + optimizer settings
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

#### `stresses.py` ✅ Implemented

- `rod_stresses(theta, design, dynamics_result)` → dict of per-angle rod σ and τ components
- `crank_stresses(theta, design, dynamics_result)` → dict of per-angle crank σ and τ components
- `pin_stresses(theta, design, dynamics_result)` → dict of per-angle pin σ and τ
- Saint-Venant torsion: `τ = T / (β·b·c²)` where `b = max(w,t)`, `c = min(w,t)` (Roark/Shigley)
- Axial hole stress: `σ = Kt_lug · F / ((w − D_hole) · t)` with `D_hole = D_pin + delta`
- The `1e-9` floor in the net-section denominator is no longer reachable in practice: Stage 2 rejects any geometry where `width - D_pin <= delta + 2·min_wall` before physics evaluation

#### `fatigue.py` ✅ Implemented

- `evaluate(sigma_rod_hist, tau_rod_hist, sigma_crank_hist, tau_crank_hist, sigma_pin_hist, tau_pin_hist, design)` → per-component fatigue dict
- Fatigue correction factors using `S'n = Sn * C_sur * C_s * C_st * C_R * C_m * C_f` (Mott Ch. 5)
  - `C_sur` = manufacturing method factor (as-machined = 0.88, from config `stress_analysis.C_sur`)
  - `C_s` = size factor computed from equivalent diameter (Mott Table 5-3)
  - `Sn` = 133 MPa — fatigue strength at design life 18.72×10⁶ cycles (Al 2024-T3, ASM)
- Modified Goodman safety factor `n_f`, ECY safety factor `n_y`, governing `n = min(n_f, n_y)`
- Basquin S-N (Miner's rule): `σa = A · N^b`, A = 924 MPa, b = −0.086
  - Source: AA2024-T3 experimental anchors (10⁷ cycles, 230 MPa) and (10⁹ cycles, 155 MPa)
  - `N_f = (σa_eq / A)^(1/b)`; valid range 10⁵–10⁹ cycles
- Miner's rule cumulative damage `D = N_design / N_f`; `failed_miner = D >= 1.0`
- All metrics returned with component suffix: `_rod`, `_crank`, `_pin`

#### `engine.py` ✅ Implemented

- `evaluate_design(design)` → 15° sweep, calls kinematics → dynamics → stresses → buckling → fatigue
- Returns summary keys: `sigma_max`, `tau_max`, `theta_sigma_max`, `theta_tau_max`, `valid_physics`, `n_buck`, `P_cr`, `N_max_comp`, `buckling_passed`, plus all fatigue keys
- Also returns per-angle history lists for CSV export:
  - `kinematics_history` — list of dicts (one per angle): `angle_deg`, `x_C`, `v_Cx`, `a_Cx`, `pos_Bx/By`, `vel_Bx/By`, `acc_Bx/By`, `phi_rad`, `omega_cb`, `alpha_cb`
  - `dynamics_history`   — list of dicts: `angle_deg`, `F_Ax/Ay`, `F_Bx/By`, `F_Cx/Cy`, `N`, `F_f`, `tau_A`
  - `stresses_history`   — list of dicts: `angle_deg`, `sigma_rod`, `tau_rod`, `sigma_crank`, `tau_crank`, `sigma_pin`, `tau_pin`, `sigma`, `tau`

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

#### `stage2_embodiment.py` ✅ Implemented

- `iter_expand_to_3d(valid_2d_designs, config)` → streaming iterator of 3D variants
- `expand_to_3d(valid_2d_designs, config)` → list wrapper
- Net-section feasibility enforced: `width - D_pin > delta + 2·min_wall` for all four pin pairs; `delta` and `min_wall` read from `config.stress_analysis`
- Rounding applied BEFORE constraint check: widths/thicknesses to `resolution_mm`, pin diameters to `pin_resolution_mm`
- `_round_to_res(value, resolution_m)` — noise-free rounding using decimal-place inference
- Seed diversification: `design_seed = base_seed + design_idx * 9973`
- Mass properties computed via `compute_design_mass_properties`; injected into design dict before engine call

#### `generate.py` ✅ Implemented

- `generate_dataset(config, seed)` → `DatasetResult` with seven DataFrames:
  `kinematics_df`, `dynamics_df`, `stresses_df`, `fatigue_df`, `buckling_df`, `passed_df`, `failed_df`
- Streaming Stage-2 consumption via `iter_expand_to_3d`
- Injects `omega`, `mu`, `g`, `alpha_r`, material props, and stress-analysis constants before engine call
- `_compute_checks` evaluates configured pass/fail checks; a design passes only if ALL pass:
  - Static: `utilization = max(sigma_max/sigma_limit, tau_max/tau_limit) <= utilization_max`,
    where `sigma_limit = yield_stress/safety_factor` and `tau_limit = yield_shear_stress/safety_factor`
  - Static FoS: `n_static_rod/crank/pin >= n_static_*_min`
  - Buckling: `n_buck >= n_buck_min`
  - Fatigue Goodman: `n_rod, n_crank, n_pin >= n_fatigue_*_min`
  - Miner's rule: `D_rod, D_crank, D_pin < D_miner_*_max`
- Designs with `valid_physics=False` or mass-property failures are silently dropped
- All DataFrames are self-contained (geometry columns repeated on every row)

---

### `scripts/`

**Purpose:** Entry points only. Each script loads config and calls library code.

| Script | Status | Description |
|---|---|---|
| `preview_stage1.py` | ✅ Complete | CLI: runs Stage 1, streams to CSV. Args: `--config`, `--seed`, `--out-dir` |
| `preview_stage2.py` | ✅ Complete | CLI: runs Stage 1 → Stage 2, computes mass properties, streams 27-column CSV. Args: `--config`, `--seed`, `--out-dir`, `--max-2d` |
| `preview_forces.py` | ✅ Complete | CLI: full pipeline → per-angle force sweep CSV (4800 rows). Args: `--config`, `--seed`, `--out-dir`, `--max-2d` |
| `generate_dataset.py` | ✅ Complete | CLI: full pipeline → 7 CSVs. Args: `--config`, `--seed`, `--out-dir` |
| `train_model.py` | ✅ Complete | CLI: Optuna sweep → saves checkpoint + scaler to `data/models/` |
| `optimize_design.py` | ✅ Complete | CLI: surrogate optimizer → top-N candidates via differential evolution |
| `validate_candidate.py` | ✅ Complete | CLI: feeds exact 10-variable geometry dict to physics engine; bypasses Stage 1; prints full pass/fail report. Edit `CANDIDATE` at top of file. Args: `--config` |
| `visualize_design.py` | 🔲 Planned | CLI: 2D mechanism drawing from `design_id`. Args: `--csv`, `--id`, `--angle`. ~130 lines |

---

### `data/`

**Purpose:** Generated artifacts only.

```
data/
├─ preview/    # All outputs: preview_*.py scripts + generate_dataset.py (default)
├─ runs/       # Named production runs: --out-dir data/runs/<name>
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

## 11. Determinism and reproducibility rules

- All physics functions must be deterministic
- Randomness only allowed in `sampling.py` and `stage1_kinematic.py` (constrained candidate generation)
- All runs must be seed-controlled via `config.random_seed`
- Stage-2 seed diversification: `design_seed = base_seed + design_idx * 9973`
- Outputs must be traceable to configs

---

## 12. Repository rules

- **Empty Directories (`.gitkeep`):** Git does not track empty directories. Placeholder `.gitkeep` files are used locally to enforce folder structures (e.g. `data/models/`). **Rule:** When you add a new file to a directory that only has a `.gitkeep`, you MUST remove the `.gitkeep` file.
