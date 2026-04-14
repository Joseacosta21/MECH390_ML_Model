# FULL TECHNICAL SPECIFICATION AND REPOSITORY CONTRACT  

## Offset Crank–Slider Data Generation, Simulation, and ML System  

**Audience:** Coding AI agents, automation systems, advanced developers
**Goal:** Full implementation, zero ambiguity

---

> Agent rules in [CLAUDE.md](CLAUDE.md). Physics derivations in [The_Mother_Doc_v7.md](The_Mother_Doc_v7.md).

---

## 1. Scope (authoritative)

Repo: **complete physics-first pipeline** — synthesize offset crank–slider mechanisms, expand to 3D parametric geometries, compute dynamics/stresses deterministically, label pass/fail, train ML for rapid evaluation/optimization. Results must be **reproducible**, **physically consistent**, **config-traceable**. See [README.md](README.md) for pipeline diagram.

---

## 2. Hard constraints (non-negotiable)

- **Constant RPM** — angular speed fixed; RPM NOT a design variable
- **ROM** — target slider stroke: `ROM_target = 250 mm`; hard acceptance target:

    ```
    |ROM_computed - ROM_target| <= ROM_tolerance
    ```
    Current: **ROM_tolerance = 0.0005 m (±0.5 mm)**
- **QRR** — must satisfy:

    ```
    QRR_min ≤ QRR ≤ QRR_max
    ```

  RPM constant → time ratios = crank-angle ratios

Constraint violation → invalid; never reaches stress evaluation.

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

- mass per body
- CG per body — `np.ndarray([x, y])`
- mass MOI per body
- area MOI of link/slider sections (bending) — `Iyy`, `Izz`
- joint reaction forces — `np.ndarray([Fx, Fy])`
- crank drive torque `τ_A` (also `T`; code key `tau_A`)
- normal and shear stresses

---

### 3.4 Mechanism geometry and coordinate system

Authoritative reference for mechanism orientation and `volume_envelope` computation. Read before touching `generate.py` lines 452–466 or any `volume_envelope` code.

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
- **Eccentricity `e`**: slider guide at `y = −e`. Kinematic constraint absorbs as `(r·sinθ + e)` in rod-angle formula; `pos_C` reported as `[x_C, 0]`, offset implicit.

#### Assembly cross-section (z-axis view, looking from +z)

Three bodies stack in z, centred on slider guide plane:

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

- **R** and **L** touch face-to-face; contact plane = z-reference.
- **L** and **S** share z-centreline; each protrudes `t_l/2` and `t_s/2` right of contact plane.
- **Left of contact plane**: `t_r` or `(t_s − t_l)/2` — whichever larger.

#### Bounding box dimensions

**T — out-of-plane depth (z-axis)**

```
T = (t_l + t_s)/2  +  max(t_r, (t_s − t_l)/2)
```

- Right of contact plane: `(t_l + t_s)/2`
- Left of contact plane: `max(t_r, (t_s − t_l)/2)`
- Note: `pin_diameter_A` does NOT contribute to T — pin inside crank bore, no added depth.

Code (`generate.py:453`):
```python
_T = (_tl + _s_h) / 2.0 + max(_tr, (_s_h - _tl) / 2.0)
```

**H — vertical extent (y-axis)**

```
H = r  +  max(r, e + s_h/2)
```

- Top: crank pin B at `y = +r` (θ = 90°).
- Bottom: lower of crank pin at `y = −r` (θ = 270°) OR slider bottom at `y = −(e + s_h/2)`.

Code (`generate.py:454`):
```python
_H = _r + max(_r, _e + _s_h / 2.0)
```

**L — horizontal extent (x-axis)**

```
L = r  +  √((r + l)² − e²)  +  s_l/2
```

- Left: crank pin B at `x = −r` (θ = 180°).
- Right: slider pin C at max extension. Horizontal reach from A = `√((r+l)² − e²)`; slider adds `s_l/2`.

Code (`generate.py:455`):
```python
_L = _r + np.sqrt(max((_r + _l)**2 - _e**2, 0.0)) + _s_l / 2.0
```

---

> Pipeline overview: [README.md §2](README.md). Full constraint derivations: [The_Mother_Doc_v7.md](The_Mother_Doc_v7.md).

---

## 5. Authoritative kinematic equations (summary)

### Vector convention

All position/velocity/acceleration: `np.ndarray` shape `(2,)` = `[x, y]`.

- **Slider C** constrained to x-axis: `pos_C = [x_C, 0]`, `vel_C = [v_Cx, 0]`, `acc_C = [a_Cx, 0]`
- **Crank pin B** full circle: `pos_B = [r·cosθ, r·sinθ]`, nonzero x and y

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

Setting S = ROM:

    r = (S / 2) · √( (4(l² − D²) − S²) / (4l² − S²) )

Pre-formula feasibility:

| Condition | Meaning |
|---|---|
| `D < l` | Valid triangle |
| `S < 2l` | Denominator positive |
| `S < 2·√(l²−D²)` | Numerator positive; physical max stroke |

Post-compute feasibility:

| Condition | Meaning |
|---|---|
| `l > r + |D|` | Full-rotation validity (no lockup) |
| `l² + r² − D² − S²/2 >= 0` | Squared-equation branch consistency |
| `|S_original(r,l,D) − S_target| <= ROM_tolerance` | Rejects extraneous roots from squaring |

### Quick return ratio

    Δθ_forward = (θ_max − θ_min) mod 2π
    Δθ_return  = 2π − Δθ_forward
    QRR = Δθ_forward / Δθ_return

Dead-center detection: robust root-finding required.

---

## 6. Newton–Euler dynamics (8×8 linear system)

Unknown vector per crank angle θ:

    x = [F_Ax, F_Ay, F_Bx, F_By, F_Cx, F_Cy, N, tau_A]

Eight equations:
1. Crank Fx balance
2. Crank Fy balance (+ gravity)
3. Crank moment about G_r (includes tau_A)
4. Rod Fx balance
5. Rod Fy balance (+ gravity)
6. Rod moment about G_l
7. Slider Fx: `−F_Cx − mu·sign(v_sx)·N = (m_s + m_block)·a_Gsx`
8. Slider Fy: `−F_Cy + N = (m_s + m_block)·(a_Gsy + g)` — larger m_block → larger N → larger friction in eq 7 (coupled)

Condition number checked against `1e12`. Ill-conditioned → `ValueError`.

COG positions:
- Crank CG: `crank_cog = 0.5 * pos_B`
- Rod CG: `rod_cog = 0.5 * (pos_B + pos_C)`
- Slider CG: `slider_cog = pos_C`

Body accelerations:
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
- Net mass MOI about CG z-axis: rectangular prism minus two circular holes (parallel-axis)
- Gross area moments: `Iyy = w*t³/12`, `Izz = t*w³/12`

### Slider body

- Net plan area: `l*w − π/4*d_C²`
- Net volume: `plan_area_net * h`
- Net mass: `volume * rho`
- Net mass MOI: rectangular box minus centered hole
- Gross area moments: `Iyy = w*h³/12`, `Izz = h*w³/12`

### Aggregator

`compute_design_mass_properties(design, config)` → flat dict:
`rho`, `mass_crank`, `mass_rod`, `mass_slider`, `I_mass_crank_cg_z`, `I_mass_rod_cg_z`,
`I_mass_slider_cg_z`, `I_area_crank_yy`, `I_area_crank_zz`, `I_area_rod_yy`, `I_area_rod_zz`,
`I_area_slider_yy`, `I_area_slider_zz`  
(+ legacy aliases `I_area_slider_x`, `I_area_slider_y`).

Slider fixed dims (`length`, `width`, `height`) from `config.geometry.slider`.
Density from `config.material.rho` (fixed scalar or `{min: x, max: x}` equal bounds).

---

## 8. Dataset definition (authoritative schema)

Each row MUST include:

### Inputs

- `r, l, e`
- all 3D geometry params (`width_r`, `width_l`, `thickness_r`, `thickness_l`, `pin_diameter_A/B/C`)
- mass properties (`mass_crank`, `mass_rod`, `mass_slider`)
- mass MOI (`I_mass_crank_cg_z`, `I_mass_rod_cg_z`, `I_mass_slider_cg_z`)
- area MOI (`I_area_crank_yy`, `I_area_crank_zz`, `I_area_rod_yy`, `I_area_rod_zz`, `I_area_slider_yy`, `I_area_slider_zz`)
- kinematic metrics (`ROM`, `QRR`, `theta_min`, `theta_max`)

### Outputs

- `sigma_max` (max normal stress over cycle)
- `tau_max` (max shear stress over cycle)
- `utilization = max(sigma_max/sigma_limit, tau_max/tau_limit)`, where
  `sigma_limit = S_y / safety_factor` and `tau_limit = 0.577 * S_y / safety_factor` (Von Mises)
- `pass_fail` (binary: 1 = pass, 0 = fail)
- optional: `theta_sigma_max`, `theta_tau_max`

---

## 9. Repository structure and file contracts

```
mech390-crank-slider-ml/
```

---

### `CLAUDE.md`

**Purpose:** Defines subagents and mandatory usage rules for Claude Code.
Auto-read at session start. Contains: project context, 4 subagent definitions with triggers, mandatory subagent rule, known issues, quick-reference table.
No physics logic here — use `instructions.md`.

---

### `configs/`

**Purpose:** Define experiments. No code logic.

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

Each config:
- sampling ranges
- constraints
- Stage-2 embodiment controls (`n_variants_per_2d`, optional `stage2_max_attempts_per_2d`)
- output paths
- NO physics logic

Config loading: numeric normalization (incl. scientific notation) and range validation.

---

### `src/mech390/config.py`

**Implemented:**

- `load_config(config_path)` → loads YAML, normalizes numeric strings, validates `{min,max}` ranges
- `get_baseline_config()` → loads `configs/generate/baseline.yaml` relative to package
- `normalize_range_def(range_def, name)` → canonicalizes `{min,max}`, `[min,max]`, or scalar to `{'min': float, 'max': float}`
- `get_stage2_param_ranges(config)` → extracts all 7 Stage-2 param ranges; precedence: nested groups → legacy flat keys
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
- `calculate_metrics(r, l, e)` → dict: `ROM`, `QRR`, `theta_retracted`, `theta_extended`, `x_min`, `x_max`; or `{'valid': False, 'reason': str}`
- NO randomness

#### `dynamics.py` ✅ Implemented

- `solve_joint_reactions_newton_euler(theta, omega, r, l, e, mass_crank, mass_rod, mass_slider, I_crank, I_rod, mu, g, alpha_r, v_eps)` → dict: `F_A`, `F_B`, `F_C`, `F_O`, `N`, `F_f`, `tau_A`
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

- `rod_stresses(theta, design, dynamics_result)` → per-angle rod σ and τ components
- `crank_stresses(theta, design, dynamics_result)` → per-angle crank σ and τ
- `pin_stresses(theta, design, dynamics_result)` → per-angle pin σ and τ
- Saint-Venant torsion: `τ = T / (β·b·c²)` where `b = max(w,t)`, `c = min(w,t)` (Roark/Shigley)
- Axial hole stress: `σ = Kt_lug · F / ((w − D_hole) · t)` with `D_hole = D_pin + delta`
- `1e-9` floor in net-section denominator no longer reachable: Stage 2 rejects `width - D_pin <= delta + 2·min_wall` before physics

**Rod normal stress model (current — Eqs 4.4b / 4.5b):**
- Gravity distributed bending **removed**: rod self-weight in F_B, F_C via Newton-Euler — separate UDL double-counts
- OOP moment `M_eta = F_r_rod_B · i_offset` **constant along rod** — fully planar (2D), no ζ pin reactions → no linear decay
- Corner stress at Pin B: `σ = |F_r_B|/A_r + M_zeta_max·c_zr/I_zr + M_eta_rod·c_yr/I_yr`
- Corner stress at Pin C: `σ = |F_r_C|/A_r + M_zeta_max·c_zr/I_zr + M_eta_rod·c_yr/I_yr` (same M_eta — constant)
- `M_zeta_max = |F_t_B| · L/4` (Pin B end), `|F_t_C| · L/4` (Pin C end)
- `M_eta_rod = |F_r_B| · i_offset` (constant, same at B and C)
- Torsion: `T_rod = |F_t_C| · i_offset` (Eq 5.6)

**Crank normal stress model (current — Eqs 6.7b / 6.7c):**
- Gravity distributed bending removed (same reason as rod)
- `M_eta_crank = F_r_crank_B · i_offset` constant along crank (no ζ reactions)
- Corner at Pin B: `σ = |F_r_B|/A_c + M_zeta_max_B·c_zc/I_zc + M_eta_crank·c_yc/I_yc`
- Corner at Pin A: `σ = |F_r_A|/A_c + M_zeta_max_A·c_zc/I_zc + T_in·c_zc/I_zc + M_eta_crank·c_yc/I_yc`
- `T_in` enters as in-plane bending at A (NOT bar torsion: `T_in·ẑ · crank_axis = 0`)

**Pin stress model (current):**
- `slider_height` injected into design dict by `generate.py` and `validate_candidate.py` (reads from `config.geometry.slider.height`)
- Pin B bending: `M_pin_B = |F_B| · max(t_crank, t_rod) / 2` — governs by larger lug arm
- Shaft A torsion: `tau_shaft_A_torsion = 16·T_in / (π·d_shaft_A³)` — T_in directly at shaft
- Combined shear at A: `tau_pin_A_total = tau_pin_A + tau_shaft_A_torsion`
- Bearing at Pin C: rod side `F_r_rod_C / (2·D_pC·t_rod)` + slider side `F_r_rod_C / (D_pC·slider_height)` — both in max block

#### `fatigue.py` ✅ Implemented

- `evaluate(sigma_rod_hist, tau_rod_hist, sigma_crank_hist, tau_crank_hist, sigma_pin_hist, tau_pin_hist, design)` → per-component fatigue dict
- Fatigue correction: `S'n = Sn * C_sur * C_s * C_st * C_R * C_m * C_f` (Mott Ch. 5)
  - `C_sur` = manufacturing method factor (as-machined = 0.88, from `stress_analysis.C_sur`)
  - `C_s` = size factor from equivalent diameter (Mott Table 5-3)
  - `Sn` = 133 MPa — fatigue strength at 18.72×10⁶ cycles (Al 2024-T3, ASM)
- Modified Goodman `n_f`, ECY `n_y`, governing `n = min(n_f, n_y)`
- Basquin S-N (Miner's rule): `σa = A · N^b`, A = 924 MPa, b = −0.086
  - Source: AA2024-T3 experimental anchors (10⁷ cycles, 230 MPa) and (10⁹ cycles, 155 MPa)
  - `N_f = (σa_eq / A)^(1/b)`; valid 10⁵–10⁹ cycles
- Miner's cumulative damage `D = N_design / N_f`; `failed_miner = D >= 1.0`
- All metrics returned with component suffix: `_rod`, `_crank`, `_pin`

#### `engine.py` ✅ Implemented

- `evaluate_design(design)` → 15° sweep, calls kinematics → dynamics → stresses → buckling → fatigue
- Returns: `sigma_max`, `tau_max`, `theta_sigma_max`, `theta_tau_max`, `valid_physics`, `n_buck`, `P_cr`, `N_max_comp`, `buckling_passed`, plus all fatigue keys
- Per-angle history lists for CSV export:
  - `kinematics_history` — per angle: `angle_deg`, `x_C`, `v_Cx`, `a_Cx`, `pos_Bx/By`, `vel_Bx/By`, `acc_Bx/By`, `phi_rad`, `omega_cb`, `alpha_cb`
  - `dynamics_history` — per angle: `angle_deg`, `F_Ax/Ay`, `F_Bx/By`, `F_Cx/Cy`, `N`, `F_f`, `tau_A`
  - `stresses_history` — per angle: `angle_deg`, `sigma_rod`, `tau_rod`, `sigma_crank`, `tau_crank`, `sigma_pin`, `tau_pin`, `sigma`, `tau`

---

### `src/mech390/datagen/`

#### `sampling.py` ✅ Implemented

- `sample_scalar(range_def, seed)` — uniform, discrete choice, or constant
- `LatinHypercubeSampler(param_ranges, n_samples, seed).generate()` — uses `scipy.stats.qmc.LatinHypercube`
- `get_sampler(method, param_ranges, n_samples, seed)` — factory for `latin_hypercube` and `random`
- NO physics imports

#### `stage1_kinematic.py` ✅ Implemented

- `solve_for_r_given_rom(l, e, target_rom, r_min, r_max, rom_tolerance)` → float or `None`
- `iter_valid_2d_mechanisms(config, n_attempts)` → streaming iterator; draws in batches until `n_samples` VALID designs yielded (or `n_attempts` budget hit)
- `generate_valid_2d_mechanisms(config, n_attempts)` → list wrapper
- Internal: pre-feasibility constrained `(l,e)` sampling; rounding to `resolution_mm` for `l`, `e`, `r`; optional constraint expression evaluation
- `_round_to_res(value, resolution_m)` — rounds to nearest multiple, then `round(raw, decimal_places)` to eliminate binary float noise

#### `stage2_embodiment.py` ✅ Implemented

- `iter_expand_to_3d(valid_2d_designs, config)` → streaming iterator of 3D variants
- `expand_to_3d(valid_2d_designs, config)` → list wrapper
- Net-section feasibility enforced: `width - D_pin > delta + 2·min_wall` for all four pin pairs; `delta` and `min_wall` from `config.stress_analysis`
- Rounding BEFORE constraint check: widths/thicknesses to `resolution_mm`, pins to `pin_resolution_mm`
- `_round_to_res(value, resolution_m)` — noise-free rounding via decimal-place inference
- Seed diversification: `design_seed = base_seed + design_idx * 9973`
- Mass properties via `compute_design_mass_properties`; injected into design dict before engine call

#### `generate.py` ✅ Implemented

- `generate_dataset(config, seed)` → `DatasetResult` with seven DataFrames:
  `kinematics_df`, `dynamics_df`, `stresses_df`, `fatigue_df`, `buckling_df`, `passed_df`, `failed_df`
- Streaming Stage-2 via `iter_expand_to_3d`
- Injects `omega`, `mu`, `g`, `alpha_r`, material props, stress-analysis constants before engine call
- `_compute_checks` evaluates pass/fail; design passes only if ALL pass:
  - Static: `utilization = max(sigma_max/sigma_limit, tau_max/tau_limit) <= utilization_max`,
    where `sigma_limit = yield_stress/safety_factor` and `tau_limit = yield_shear_stress/safety_factor`
  - Static FoS: `n_static_rod/crank/pin >= n_static_*_min`
  - Buckling: `n_buck >= n_buck_min`
  - Fatigue Goodman: `n_rod, n_crank, n_pin >= n_fatigue_*_min`
  - Miner's rule: `D_rod, D_crank, D_pin < D_miner_*_max`
- `valid_physics=False` or mass-property failures: silently dropped
- All DataFrames self-contained (geometry columns repeated per row)

---

### `src/mech390/ml/`

ML stack fully data-size-agnostic. All dims (`input_dim`, `n_reg_targets`) derived at runtime from `features.py` constants — no hardcoded integers in training path. Adding features/targets: edit `features.py` and retrain.

#### `features.py` ✅ Implemented

Constants (authoritative — all modules derive dims from these):
- `INPUT_FEATURES` (list, currently 10) — design variables fed to NN
- `REGRESSION_TARGETS` (list, currently 8) — continuous outputs from regression head
- `CLASSIFICATION_TARGET` — `'pass_fail'` (binary)

Functions:
- `load_dataset(csv_pass, csv_fail)` → merged DataFrame; derives `min_n_static = min(n_static_rod, n_static_crank, n_static_pin)`; drops NaN rows
- `split_dataset(df, train_frac, val_frac, random_seed)` → `(train_df, val_df, test_df)` — stratified on `pass_fail`
- `fit_scaler(train_df)` → fitted `StandardScaler` on `INPUT_FEATURES` from train split only
- `get_arrays(df, scaler)` → `(X, y_clf, y_reg)` as `float32`; X shape `(N, len(INPUT_FEATURES))`, y_reg shape `(N, len(REGRESSION_TARGETS))`
- `compute_target_stats(train_df)` → `{target: {min, max}}` dict; used by optimizer to normalize objectives to [0, 1]
- `normalize_targets(y_reg, target_stats)` / `denormalize_targets(y_norm, target_stats)` — min-max to [0, 1] before MSE; denormalize before reporting or optimizer
- `save_scaler(scaler, path)` / `load_scaler(path)` — pickle persistence

#### `models.py` ✅ Implemented

- `CrankSliderSurrogate(input_dim, hidden_sizes, n_reg_targets, dropout_rate, use_batch_norm)` — shared ReLU FC trunk → classification head (1 output, BCEWithLogitsLoss) + regression head (`n_reg_targets` outputs, MSELoss)
  - `forward(x)` → `(logit_clf, pred_reg)` — apply `sigmoid` externally for pass probability
  - `predict_pass_prob(x)` → pass probability in [0, 1]
- `save_checkpoint(model, optimizer_state, epoch, val_f1, hparams, path)` — saves full state dict + hparams; hparams must include `input_dim`, `n_reg_targets`, `use_batch_norm` for reconstruction without source
- `load_checkpoint(path, device)` → raw dict
- `build_model_from_hparams(hparams)` — reconstructs from checkpoint hparams; raises `KeyError` on missing key (fail-fast on stale checkpoints)

#### `train.py` ✅ Implemented

- `run_training(cfg)` — full pipeline: load data → StandardScaler → normalize regression targets → Optuna sweep → save best checkpoint + scaler + target_stats
- `_train_one_trial(hparams, loaders, cfg, device)` → `(best_val_f1, model)` — trains one config; early stopping on `val_f1 > best_val_f1` (not val_loss)
- `_make_objective(loaders, cfg, device, best_tracker)` → Optuna objective; tracks globally best model across all trials
- All dims from `len(F.INPUT_FEATURES)` and `len(F.REGRESSION_TARGETS)` — no hardcoded integers
- Config: `configs/train/surrogate.yaml`

#### `infer.py` ✅ Implemented

- `SurrogatePredictor(checkpoint, scaler_path, stats_path, device)` — loads checkpoint, validates `ckpt['hparams']['n_reg_targets'] == len(F.REGRESSION_TARGETS)` (raises `ValueError` on mismatch), builds model, loads scaler
- `predict(design)` — single dict, list of dicts, or DataFrame; returns dict (single) or DataFrame (batch); keys: `pass_prob`, `pass_fail_pred`, + all `REGRESSION_TARGETS` in physical units (denormalized)

#### `optimize.py` ✅ Implemented

- `run_optimization(generate_cfg, optimize_cfg, predictor)` → top-N candidates ranked by weighted score
- Uses `scipy.optimize.differential_evolution` over bounds from `baseline.yaml`
- Score: weighted sum of normalized regression targets (from `search.yaml`) penalized by `pass_prob < threshold`
- Three hard analytical constraints as penalties (see CLAUDE.md — Optimizer Constraints):
  1. Net-section feasibility: `width - D_pin > delta + 2×min_wall`
  2. Kinematic feasibility: `l > r + e`
  3. Euler buckling: `P_cr = π²EI/l²`; penalizes `n_buck < 3.0`
- **OOD penalty (ML-P2):** Predictions outside training range by more than `ood_tolerance` (default 10%) penalized as `weight × ood_excess × ood_penalty_scale` (default 10.0). Per-objective before direction flip. Both params configurable in `search.yaml` under `constraints`.

---

### `scripts/`

**Purpose:** Entry points only. Each script loads config and calls library code.

| Script | Status | Description |
|---|---|---|
| `preview_stage1.py` | ✅ Complete | CLI: Stage 1, streams to CSV. Args: `--config`, `--seed`, `--out-dir` |
| `preview_stage2.py` | ✅ Complete | CLI: Stage 1 → Stage 2, mass properties, streams 27-column CSV. Args: `--config`, `--seed`, `--out-dir`, `--max-2d` |
| `preview_forces.py` | ✅ Complete | CLI: full pipeline → per-angle force sweep CSV (4800 rows). Args: `--config`, `--seed`, `--out-dir`, `--max-2d` |
| `generate_dataset.py` | ✅ Complete | CLI: full pipeline → 7 CSVs. Args: `--config`, `--seed`, `--out-dir` |
| `train_model.py` | ✅ Complete | CLI: Optuna sweep → saves checkpoint + scaler to `data/models/` |
| `optimize_design.py` | ✅ Complete | CLI: surrogate optimizer → top-N candidates via differential evolution |
| `validate_candidate.py` | ✅ Complete | CLI: feeds exact 10-variable geometry dict to physics engine; bypasses Stage 1; prints full pass/fail report. Edit `CANDIDATE` at top. Args: `--config` |
| `run_pipeline.py` | ✅ Complete | CLI: end-to-end orchestrator — generate → train → optimize. Args: `--generate-config`, `--train-config`, `--optimize-config`, `--seed`, `--out-dir`, `--skip-datagen`, `--skip-training`, `--log-level` |
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

- All physics functions deterministic
- Randomness only in `sampling.py` and `stage1_kinematic.py`
- All runs seed-controlled via `config.random_seed`
- Stage-2 seed diversification: `design_seed = base_seed + design_idx * 9973`
- Outputs traceable to configs

---

## 12. Repository rules

- **Empty Directories (`.gitkeep`):** Git skips empty dirs; `.gitkeep` files used locally to enforce folder structure (e.g. `data/models/`). **Rule:** When adding a file to a `.gitkeep`-only dir, MUST remove the `.gitkeep`.