# Codex Session Notes

Date: 2026-04-06

## What We Talked About

- Fatigue nomenclature cleanup:
  - Replace old endurance naming with:
    - `Sn` = uncorrected fatigue strength
    - `S'n` (implemented as `S_n_prime` in code) = corrected fatigue strength
  - Use formula:
    - `S'n = Sn * C_s * C_st * C_R * C_m * C_f`
  - Remove temperature factor from fatigue correction path.

- Baseline configuration linkage:
  - Add fatigue factors to config and wire them into generation/evaluation.
  - Make pass/fail thresholds explicit in `limits`.

- Static stress allowables:
  - Remove placeholder `sigma_allow` / `tau_allow`.
  - Derive static limits from material properties:
    - `sigma_limit = yield_stress / safety_factor`
    - `tau_limit = yield_shear_stress / safety_factor`

- Documentation alignment:
  - Update markdown docs to match current code behavior.

- Clarifications requested:
  - Pin checks include per-pin bearing and shear terms (worst-case aggregated).
  - Buckling currently checks rod only.
  - Both overall (`sigma_max`) and per-component stress peaks exist.
  - `safety_factor` currently affects static stress checks only.

## What I Changed

### 1) Fatigue naming + factors

- Updated fatigue implementation to use `Sn` and `S_n_prime`.
- Replaced old factor path with `C_st`, `C_R`, `C_m`, `C_f` (with `C_s` computed in fatigue module).
- Removed temperature-factor usage in the correction formula.

Files:
- `src/mech390/physics/fatigue.py`
- `src/mech390/datagen/generate.py`
- `src/mech390/physics/engine.py`

### 2) Config updates

- Added to `configs/generate/baseline.yaml`:
  - `material.Sn`
  - `stress_analysis.C_st`, `C_R`, `C_f`, `C_m`
- Added explicit limits fields (pass/fail thresholds):
  - `utilization_max`
  - `n_static_min` (plus optional per-component overrides)
  - `n_fatigue_rod_min`, `n_fatigue_crank_min`, `n_fatigue_pin_min`
  - `n_buck_min`
  - `D_miner_max` (plus optional per-component overrides)

### 3) Static-limit derivation from material

- Removed direct use of `limits.sigma_allow` and `limits.tau_allow`.
- Static checks now use material-derived limits:
  - `yield_stress` and `yield_shear_stress`, scaled by `limits.safety_factor`.
- Updated small test config accordingly.

Files:
- `src/mech390/datagen/generate.py`
- `configs/generate/baseline.yaml`
- `configs/generate/test_small.yaml`
- `tests/test_datagen_units.py`

### 4) Docs updated

Files updated:
- `instructions.md`
- `README.md`

Also added TODO item:
- `Size factor formula check` in `README.md`.

## Current Pass/Fail Logic (Now)

A design passes only if all checks pass:

1. Static utilization:
   - `utilization = max(sigma_max/sigma_limit, tau_max/tau_limit)`
   - pass if `utilization <= utilization_max`

2. Static FoS per component:
   - `n_static_rod/crank/pin >= n_static_*_min`

3. Buckling:
   - rod-only Euler buckling check
   - `n_buck >= n_buck_min`

4. Fatigue governing FoS:
   - `n_rod`, `n_crank`, `n_pin >= n_fatigue_*_min`

5. Miner damage:
   - `D_rod`, `D_crank`, `D_pin < D_miner_*_max`

## What Should Be Done Next

1. Regenerate outputs
- Re-run dataset generation so CSVs match new nomenclature and thresholds.

2. Decide on utilization vs FoS-first reporting
- Keep both (for compatibility) or shift to FoS-first columns as primary outputs.

3. Optional physics scope expansion
- Add crank buckling check if required by project criteria.
- Consider exposing per-pin detailed outputs (A/B/C) in exported CSVs, not only worst-case `sigma_pin` / `tau_pin`.

4. Validate equations
- Complete the backlog item: `Size factor formula check`.

5. Tests/runtime verification
- Run full unit/integration tests in an environment with required dependencies installed.

