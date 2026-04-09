# CODEx Upgrade Summary

## What was changed
- Added a new analytical thermal decline module (`models/thermal_decline.py`) with producer-level and pattern-level outputs (`G(t)`, temperatures, power, breakthrough proxies, and time-averaged power).
- Added coupled hydraulic + thermal performance evaluation and objective composition (`models/performance_metrics.py`).
- Added thesis-oriented optimization workflows (`optimizers/pattern_optimization.py`) including:
  - inverted five-spot spacing sweep
  - 3 injector / 5 producer differential evolution optimization.
- Added scripts for thesis workflows:
  - `scripts/demo_coupled_layout_optimization.py`
  - `scripts/run_validation.py`
  - `scripts/run_sensitivity.py`
- Added tests for thermal decline and validation benchmarks:
  - `tests/test_thermal_decline.py`
  - `tests/test_validation_cases.py`

## New assumptions
- Thermal decline follows an equivalent-volume no-conduction model:
  - `G(t) = (1 + t/tau)^(-1)`
  - `tau = V_eff * rho_eff * c_eff / (m_dot * c_CO2)`.
- Effective swept volume is mapped from existing 3-injector/5-producer geometric allocation (`swept_volumes_3inj5prod`) with fallback equal partitioning for non-matching patterns.
- Wellbore thermal losses are not part of primary thermal model (consistent with thesis scope).

## How to run
- Demo: `python scripts/demo_coupled_layout_optimization.py`
- Validation: `python scripts/run_validation.py`
- Sensitivity analysis: `python scripts/run_sensitivity.py`
- Tests: `pytest -q`

## What remains approximate
- Thermal validation currently uses analytically-checkable synthetic benchmarks (shape and scaling checks) rather than fixed literature datasets.
- Placeholder hooks are present in validation script output for literature-calibrated benchmarks.
- Depth perturbation in sensitivity is reported as a proxy row but not directly coupled into a full thermophysical property model.

## Literature-specific calibration still needed
- Insert reference datasets for dimensionless `G(t)` curves from selected benchmark studies.
- Add target values and acceptance tolerances for at least one published impedance benchmark and one thermal decline benchmark.
- Optionally calibrate effective thermal properties (`rho_eff`, `c_eff`, `c_CO2`) against chosen reservoir/fluid conditions from thesis reference cases.
