# Debugging Summary: Objective + Sensitivity Pipeline

## Root causes identified

1. **Pressure sensitivity was weak in objective despite hydraulic response being present.**
   - Permeability `k` and thickness `b` already affected hydraulic impedance and pressure drop via `compute_pairwise_impedance(...)`.
   - However, the scalar objective only used pressure effects when a hard pressure constraint penalty activated, and otherwise ignored pressure-drop magnitude.
   - This made `k` appear nearly disconnected from objective in many feasible points.

2. **Injection temperature `T_inj` sensitivity was masked by normalization.**
   - Thermal power `P_avg_w` responded to `T_inj` correctly.
   - But `P_avg_norm = P_avg_w / (q c (T0 - T_inj))` used a denominator that changes with `T_inj` in almost the same proportion as numerator.
   - This ratio largely canceled the injection-temperature effect in objective calculations.

3. **Depth was not physically connected in this reduced-order pipeline.**
   - `depth` was swept in `scripts/run_sensitivity.py` but never passed into physics models (hydraulic or thermal) in a way that changed outputs.
   - Therefore it was a dummy sensitivity parameter in the current implementation.

## What was changed

### 1) Objective and performance metrics were made transparent and connected
- Updated objective form in `models/performance_metrics.py` to include an explicit pressure term:
  - `J = -w_power*P_avg_norm + w_flow_cv*CV_prod + w_pressure*max_pressure_drop_norm + w_penalty*constraint_penalty`.
- Added explicit penalty components:
  - `pressure_penalty`, `spacing_penalty`, `thermal_penalty` (currently zero unless added later), and aggregate `constraint_penalty`.
- Added richer scalar metrics for diagnostics, including:
  - pressure-normalized terms,
  - breakthrough summary metrics,
  - reservoir reference temperature,
  - depth activity flag.
- Added `objective_components` to expose terms directly.

### 2) Fixed `T_inj` masking in normalization
- Changed default thermal reference normalization to a reservoir-side reference independent of `T_inj`:
  - from `q*c*(T0 - T_inj)`
  - to `q*c*(T0 - 273.15)`
- This preserves thermal-term comparability while allowing `T_inj` perturbations to propagate to objective.

### 3) Sensitivity script upgraded to diagnose disconnections
- `scripts/run_sensitivity.py` now writes many raw component metrics per run (objective, thermal, hydraulic, penalties, objective components).
- Added classification summary for each parameter and layout:
  - affects hydraulic only,
  - affects thermal only,
  - affects both,
  - currently inactive / not connected.
- Added warning label when a parameter affects physics but not objective.
- Added new summary output file: `scripts/sensitivity_summary.csv`.

### 4) Tests added for end-to-end consistency
- Added `tests/test_performance_metrics_pipeline.py` to validate that:
  - `k` changes pressure and objective,
  - `T_inj` changes thermal power and objective,
  - depth is explicitly flagged inactive,
  - `q_total` and spacing scale affect coupled outputs.

## Status of the specific requested issues

- **Issue 1 (`k` changes pressure but not objective):** fixed by explicit pressure term in objective and by exposing pressure term diagnostics.
- **Issue 2 (`T_inj` changes power but not objective):** fixed by changing default normalization that previously canceled `T_inj` effect.
- **Issue 3 (depth no effect):** clarified as intentionally inactive in current reduced-order model and explicitly flagged as such in metrics and sensitivity summaries.

## Physical consistency checks (post-fix)

- `k` increase → pressure drop decreases (observed).
- `b` increase → pressure drop decreases and thermal capacity proxy improves via swept volume (observed).
- `T_inj` increase (fixed `T0`) → thermal power decreases (observed).
- `depth` currently inactive by design in this reduced-order script path and now labeled accordingly.
- `q_total` changes both thermal power and pressure-drop metrics (observed).
- spacing scale (`r_scale`) changes hydraulic and thermal metrics (observed).

## Remaining limitations

- Depth coupling (e.g., geothermal gradient, hydrostatic/wellbore coupling) is not yet included in this reduced-order coupled layout evaluator.
- Thermal penalty is scaffolded but remains zero unless additional thermal constraints are introduced.
