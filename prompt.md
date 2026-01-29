%==============================================================================
% FINAL LaTeX PROMPT FOR CLAUDE
% PROJECT: CPG Well Field Optimization (3 Injectors / 5 Producers)
% CONFIG: Center Producer (1) + Inner Injector Ring (3) + Outer Producer Ring (4)
% CLOSURE CHOSEN: (B) Fixed per-well mass-rate allocation -> compute Δp at each well
% PRIMARY SOURCE: "CO2 mass and lifetime (2).pdf" (Factor2 Energy)
% SECONDARY: Ganjdanesh (2018), Birdsell (2024), Adams (2021)
%==============================================================================

Task: Well field optimization
Geothermal power plants represent a fundamental alternative to conventional power plants. Unlike wind and solar power, geothermal energy is available year-round without fluctuations. An alternative technology based on supercritical carbon dioxide ( sCO₂ ) has been the subject of numerous research studies for several years. Such Next Generation Power (NGP) systems offer several thermophysical advantages compared to conventional hydrothermal power plants, for example, due to a stronger thermosiphon effect.
Previous work confirmed the thermodynamic advantages and developed initial system optimizations. An initial assessment of the levelized cost of electricity (LCOE) demonstrated the fundamental competitiveness of this technology. To date, the work has focused on the design of the geothermal system, specifying injection and production wells and their distribution underground.
The aim of this work is to optimize the distribution of a specific number of boreholes (e.g., 3 injectors and 5 producers) in the subsurface so that, firstly, all boreholes are equally utilized, meaning that similar pressure losses occur across the injection and production boreholes, and secondly, the so-called cold The breakthrough at all production wells should occur at roughly the same time, if possible. Furthermore, the distribution and spacing between the various wells must adhere to specific requirements, such as minimum distances, to ensure the CO₂ has sufficient time to warm up . The development of fluid flow in the reservoir must also be considered , necessitating the creation of a corresponding model.

Work steps:
•	Literature review and study of previous works.
•	Creation of a model to simulate the reservoir system, the production and injection wells, taking into account the thermosiphon effect.
•	Development of a model to quantify the lifetime of boreholes based on the conditions to which they are exposed.
•	Definition of an optimization method with which the boreholes can be distributed within a defined area, taking into account the various constraints.
•	Conducting various simulations and comparisons to better understand the dynamics.
•	Documentation of the work.

Below is the specific prompt

Generate a clean, runnable Python implementation (modules + a Jupyter notebook demo)
for well-field optimization of a CO2-based closed-loop geothermal pattern:
1 center producer + 3 injectors on an inner ring + 4 producers on an outer ring.
The goal is to minimize pressure-drop non-uniformity (utilization factor) under fixed flow allocation,
while also estimating thermal lifetime using a conical frustum reservoir volume model.
Do NOT use geometric shortcuts as the objective; every iteration must evaluate the physics forward model.

------------------------------------------------------------------------------
A. DELIVERABLES (WHAT TO OUTPUT)
------------------------------------------------------------------------------
1) A small Python package (pure Python, no obscure deps):
   - config.py          : all constants in one place (easy to edit)
   - geometry.py        : well coordinates, constraints, distance matrix
   - hydraulics.py      : pressure potential, Δp at wells, pressure/velocity field on a grid
   - thermal.py         : frustum heat content, heat extraction rate, lifetime τ
   - breakthrough.py    : streamline plotting + time-of-flight (TOF) proxy and CV(TOF)
   - objective.py       : objective J + penalties
   - optimize.py        : SciPy differential_evolution wrapper
   - plots.py           : publication-style plots (layout, pressure contours, streamlines, metrics)
2) A Jupyter notebook "main.ipynb" that:
   - runs the optimizer,
   - prints best variables and metrics,
   - produces the required plots.

Use only: numpy, scipy, matplotlib, CoolProp.
If CoolProp is missing, code should raise a clear error explaining how to install it.

Add type hints + docstrings with the exact equations implemented.

------------------------------------------------------------------------------
B. GEOMETRY (WELL COORDINATES + OPT VARIABLES)
------------------------------------------------------------------------------
Target layout (2D areal, thickness H used in equations):
- Producer P0 at center: (0,0)
- Injectors I1..I3 on inner ring radius R_in at angles: 0, 2π/3, 4π/3 (fixed for stability)
- Outer producers P1..P4 on outer ring radius R_out with angles not forced to 90° symmetry.
  Their angles are optimization variables but must remain ordered and not overlap.

Optimization variables x:
- R_in [m]
- R_out [m], with constraint R_out >= R_in + ΔR_min
- θ0 [rad] : global rotation for outer ring
- ε1, ε2, ε3 [rad] : deviations from perfect 90° increments (for ordered non-overlapping angles)

Define outer producer angles as:
  θP1 = θ0
  θP2 = θ0 + (π/2 + ε1)
  θP3 = θP2 + (π/2 + ε2)
  θP4 = θP3 + (π/2 + ε3)
and enforce closure:
  ε4 = -(ε1 + ε2 + ε3) so that total sum of increments = 2π,
  with final increment (π/2 + ε4) > 0.

Thus only 6 variables total: [R_in, R_out, θ0, ε1, ε2, ε3].

Hard geometric constraints (penalize if violated):
- Minimum well spacing: min_{a<b} L_ab >= S_min
- Radial gap: R_out - R_in >= ΔR_min
- Outer angle increments must stay positive: (π/2 + εk) >= Δθ_min for k=1..4

Bounds (default; expose in config.py):
- R_in ∈ [500, 1500]
- R_out ∈ [R_in + 500, 4000]
- θ0 ∈ [0, 2π]
- ε1, ε2, ε3 ∈ [-ε_max, +ε_max], with ε_max e.g. 25° in radians

------------------------------------------------------------------------------
C. HYDRAULICS FORWARD MODEL (NO GEOMETRIC PROXIES)
------------------------------------------------------------------------------
Use steady Darcy flow in a homogeneous, isotropic reservoir of thickness H.
Compute pressure as a superposition of 2D logarithmic potentials of point sources/sinks.

Key choice (closure B):
- Per-well mass rates are fixed by equal allocation:
    ṁ_inj_each   = ṁ_total / 3
    ṁ_prod_each  = ṁ_total / 5
  Injection has positive volumetric rate, production negative.
  Convert mass rate to volumetric rate:
    Q_j [m^3/s] = sign_j * ṁ_j / ρ

Fluid properties (μ, ρ) MUST be computed using CoolProp at representative conditions.
Use mean conditions by default:
  P_mean = 0.5*(P_inj + P_prod)
  T_mean = 0.5*(T_inj + T_prod)
Convert temperatures to Kelvin for CoolProp.

Pressure potential at a point x = (x,y):
Let r_j(x) = distance from x to well j, with lower cutoff r_w = D/2 to avoid singularities.
Use a reference radius R_b (outer boundary / normalization radius) so logs are dimensionless.

Define the pressure field (up to an additive constant P_ref):
  p(x) = P_ref - F_two_sided * (μ / (2π κ H)) * Σ_j Q_j * ln( max(r_j(x), r_w) / R_b )

Where:
- μ: viscosity [Pa·s] from CoolProp
- ρ: density [kg/m^3] from CoolProp
- κ: permeability [m^2]
- H: thickness [m]
- D: well diameter [m], r_w = D/2
- R_b: boundary/reference radius [m] (user config; cancels out if ΣQ_j=0)
- F_two_sided: dimensionless factor, configurable to match the PDF convention.

Well bottom-hole pressure proxy:
- Evaluate p at each well coordinate using the same formula, with r_j = r_w for self-distance.
- Define Δp_i as pressure drop magnitude relative to P_ref:
    Δp_i := P_ref - p(x_i)
  This equals the positive sum term and is independent of P_ref when ΣQ_j = 0.
- Compute group-wise utilization factors:
    CV_inj  = std(Δp_injectors) / mean(Δp_injectors)
    CV_prod = std(Δp_producers) / mean(Δp_producers)

Pressure map:
- On a 2D grid covering ±(R_out + margin), compute p(x,y) and plot contours.

Velocity field (Darcy flux):
  q(x) = -(κ/μ) ∇p(x)
Compute ∇p analytically from log terms:
  ∂/∂x ln r = (x-xj)/r^2, ∂/∂y ln r = (y-yj)/r^2
so you can compute q(x) without finite differencing.

------------------------------------------------------------------------------
D. BREAKTHROUGH / PLUME UNIFORMITY (PHYSICS-BASED PROXY)
------------------------------------------------------------------------------
We need a "cold breakthrough uniformity" indicator that is NOT a pure distance proxy.

Implement streamlines and time-of-flight (TOF) proxy using the computed velocity field:
- Define pore velocity v_p(x) = q(x) / φ  (φ = porosity)
- For each injector, release N_seed starting points on a small circle around the well
  (radius r_seed ~ 3*r_w to avoid singularity).
- Numerically integrate particle trajectories using solve_ivp:
    dX/dt = v_p(X)
  Stop if X comes within r_capture of any producer (capture radius ~ few*r_w) or t exceeds t_max.
- Record travel times for trajectories reaching each producer.

Define a per-producer breakthrough proxy:
  t_bt,k = median TOF of trajectories that end at producer k
Compute:
  CV_tof = std(t_bt over 5 producers) / mean(t_bt over 5 producers)

Also plot:
- Streamlines (matplotlib streamplot) on top of pressure contours
- Seeds around injectors, capture circles around producers

------------------------------------------------------------------------------
E. THERMAL MODEL & LIFETIME τ (FACTOR2 SCREENING MODEL)
------------------------------------------------------------------------------
Use a conical frustum reservoir volume tied to geometry (R_in, R_out), NOT a cylinder.

Reservoir volume:
  V_res = (1/3) π H (R_out^2 + R_out R_in + R_in^2)

Mass of rock + residual water (residual water parameters remain configurable defaults):
  m_rock = V_res (1-φ) ρ_rock
  m_wat  = V_res φ ρ_wat S_wirr

Total available sensible heat:
  Q_res = (m_rock c_p,rock + m_wat c_p,wat) (T_res - T_working)

Heat extraction rate (use cp(Tprod-Tinj), as required):
  c_p,CO2 = CoolProp CP mass at (P_mean, T_mean) [J/kg/K]
  Q̇_CO2  = ṁ_total * c_p,CO2 * (T_prod - T_inj)
Ensure (T_prod - T_inj) > 0 by defaults; document that the user must keep it positive.

Lifetime:
  τ_years = Q_res / ( Q̇_CO2 * 3600 * 24 * 365 )

All constants must be in config.py and documented.

------------------------------------------------------------------------------
F. OBJECTIVE FUNCTION (MINIMIZE J)
------------------------------------------------------------------------------
Normalize mixed units. Use a reference lifetime τ_ref to scale to O(1).

Primary objective (closure B):
  J = w1*CV_inj + w2*CV_prod + w4*CV_tof - w3*(τ_years / τ_ref) + Penalty

Default weights in config.py:
- w1 = 1.0
- w2 = 1.0
- w3 = 1.0
- w4 = 0.5   (reasonable default: include TOF uniformity, but not overpower pressure CV)
- τ_ref = 30 years

Penalty terms (add large positive penalty):
- any spacing violation
- any angle increment violation
- any invalid geometry (R_out <= R_in, etc.)
Implement penalties smoothly (e.g., quadratic) for DE stability:
  Penalty = λ * Σ max(0, violation)^2

------------------------------------------------------------------------------
G. OPTIMIZATION
------------------------------------------------------------------------------
Use scipy.optimize.differential_evolution with:
- bounds as specified,
- a fixed random seed for reproducibility,
- reasonable popsize / maxiter defaults (configurable),
- optional parallel workers if available.

Return:
- best x*
- best metrics: CV_inj, CV_prod, CV_tof, τ_years
- derived well coordinates

------------------------------------------------------------------------------
H. REQUIRED PLOTS / OUTPUTS
------------------------------------------------------------------------------
1) Well layout plot (with labels P0, I1..I3, P1..P4)
2) Pressure map: contour plot of p(x,y) (or Δp(x,y) relative to P_ref)
3) Streamlines + seeds + capture circles (plume extension / flow connectivity)
4) Bar/marker plots:
   - Δp for each injector and producer
   - t_bt for each producer
5) Printed summary:
   - optimized variables
   - constraints status (min spacing etc.)
   - objective breakdown

------------------------------------------------------------------------------
I. CONFIG DEFAULTS (MUST MATCH THESE VALUES)
------------------------------------------------------------------------------
In config.py set the following defaults exactly:

Hydraulics:
- M_DOT_TOTAL = 100.0              % kg/s
- K_PERM      = 5e-14              % m^2
- H_THICK     = 300.0              % m
- D_WELL      = 0.41               % m
- R_WELL      = D_WELL/2
- P_INJ       = 200e5              % Pa (default)
- P_PROD      = 150e5              % Pa (default)
- T_INJ_C     = 40.0               % °C (default)
- T_PROD_C    = 100.0              % °C (default; must be > T_INJ_C)
- F_TWO_SIDED = 1.0                % configurable factor for PDF convention
- R_B         = 10000.0            % m (reference radius, cancels if ΣQ=0)

Thermal (given + defaults):
- POROSITY    = 0.10               % (-)
- RHO_ROCK    = 2300.0             % kg/m^3
- CP_ROCK     = 0.92e3             % J/kg/K   (0.92 kJ/kg/°C)
- RHO_WAT     = 1000.0             % kg/m^3 (default)
- CP_WAT      = 4180.0             % J/kg/K (default)
- S_WIRR      = 0.20               % (-) (default)
- T_RES_C     = 120.0              % °C (default)
- T_WORK_C    = 40.0               % °C (default)

Breakthrough / TOF:
- Use v_p = q/φ
- N_SEED per injector default = 40
- r_seed = 3*R_WELL
- r_capture = 3*R_WELL
- t_max = 5e9 seconds (default, adjustable)
- dt output controls for solve_ivp (default)

Geometry constraints (defaults):
- S_MIN = 500.0                    % m
- DELTA_R_MIN = 500.0              % m
- DELTA_THETA_MIN = 10° in radians % positive increment constraint
- EPS_MAX = 25° in radians

Optimization (defaults):
- differential_evolution popsize, maxiter reasonable so notebook runs in <2 minutes
- random_seed fixed

Now generate the full codebase + notebook with these defaults, end-to-end runnable.
