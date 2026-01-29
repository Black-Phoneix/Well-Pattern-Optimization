import importlib
import numpy as np

# Geometry utilities from patterns.geometry:
# - generate_center_ring_pattern: create injector/producer coordinates for a center + ring layout
# - distance_matrix: pairwise distances between two well sets
# - minimum_spacing: minimum inter-well distance for constraint checking
from patterns.geometry import generate_center_ring_pattern, distance_matrix, minimum_spacing

# Import the impedance model from models module
from models.impedance import impedance_doublet


# =========================
# 1) Physical / reservoir constants (homogeneous reservoir)
# =========================
MU = 5e-5          # Fluid dynamic viscosity [Pa·s]
RHO_FLUID = 600.0  # Fluid density [kg/m^3]
CP_FLUID = 2000.0  # Fluid specific heat capacity [J/(kg·K)]
K_PERM = 50e-15    # Permeability [m^2]
H_THICK = 50.0     # Reservoir thickness [m]
RW = 0.15          # Wellbore radius [m]
POROSITY = 0.20    # Porosity [-]
RHO_ROCK = 2600.0  # Rock density [kg/m^3]
CP_ROCK = 1000.0   # Rock specific heat capacity [J/(kg·K)]


# =========================
# 2) Operating conditions and domain size
# =========================
P_INJ = 200e5          # Injection pressure [Pa]
T_INJ = 40.0           # Injection temperature [°C]
T_RES_INITIAL = 100.0  # Initial reservoir temperature [°C]
OPERATION_YEARS = 30   # Planned operating time [years]
R_BOUNDARY = 1000.0    # Outer boundary radius (lease / model domain) [m]


# =========================
# 3) Well pattern definition (center + rings)
# =========================
N_INJ = 3               # Number of injectors on injector ring
N_PROD_OUTER = 4        # Number of producers on outer ring
CENTER_PRODUCER = True  # If True: add 1 producer at the center (total producers = 1 + N_PROD_OUTER)


# =========================
# 4) Optimization constraints and scaling targets
# =========================
MIN_WELL_SPACING = 100.0  # Minimum allowable distance between any two wells [m]
M_TOTAL_TARGET = 200.0    # Target total production mass flow rate (shared across producers) [kg/s]
CUTOFF_FRACTION = 0.1     # Thermal cutoff fraction for lifetime definition (dimensionless)


# =========================
# 5) Objective weights and penalty
# =========================
WEIGHT_PRESSURE = 1.0   # Penalize producer pressure non-uniformity (proxy for imbalance)
WEIGHT_LIFE_CV = 2.0    # Penalize coefficient of variation of lifetimes across producers
WEIGHT_PUMP = 0.0       # Optional: penalize mean pressure drop (pumping effort); set to 0.0 to disable
PENALTY_LARGE = 1e9     # Large penalty for infeasible patterns (constraints / numerical failures)


# =========================
# 6) Thermal model import with robust fallback
# =========================
def _import_thermal_function():
    """
    Returns a function CALC_THERMAL(m_dot, V_bulk, ...) -> (P_avg, lifetime_years)

    Preferred: models.thermal.calculate_thermal_power_and_lifetime
    Fallback: lumped-capacity / equivalent-volume model (no conduction),
              producing a hyperbolic temperature decline and cutoff-based lifetime.
    """
    try:
        mod = importlib.import_module("models.thermal")
        return mod.calculate_thermal_power_and_lifetime
    except Exception:
        # Fallback: fast analytical approximation
        def _fallback(
            m_dot, V_bulk, T_inj, T_res_initial, cp_fluid, rho_fluid, rho_rock, cp_rock, porosity,
            operation_time_years=30, cutoff_fraction=0.1
        ):
            # Convert operation horizon to seconds
            t_end_seconds = operation_time_years * 365.0 * 24.0 * 3600.0

            # Maximum temperature difference available for heat extraction
            dT_max = T_res_initial - T_inj

            # Effective volumetric heat capacity of the rock-fluid mixture
            rhoc_eff = porosity * rho_fluid * cp_fluid + (1.0 - porosity) * rho_rock * cp_rock

            # Guard against degenerate flows/volumes
            if m_dot <= 1e-6 or V_bulk <= 0.0:
                return 0.0, 1e9

            # Characteristic time constant for heat depletion (advective, lumped model)
            tau = (V_bulk * rhoc_eff) / (m_dot * cp_fluid)

            # Time-average of G(t) = (1 + t/tau)^(-1) over [0, t_end]
            integral_G = tau * np.log(t_end_seconds / tau + 1.0)

            # Average thermal power over the operation horizon
            P_avg = (m_dot * cp_fluid * dT_max) * (integral_G / t_end_seconds)

            # Lifetime defined as time when G(t) drops to cutoff_fraction
            cutoff = float(cutoff_fraction)
            cutoff = np.clip(cutoff, 1e-6, 0.999999)
            lifetime_seconds = tau * (1.0 / cutoff - 1.0)
            lifetime_years = lifetime_seconds / (365.0 * 24.0 * 3600.0)
            return P_avg, lifetime_years

        return _fallback


# Thermal evaluation function handle (either imported or fallback)
CALC_THERMAL = _import_thermal_function()


# =========================
# 7) Hydraulics: admittance aggregation and equal-flow production pressures
# =========================
def _admittance_sum_per_producer(injectors, producers, mu, rho, k, b, rw):
    """
    Compute total hydraulic admittance S_j for each producer j:
      Z_ij = impedance between injector i and producer j (doublet formula)
      Y_ij = 1 / Z_ij
      S_j  = sum_i Y_ij  (superposition of injector contributions)
    """
    D = distance_matrix(injectors, producers)     # Pairwise distances [m]
    Z = impedance_doublet(mu, rho, k, b, D, rw)   # Pairwise impedances
    Y = 1.0 / Z                                  # Admittance matrix
    return np.sum(Y, axis=1)                      # Sum over injectors -> per producer


def producer_pressures_equal_flow(injectors, producers, P_inj, m_total, mu, rho, k, b, rw, thermo_dP=0.0):
    """
    Equal-flow producer model:
    - Assume each producer takes the same mass flow: m_each = m_total / n_prod
    - Injection pressure is fixed at P_inj
    - Producer pressures adjust based on local admittance S:
        P_prod = P_inj + thermo_dP - (m_each / S)
      (thermo_dP can represent an imposed thermosiphon / buoyancy contribution)
    """
    S = _admittance_sum_per_producer(injectors, producers, mu, rho, k, b, rw)
    n_prod = len(producers)

    m_each = float(m_total) / float(n_prod)
    flow_vec = np.full(n_prod, m_each, dtype=float)

    # Numerical floor to avoid division by zero if S is extremely small
    eps = 1e-30
    P_prod_vec = float(P_inj) + float(thermo_dP) - flow_vec / np.maximum(S, eps)

    return P_prod_vec, flow_vec, S


# =========================
# 8) Penalty metric: producer pressure uniformity (normalized)
# =========================
def _pressure_uniformity(P_prod_vec, mode="variance"):
    """
    Quantify how 'balanced' producer pressures are.
    - variance: normalized variance of P_prod across producers
    - range: normalized max-min spread of P_prod across producers
    Normalization uses a scale ~ mean pressure to make the metric dimensionless.
    """
    mean_p = float(np.mean(P_prod_vec))
    scale = max(abs(mean_p), 1.0)

    if mode == "variance":
        return float(np.var(P_prod_vec) / (scale * scale))
    if mode == "range":
        return float((np.max(P_prod_vec) - np.min(P_prod_vec)) / scale)

    raise ValueError("mode must be 'variance' or 'range'")


# =========================
# 9) Thermal volume allocation (geometric proxy)
# =========================
def _volume_allocation_bulk(R_inj, R_boundary, H, n_prod, center_producer=True):
    """
    Allocate an effective bulk reservoir volume to each producer for the thermal model.

    Geometric approximation:
    - Total volume: V_total = pi * R_boundary^2 * H
    - If a center producer exists:
        * Center producer gets V_center = pi * R_inj^2 * H
        * Outer producers share the remaining annulus volume equally
    - If no center producer:
        * All producers share V_total equally

    Returned volumes are scaled so sum(V_bulk_each) == V_total (numerical consistency).
    """
    V_total = np.pi * (R_boundary ** 2) * H

    if center_producer:
        n_outer = n_prod - 1
        V_center = np.pi * (R_inj ** 2) * H
        V_outer_total = max(0.0, np.pi * (R_boundary ** 2 - R_inj ** 2) * H)
        V_outer = (V_outer_total / n_outer) if n_outer > 0 else 0.0
        V_geo = np.array([V_center] + [V_outer] * n_outer, dtype=float)
    else:
        V_geo = np.full(n_prod, V_total / n_prod, dtype=float)

    # Rescale so that total allocated volume exactly matches V_total
    s = float(np.sum(V_geo))
    if s <= 1e-12:
        return None
    return V_geo * (V_total / s)


# =========================
# 10) Main objective function for optimization
# =========================
def objective_function(
    params,
    pressure_mode="variance",
    thermo_dP=0.0,
    dmin=MIN_WELL_SPACING,
    m_total_target=M_TOTAL_TARGET
):
    """
    Decision variables:
      params = [R_inj, R_prod]
        R_inj: injector ring radius [m]
        R_prod: producer ring radius [m]

    Objective design intent:
    - Maximize normalized average thermal power (negative sign in cost)
    - Minimize lifetime imbalance across producers (CV of lifetimes)
    - Minimize producer pressure non-uniformity (proxy for balanced utilization)
    - Enforce feasibility via large penalties (bounds + minimum spacing + finite outputs)
    """
    R_inj, R_prod = float(params[0]), float(params[1])

    # Basic geometric feasibility: rings must be ordered and within the domain
    if not (0.0 < R_inj < R_prod < R_BOUNDARY):
        return PENALTY_LARGE

    # Generate the well coordinates for the chosen parameterization
    inj_wells, prod_wells = generate_center_ring_pattern(
        n_inj=N_INJ,
        n_prod_outer=N_PROD_OUTER,
        R_inj=R_inj,
        R_prod=R_prod,
        center_producer=CENTER_PRODUCER,
    )

    # Minimum inter-well spacing constraint (avoid unrealistic clustering / short-circuit risk)
    all_wells = inj_wells + prod_wells
    if dmin > 0.0 and minimum_spacing(all_wells) < dmin:
        return PENALTY_LARGE

    # Hydraulics: compute producer pressures given equal split of total mass flow
    P_prod_vec, flows, S = producer_pressures_equal_flow(
        inj_wells,
        prod_wells,
        P_INJ,
        m_total_target,
        MU,
        RHO_FLUID,
        K_PERM,
        H_THICK,
        RW,
        thermo_dP=thermo_dP,
    )

    # Numerical robustness check
    if (not np.all(np.isfinite(P_prod_vec))) or (not np.all(np.isfinite(flows))) or (not np.all(np.isfinite(S))):
        return PENALTY_LARGE

    # Thermal volumes (geometric allocation) per producer
    V_bulk_each = _volume_allocation_bulk(
        R_inj, R_BOUNDARY, H_THICK, len(prod_wells),
        center_producer=CENTER_PRODUCER
    )
    if V_bulk_each is None:
        return PENALTY_LARGE

    # Thermal evaluation: compute per-producer average power and lifetime
    total_power_mw = 0.0
    lifetimes = []

    for i in range(len(prod_wells)):
        p_avg, t_life = CALC_THERMAL(
            float(flows[i]),
            float(V_bulk_each[i]),
            T_INJ,
            T_RES_INITIAL,
            CP_FLUID,
            RHO_FLUID,
            RHO_ROCK,
            CP_ROCK,
            POROSITY,
            operation_time_years=OPERATION_YEARS,
            cutoff_fraction=CUTOFF_FRACTION,
        )
        if (not np.isfinite(p_avg)) or (not np.isfinite(t_life)):
            return PENALTY_LARGE

        # Convert W -> MW and accumulate total
        total_power_mw += float(p_avg) / 1e6
        lifetimes.append(float(t_life))

    # Reference power for normalization: ideal initial thermal power at full delta-T (no depletion)
    P_ref_mw = (float(m_total_target) * CP_FLUID * (T_RES_INITIAL - T_INJ)) / 1e6
    score_power = total_power_mw / max(P_ref_mw, 1e-12)  # dimensionless power score

    # Lifetime balance metric: coefficient of variation (std/mean) across producers
    avg_life = float(np.mean(lifetimes))
    cv_life = float(np.std(lifetimes) / max(avg_life, 1e-12))

    # Pressure uniformity metric (dimensionless)
    pressure_penalty = _pressure_uniformity(P_prod_vec, mode=pressure_mode)

    # Scalar cost (minimize):
    # - maximize power  -> subtract score_power
    # - penalize lifetime imbalance
    # - penalize pressure imbalance
    cost = -score_power
    cost += WEIGHT_LIFE_CV * cv_life
    cost += WEIGHT_PRESSURE * pressure_penalty

    # Optional pumping effort penalty based on mean pressure drop (only if enabled)
    if WEIGHT_PUMP != 0.0:
        dP_mean = float(P_INJ - np.mean(P_prod_vec))
        cost += WEIGHT_PUMP * max(0.0, dP_mean) / max(P_INJ, 1.0)

    return float(cost)
