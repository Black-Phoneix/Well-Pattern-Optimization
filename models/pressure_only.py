"""
Pressure-only allocation model for CPG well-field optimization.

This module implements a steady-state, linear Darcy-style pressure/impedance model
for computing producer bottom-hole pressures and flow allocation in a system with:
- Fixed injector pressure (Dirichlet boundary condition)
- Fixed producer mass flow rate (Neumann boundary condition)

No thermal effects are considered. This is a pressure-only model.

Unit Conventions:
-----------------
- Pressure: Pa
- Mass flow rate: kg/s
- Distance: m
- Permeability: m^2
- Viscosity: Pa·s
- Density: kg/m^3
- Reservoir thickness: m
- Well radius: m
"""

from typing import Dict, Tuple, Union, List
import numpy as np

# Import existing modules for reuse
from patterns.geometry import Well, distance_matrix as geometry_distance_matrix
from well_model.production_well_dynamics import dry_CO2_model


def compute_distance_matrix_from_arrays(
    inj_xy: np.ndarray,
    prod_xy: np.ndarray
) -> np.ndarray:
    """
    Compute distance matrix from coordinate arrays.

    Parameters
    ----------
    inj_xy : np.ndarray
        Injector coordinates, shape (n_inj, 2) with columns [x, y] in meters.
    prod_xy : np.ndarray
        Producer coordinates, shape (n_prod, 2) with columns [x, y] in meters.

    Returns
    -------
    np.ndarray
        Distance matrix D of shape (n_prod, n_inj), where D[i, j] is the
        distance from producer i to injector j in meters.
    """
    inj_xy = np.asarray(inj_xy)
    prod_xy = np.asarray(prod_xy)

    # Ensure 2D arrays
    if inj_xy.ndim == 1:
        inj_xy = inj_xy.reshape(1, -1)
    if prod_xy.ndim == 1:
        prod_xy = prod_xy.reshape(1, -1)

    n_prod = prod_xy.shape[0]
    n_inj = inj_xy.shape[0]

    # Broadcast distance computation
    # prod_xy[:, np.newaxis, :] has shape (n_prod, 1, 2)
    # inj_xy[np.newaxis, :, :] has shape (1, n_inj, 2)
    # diff has shape (n_prod, n_inj, 2)
    diff = prod_xy[:, np.newaxis, :] - inj_xy[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))

    return D


def impedance_doublet(
    mu: float,
    rho: float,
    k: float,
    b: float,
    L: Union[float, np.ndarray],
    rw: float
) -> Union[float, np.ndarray]:
    """
    Analytical hydraulic impedance for an injector–producer pair.

    Computes the steady-state impedance based on the radial flow equation:
        Z(L) = (mu / (rho * k * b)) * (1 / (2*pi)) * ln(L / rw)

    This returns impedance in units of Pa per (kg/s).

    Parameters
    ----------
    mu : float
        Dynamic viscosity in Pa·s.
    rho : float
        Fluid density in kg/m³.
    k : float
        Permeability in m².
    b : float
        Reservoir thickness in m.
    L : float or np.ndarray
        Distance(s) between injector and producer in m.
    rw : float
        Well radius in m.

    Returns
    -------
    float or np.ndarray
        Impedance Z in Pa per (kg/s). Same shape as L.

    Notes
    -----
    - For distances L < rw, L is clamped to rw to avoid singularity.
    - The formula represents the pressure drop per unit mass flow rate
      for steady-state radial flow.
    """
    # Numerical protection against L -> 0
    L = np.maximum(L, rw)
    C1 = 1.0 / (2.0 * np.pi)
    return (mu / (rho * k * b)) * C1 * np.log(L / rw)


def compute_pairwise_impedance(
    inj_xy: np.ndarray,
    prod_xy: np.ndarray,
    params: dict
) -> np.ndarray:
    """
    Compute pairwise impedance matrix between injectors and producers.

    Parameters
    ----------
    inj_xy : np.ndarray
        Injector coordinates, shape (n_inj, 2) with columns [x, y] in meters.
    prod_xy : np.ndarray
        Producer coordinates, shape (n_prod, 2) with columns [x, y] in meters.
    params : dict
        Reservoir and fluid parameters. Required keys:
        - 'mu': Dynamic viscosity in Pa·s
        - 'rho': Fluid density in kg/m³
        - 'k': Permeability in m²
        - 'b': Reservoir thickness in m
        - 'rw': Well radius in m

    Returns
    -------
    np.ndarray
        Impedance matrix Z of shape (n_prod, n_inj), where Z[i, j] is the
        impedance between producer i and injector j in Pa per (kg/s).

    Raises
    ------
    KeyError
        If any required parameter is missing from params dict.

    Examples
    --------
    >>> inj_xy = np.array([[0, 0], [1000, 0]])  # 2 injectors
    >>> prod_xy = np.array([[500, 500]])        # 1 producer
    >>> params = {'mu': 1e-3, 'rho': 1000, 'k': 1e-13, 'b': 50, 'rw': 0.1}
    >>> Z = compute_pairwise_impedance(inj_xy, prod_xy, params)
    >>> Z.shape
    (1, 2)
    """
    # Extract parameters
    mu = params['mu']    # Viscosity [Pa·s]
    rho = params['rho']  # Density [kg/m³]
    k = params['k']      # Permeability [m²]
    b = params['b']      # Reservoir thickness [m]
    rw = params['rw']    # Well radius [m]

    # Compute distance matrix
    D = compute_distance_matrix_from_arrays(inj_xy, prod_xy)

    # Compute impedance using the doublet formula
    Z = impedance_doublet(mu, rho, k, b, D, rw)

    return Z


def solve_producer_bhp_equal_rate(
    P_inj: float,
    q_prod: float,
    Z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve for producer bottom-hole pressures under equal-rate constraint.

    Given:
    - All injectors have the same bottom-hole pressure P_inj (Dirichlet BC)
    - All producers have the same mass flow rate q_prod (Neumann BC)

    This function computes:
    1. Producer BHPs P_prod[i] for each producer
    2. Pairwise flow rates q_ij from injector j to producer i
    3. Injector mass flow rates q_inj[j]

    Mathematical formulation:
    -------------------------
    For flow from injector j to producer i:
        q_ij = (P_inj - P_prod[i]) / Z_ij

    Equal-rate constraint at each producer:
        sum_j q_ij = q_prod

    Solution:
        P_prod[i] = P_inj - q_prod / sum_j (1/Z_ij)
        q_ij = (P_inj - P_prod[i]) / Z_ij
        q_inj[j] = sum_i q_ij

    Parameters
    ----------
    P_inj : float
        Injector bottom-hole pressure in Pa. All injectors are assumed
        to operate at this same pressure.
    q_prod : float
        Production mass flow rate per producer in kg/s. This is the total
        flow into each producer (from all injectors combined).
    Z : np.ndarray
        Impedance matrix of shape (n_prod, n_inj), where Z[i, j] is the
        impedance between producer i and injector j in Pa per (kg/s).

    Returns
    -------
    P_prod : np.ndarray
        Producer bottom-hole pressures, shape (n_prod,) in Pa.
    q_ij : np.ndarray
        Pairwise flow rates, shape (n_prod, n_inj) in kg/s.
        q_ij[i, j] = flow from injector j to producer i.
    q_inj : np.ndarray
        Injector mass flow rates, shape (n_inj,) in kg/s.
        q_inj[j] = total flow from injector j (sum over all producers).

    Raises
    ------
    ValueError
        If Z contains invalid values (zero, negative, or infinite impedance).

    Examples
    --------
    >>> Z = np.array([[1e6, 2e6], [2e6, 1e6]])  # 2 producers, 2 injectors
    >>> P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(30e6, 10.0, Z)
    >>> P_prod.shape
    (2,)
    """
    Z = np.asarray(Z)
    n_prod, n_inj = Z.shape

    # Validate impedance matrix
    if np.any(Z <= 0):
        raise ValueError("Impedance matrix contains non-positive values")
    if np.any(~np.isfinite(Z)):
        raise ValueError("Impedance matrix contains non-finite values")

    # Compute admittance (1/Z) for each pair
    Y = 1.0 / Z  # Admittance matrix [kg/s per Pa]

    # Total admittance for each producer (sum over all injectors)
    S_vec = np.sum(Y, axis=1)  # Shape: (n_prod,)

    # Solve for producer pressures using the equal-rate constraint
    # P_prod[i] = P_inj - q_prod / S_vec[i]
    P_prod = P_inj - q_prod / S_vec

    # Compute pairwise flows
    # q_ij[i, j] = (P_inj - P_prod[i]) / Z[i, j]
    dP = P_inj - P_prod  # Pressure drop at each producer, shape (n_prod,)
    q_ij = dP[:, np.newaxis] / Z  # Broadcasting: (n_prod, 1) / (n_prod, n_inj)

    # Compute injector flow rates (sum over all producers)
    q_inj = np.sum(q_ij, axis=0)  # Shape: (n_inj,)

    return P_prod, q_ij, q_inj


def pressure_drop_variance(P_inj: float, P_prod: np.ndarray) -> float:
    """Return variance of producer pressure drop ``P_inj - P_prod`` in Pa²."""
    dP = P_inj - np.asarray(P_prod)
    return float(np.var(dP))


def producer_wellhead_pressures_from_bhp(
    P_prod_bhp: np.ndarray,
    q_prod_vec: np.ndarray,
    inlet_temperature: float = 343.15,
    depth: float = 3000.0,
    fluid: str = 'CO2',
    D: float = 0.27,
    L_segment: float = 100.0,
    roughness: float = 55e-6,
) -> np.ndarray:
    """Convert producer BHP values to producer wellhead pressures.

    Uses one default-parameter call to ``dry_CO2_model`` and applies the computed
    wellbore pressure loss as a shared offset (fast approximation).
    """
    P_prod_bhp = np.asarray(P_prod_bhp, dtype=float)
    q_prod_vec = np.asarray(q_prod_vec, dtype=float)

    if P_prod_bhp.shape != q_prod_vec.shape:
        raise ValueError('P_prod_bhp and q_prod_vec must have the same shape.')
    if np.any(q_prod_vec <= 0.0):
        raise ValueError('q_prod_vec must contain positive values.')

    q_ref = float(np.mean(q_prod_vec))
    p_ref = float(np.mean(P_prod_bhp))
    _, p_profile, _, _, _ = dry_CO2_model(
        mass_flow=q_ref,
        inlet_pressure=p_ref,
        inlet_temperature=inlet_temperature,
        depth=depth,
        fluid=fluid,
        D=D,
        L_segment=L_segment,
        roughness=roughness,
    )
    dp_wellbore = p_ref - float(p_profile[-1])
    return P_prod_bhp - dp_wellbore


def wellhead_pressure_variance(
    P_prod_bhp: np.ndarray,
    q_prod_vec: np.ndarray,
) -> float:
    """Return variance of producer wellhead pressures in Pa²."""
    p_wh = producer_wellhead_pressures_from_bhp(P_prod_bhp, q_prod_vec)
    return float(np.var(p_wh))


def _min_circular_separation_deg(angles_rad: np.ndarray) -> float:
    """Return minimum pairwise angular separation on a circle in degrees."""
    wrapped = np.mod(np.asarray(angles_rad), 2.0 * np.pi)
    wrapped.sort()
    gaps = np.diff(np.concatenate([wrapped, wrapped[:1] + 2.0 * np.pi]))
    return float(np.min(gaps) * 180.0 / np.pi)


def optimize_outer_producer_ring(
    inj_xy: np.ndarray,
    P_inj: float,
    q_prod: float,
    params: dict,
    R_inj: float,
    R_prod_bounds: Tuple[float, float],
    n_outer: int = 4,
    n_radius_samples: int = 60,
    n_angle_trials: int = 4000,
    min_angle_deg: float = 10.0,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Optimize outer producer coordinates for minimum pressure-drop variance.

    The optimization assumes:
    - 1 center producer fixed at (0, 0)
    - ``n_outer`` producers on one ring with shared radius ``R_prod``
    - each outer producer angle is independently optimized

    Constraints:
    - shared outer radius ``R_prod``
    - minimum circular angular separation >= ``min_angle_deg``

    Objective:
        minimize var(P_inj - P_prod[i])

    where ``P_prod`` is computed from superposition of all injector-to-producer
    conductances ``sum_j (1/Z_ij)`` under equal-rate constraints.
    """
    r_min, r_max = R_prod_bounds
    if r_min <= R_inj:
        raise ValueError('R_prod_bounds[0] must be greater than R_inj.')
    if r_max <= r_min:
        raise ValueError('R_prod_bounds must satisfy r_max > r_min.')
    if min_angle_deg <= 0.0:
        raise ValueError('min_angle_deg must be positive.')
    if n_outer * min_angle_deg >= 360.0:
        raise ValueError('min_angle_deg is too large for the number of outer producers.')

    radii = np.linspace(r_min, r_max, n_radius_samples)
    rng = np.random.default_rng(random_seed)

    # Candidate angle sets: include baseline equally spaced set + random independent sets
    angle_candidates = [np.arange(n_outer) * (2.0 * np.pi / n_outer)]
    max_attempts = max(50 * n_angle_trials, 2000)
    attempts = 0
    while len(angle_candidates) < n_angle_trials + 1 and attempts < max_attempts:
        attempts += 1
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=n_outer))
        if _min_circular_separation_deg(angles) >= min_angle_deg:
            angle_candidates.append(angles)

    if len(angle_candidates) < 2:
        raise ValueError('Failed to generate valid outer producer angles under min_angle_deg constraint.')

    best = None
    for R_prod in radii:
        for angles in angle_candidates:
            prod_xy = np.vstack([
                np.array([[0.0, 0.0]]),
                np.column_stack((R_prod * np.cos(angles), R_prod * np.sin(angles))),
            ])

            Z = compute_pairwise_impedance(inj_xy, prod_xy, params)
            P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(P_inj, q_prod, Z)
            q_prod_vec = np.full(P_prod.shape, q_prod, dtype=float)
            var_dp = wellhead_pressure_variance(P_prod, q_prod_vec)

            if best is None or var_dp < best['variance_dP']:
                best = {
                    'prod_xy': prod_xy,
                    'P_prod': P_prod,
                    'q_ij': q_ij,
                    'q_inj': q_inj,
                    'Z': Z,
                    'R_prod': np.array(R_prod),
                    'outer_angles_rad': np.array(angles),
                    'outer_angles_deg': np.array(angles * 180.0 / np.pi),
                    'min_angle_deg_achieved': np.array(_min_circular_separation_deg(angles)),
                    'variance_dP': np.array(var_dp),
                }

    return best


def solve_producer_bhp_variable_rate(
    P_inj: float,
    q_prod_vec: np.ndarray,
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve producer BHPs when each producer has its own target mass flow rate.

    Parameters
    ----------
    P_inj : float
        Shared injector BHP in Pa.
    q_prod_vec : np.ndarray
        Producer flow targets of shape (n_prod,) in kg/s.
    Z : np.ndarray
        Impedance matrix of shape (n_prod, n_inj) in Pa/(kg/s).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``P_prod`` (Pa), ``q_ij`` (kg/s), and ``q_inj`` (kg/s).
    """
    Z = np.asarray(Z, dtype=float)
    q_prod_vec = np.asarray(q_prod_vec, dtype=float)

    if Z.ndim != 2:
        raise ValueError('Z must be 2D [n_prod, n_inj].')
    n_prod, _ = Z.shape
    if q_prod_vec.shape != (n_prod,):
        raise ValueError('q_prod_vec shape must be (n_prod,).')
    if np.any(q_prod_vec <= 0):
        raise ValueError('q_prod_vec must contain positive values.')
    if np.any(Z <= 0) or np.any(~np.isfinite(Z)):
        raise ValueError('Impedance matrix contains invalid values.')

    Y = 1.0 / Z
    S_vec = np.sum(Y, axis=1)
    P_prod = P_inj - q_prod_vec / S_vec

    dP = P_inj - P_prod
    q_ij = dP[:, np.newaxis] / Z
    q_inj = np.sum(q_ij, axis=0)
    return P_prod, q_ij, q_inj


def _pressure_uniformity_ratio(P_prod: np.ndarray) -> float:
    """Relative producer-pressure spread: (max-min)/mean."""
    P_prod = np.asarray(P_prod, dtype=float)
    mean_p = float(np.mean(P_prod))
    if mean_p == 0.0:
        return np.inf
    return float((np.max(P_prod) - np.min(P_prod)) / abs(mean_p))


def optimize_producer_layout_priority(
    inj_xy: np.ndarray,
    P_inj: float,
    q_total: float,
    params: dict,
    R_inj: float,
    outer_radius_bounds: Tuple[float, float],
    center_radius_max: float = 150.0,
    min_outer_gap_deg: float = 20.0,
    lambda_r: float = 1.0,
    injector_flow_bounds: Tuple[float, float] | None = None,
    n_trials: int = 12000,
    pressure_tolerance_ratio: float = 0.05,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Priority optimizer: pressure consistency first, then layout uniformity.

    Decision variables (5 producers):
    - one "center" producer location (free within ``center_radius_max``)
    - four outer producer locations (shared outer radius + constrained angles)
    - producer flow split ``q_prod_vec`` with ``sum(q_prod_vec) = q_total``
      (which implicitly adjusts injector mass flows through coupling)

    Priority (lexicographic):
    1) Minimize producer pressure spread and satisfy ``pressure_tolerance_ratio``.
    2) Under (1), maximize spacing/uniformity:
       - larger min injector-to-producer distance (all producers)
       - larger min injector-to-outer-producer distance
       - larger min producer-to-producer distance
       - more even angular gaps among outer producers.
    """
    inj_xy = np.asarray(inj_xy, dtype=float)
    center_xy = np.mean(inj_xy, axis=0)

    r_min, r_max = outer_radius_bounds
    if r_min <= R_inj:
        raise ValueError('outer_radius_bounds[0] must be greater than R_inj.')
    if r_max <= r_min:
        raise ValueError('outer_radius_bounds must satisfy r_max > r_min.')
    if q_total <= 0.0:
        raise ValueError('q_total must be positive.')
    if center_radius_max < 0.0:
        raise ValueError('center_radius_max must be non-negative.')
    if min_outer_gap_deg <= 0.0:
        raise ValueError('min_outer_gap_deg must be positive.')
    if min_outer_gap_deg >= 90.0:
        raise ValueError('min_outer_gap_deg must be < 90 for 4 outer producers.')
    if injector_flow_bounds is not None:
        qinj_min, qinj_max = injector_flow_bounds
        if qinj_min < 0 or qinj_max <= qinj_min:
            raise ValueError('injector_flow_bounds must satisfy 0 <= min < max.')

    rng = np.random.default_rng(random_seed)
    best = None

    for _ in range(n_trials):
        # center producer (not fixed at injector-ring center)
        theta_c = rng.uniform(0.0, 2.0 * np.pi)
        r_c = rng.uniform(0.0, center_radius_max)
        p_center = center_xy + np.array([r_c * np.cos(theta_c), r_c * np.sin(theta_c)])

        # Outer producers on a shared radius. Angle gaps constrained to 90±10 deg.
        outer_r_scalar = rng.uniform(r_min, r_max)
        base_theta = rng.uniform(0.0, 2.0 * np.pi)
        # perturb each nominal 90deg gap, then normalize to 360deg
        gap_nominal = np.full(4, 90.0)
        gap_perturb = rng.uniform(-10.0, 10.0, size=4)
        gap_deg = gap_nominal + gap_perturb
        gap_deg *= 360.0 / np.sum(gap_deg)
        if np.min(gap_deg) < min_outer_gap_deg or np.max(gap_deg) > 100.0:
            continue

        outer_theta = np.radians(np.cumsum(np.concatenate([[base_theta * 180.0 / np.pi], gap_deg[:-1]])))
        outer_theta = np.mod(outer_theta, 2.0 * np.pi)
        outer_theta = np.sort(outer_theta)
        outer_r = np.full(4, outer_r_scalar)
        outer_xy = center_xy + np.column_stack((outer_r_scalar * np.cos(outer_theta), outer_r_scalar * np.sin(outer_theta)))

        prod_xy = np.vstack([p_center, outer_xy])

        # variable producer flow split (positive and summing to q_total)
        frac = rng.dirichlet(alpha=np.ones(5) * 3.0)
        q_prod_vec = q_total * frac

        Z = compute_pairwise_impedance(inj_xy, prod_xy, params)
        P_prod, q_ij, q_inj = solve_producer_bhp_variable_rate(P_inj, q_prod_vec, Z)

        if injector_flow_bounds is not None:
            qinj_min, qinj_max = injector_flow_bounds
            if np.any(q_inj < qinj_min) or np.any(q_inj > qinj_max):
                continue

        pressure_ratio = _pressure_uniformity_ratio(
            producer_wellhead_pressures_from_bhp(P_prod, q_prod_vec)
        )
        pressure_violation = max(0.0, pressure_ratio - pressure_tolerance_ratio)

        # uniformity terms
        d_ip_all = np.linalg.norm(prod_xy[:, None, :] - inj_xy[None, :, :], axis=2)
        d_ip_outer = np.linalg.norm(outer_xy[:, None, :] - inj_xy[None, :, :], axis=2)
        min_ip_all = float(np.min(d_ip_all))
        min_ip_outer = float(np.min(d_ip_outer))

        d_pp = np.linalg.norm(prod_xy[:, None, :] - prod_xy[None, :, :], axis=2)
        d_pp = d_pp + np.eye(5) * 1e9
        min_pp = float(np.min(d_pp))

        gaps = np.diff(np.concatenate([outer_theta, outer_theta[:1] + 2.0 * np.pi]))
        gap_cv = float(np.std(gaps) / np.mean(gaps))
        std_outer_r = float(np.std(outer_r))

        # second-stage utility (higher is better)
        utility = 1.0 * min_ip_all + 0.7 * min_pp - 120.0 * gap_cv - lambda_r * std_outer_r

        candidate = {
            'prod_xy': prod_xy,
            'q_prod_vec': q_prod_vec,
            'P_prod': P_prod,
            'q_ij': q_ij,
            'q_inj': q_inj,
            'Z': Z,
            'pressure_uniformity_ratio': np.array(pressure_ratio),
            'pressure_tolerance_ratio': np.array(pressure_tolerance_ratio),
            'min_ip_all': np.array(min_ip_all),
            'min_ip_outer': np.array(min_ip_outer),
            'min_pp': np.array(min_pp),
            'gap_cv_outer': np.array(gap_cv),
            'std_outer_r': np.array(std_outer_r),
            'utility': np.array(utility),
            'outer_theta_deg': np.array(outer_theta * 180.0 / np.pi),
            'outer_r': np.array(outer_r),
            'center_xy': np.array(p_center),
            'injector_flow_bounds': None if injector_flow_bounds is None else np.array(injector_flow_bounds),
        }

        if best is None:
            best = candidate
            continue

        # layered comparison: pressure first, then spacing/uniformity hierarchy
        best_violation = max(0.0, float(best['pressure_uniformity_ratio']) - pressure_tolerance_ratio)
        if pressure_violation < best_violation - 1e-12:
            best = candidate
            continue
        if pressure_violation > best_violation + 1e-12:
            continue

        if pressure_ratio < float(best['pressure_uniformity_ratio']) - 1e-12:
            best = candidate
            continue
        if pressure_ratio > float(best['pressure_uniformity_ratio']) + 1e-12:
            continue

        if min_ip_all > float(best['min_ip_all']) + 1e-12:
            best = candidate
            continue
        if min_ip_all < float(best['min_ip_all']) - 1e-12:
            continue

        if min_ip_outer > float(best['min_ip_outer']) + 1e-12:
            best = candidate
            continue
        if min_ip_outer < float(best['min_ip_outer']) - 1e-12:
            continue

        if min_pp > float(best['min_pp']) + 1e-12:
            best = candidate
            continue
        if min_pp < float(best['min_pp']) - 1e-12:
            continue

        if gap_cv < float(best['gap_cv_outer']) - 1e-12:
            best = candidate
            continue
        if gap_cv > float(best['gap_cv_outer']) + 1e-12:
            continue

        if std_outer_r < float(best['std_outer_r']) - 1e-12:
            best = candidate
            continue

        if utility > float(best['utility']):
            best = candidate

    return best


def validate_solution(
    q_ij: np.ndarray,
    q_prod: float,
    tol: float = 1e-6
) -> None:
    """
    Validate the computed flow solution against physical constraints.

    This function checks:
    1. Mass balance at each producer: sum_j q_ij ≈ q_prod
    2. All pairwise flows are non-negative

    Parameters
    ----------
    q_ij : np.ndarray
        Pairwise flow rates, shape (n_prod, n_inj) in kg/s.
    q_prod : float
        Expected production rate per producer in kg/s.
    tol : float, optional
        Relative tolerance for mass balance check. Default is 1e-6.

    Raises
    ------
    ValueError
        If mass balance constraint is violated for any producer.
    ValueError
        If any pairwise flow is negative.

    Examples
    --------
    >>> q_ij = np.array([[5.0, 5.0], [5.0, 5.0]])  # 2 producers, 2 injectors
    >>> validate_solution(q_ij, q_prod=10.0)  # Should pass silently
    """
    q_ij = np.asarray(q_ij)
    n_prod, n_inj = q_ij.shape

    # Check for negative flows
    if np.any(q_ij < 0):
        neg_indices = np.argwhere(q_ij < 0)
        raise ValueError(
            f"Negative pairwise flows detected at indices: {neg_indices.tolist()}. "
            f"This indicates P_prod > P_inj for some producers, which is physically invalid."
        )

    # Check mass balance at each producer
    q_prod_computed = np.sum(q_ij, axis=1)
    relative_error = np.abs(q_prod_computed - q_prod) / np.abs(q_prod)

    for i, err in enumerate(relative_error):
        if err > tol:
            raise ValueError(
                f"Mass balance violated at producer {i}: "
                f"computed flow = {q_prod_computed[i]:.6f} kg/s, "
                f"expected = {q_prod:.6f} kg/s, "
                f"relative error = {err:.2e} > tolerance {tol:.2e}"
            )


def validate_solution_variable_rate(
    q_ij: np.ndarray,
    q_prod_vec: np.ndarray,
    tol: float = 1e-6,
) -> None:
    """Validate producer balances for variable per-producer target rates."""
    q_ij = np.asarray(q_ij, dtype=float)
    q_prod_vec = np.asarray(q_prod_vec, dtype=float)

    if q_ij.ndim != 2:
        raise ValueError('q_ij must be 2D [n_prod, n_inj].')
    n_prod, _ = q_ij.shape
    if q_prod_vec.shape != (n_prod,):
        raise ValueError('q_prod_vec shape must be (n_prod,).')

    if np.any(q_ij < 0):
        neg_indices = np.argwhere(q_ij < 0)
        raise ValueError(f'Negative pairwise flows detected at indices: {neg_indices.tolist()}')

    q_prod_computed = np.sum(q_ij, axis=1)
    denom = np.maximum(np.abs(q_prod_vec), 1e-30)
    relative_error = np.abs(q_prod_computed - q_prod_vec) / denom
    for i, err in enumerate(relative_error):
        if err > tol:
            raise ValueError(
                f"Mass balance violated at producer {i}: computed flow = {q_prod_computed[i]:.6f} kg/s, "
                f"expected = {q_prod_vec[i]:.6f} kg/s, relative error = {err:.2e} > tolerance {tol:.2e}"
            )


def validate_total_mass_balance(
    q_inj: np.ndarray,
    q_prod: float,
    n_prod: int,
    tol: float = 1e-6
) -> None:
    """
    Validate total mass balance: sum of injection equals sum of production.

    Parameters
    ----------
    q_inj : np.ndarray
        Injector mass flow rates, shape (n_inj,) in kg/s.
    q_prod : float
        Production rate per producer in kg/s.
    n_prod : int
        Number of producers.
    tol : float, optional
        Relative tolerance. Default is 1e-6.

    Raises
    ------
    ValueError
        If total injection does not match total production within tolerance.
    """
    total_injection = np.sum(q_inj)
    total_production = q_prod * n_prod

    relative_error = np.abs(total_injection - total_production) / np.abs(total_production)

    if relative_error > tol:
        raise ValueError(
            f"Total mass balance violated: "
            f"total injection = {total_injection:.6f} kg/s, "
            f"total production = {total_production:.6f} kg/s, "
            f"relative error = {relative_error:.2e} > tolerance {tol:.2e}"
        )


def wells_to_coordinate_arrays(
    wells: List[Well]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of Well objects to separate injector and producer coordinate arrays.

    Parameters
    ----------
    wells : List[Well]
        List of Well objects with 'x', 'y', and 'kind' attributes.

    Returns
    -------
    inj_xy : np.ndarray
        Injector coordinates, shape (n_inj, 2).
    prod_xy : np.ndarray
        Producer coordinates, shape (n_prod, 2).
    """
    injectors = [(w.x, w.y) for w in wells if w.kind == 'injector']
    producers = [(w.x, w.y) for w in wells if w.kind == 'producer']

    inj_xy = np.array(injectors) if injectors else np.empty((0, 2))
    prod_xy = np.array(producers) if producers else np.empty((0, 2))

    return inj_xy, prod_xy


def solve_pressure_allocation(
    injectors: List[Well],
    producers: List[Well],
    P_inj: float,
    q_prod: float,
    params: dict,
    validate: bool = True
) -> dict:
    """
    High-level API to solve the pressure-only allocation problem.

    This is a convenience function that combines impedance computation,
    pressure solution, and validation into a single call.

    Parameters
    ----------
    injectors : List[Well]
        List of injector Well objects.
    producers : List[Well]
        List of producer Well objects.
    P_inj : float
        Injector bottom-hole pressure in Pa.
    q_prod : float
        Production mass flow rate per producer in kg/s.
    params : dict
        Reservoir and fluid parameters. Required keys:
        - 'mu': Dynamic viscosity in Pa·s
        - 'rho': Fluid density in kg/m³
        - 'k': Permeability in m²
        - 'b': Reservoir thickness in m
        - 'rw': Well radius in m
    validate : bool, optional
        If True, validate the solution. Default is True.

    Returns
    -------
    dict
        Solution dictionary with keys:
        - 'P_prod': Producer bottom-hole pressures (n_prod,) [Pa]
        - 'q_ij': Pairwise flow rates (n_prod, n_inj) [kg/s]
        - 'q_inj': Injector flow rates (n_inj,) [kg/s]
        - 'Z': Impedance matrix (n_prod, n_inj) [Pa/(kg/s)]
        - 'injectors': List of injector Well objects
        - 'producers': List of producer Well objects

    Examples
    --------
    >>> from patterns.geometry import generate_center_ring_pattern
    >>> injectors, producers = generate_center_ring_pattern(3, 4, 600.0, 300.0)
    >>> params = {'mu': 5e-5, 'rho': 800, 'k': 5e-14, 'b': 300, 'rw': 0.1}
    >>> result = solve_pressure_allocation(injectors, producers, 30e6, 10.0, params)
    """
    # Convert wells to coordinate arrays
    inj_xy = np.array([[w.x, w.y] for w in injectors])
    prod_xy = np.array([[w.x, w.y] for w in producers])

    # Compute impedance matrix
    Z = compute_pairwise_impedance(inj_xy, prod_xy, params)

    # Solve for pressures and flows
    P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(P_inj, q_prod, Z)

    # Validate if requested
    if validate:
        validate_solution(q_ij, q_prod)
        validate_total_mass_balance(q_inj, q_prod, len(producers))

    return {
        'P_prod': P_prod,
        'q_ij': q_ij,
        'q_inj': q_inj,
        'Z': Z,
        'injectors': injectors,
        'producers': producers,
    }
