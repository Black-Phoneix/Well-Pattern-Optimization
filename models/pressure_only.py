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

from typing import Tuple, Union, List
import numpy as np

# Import existing modules for reuse
from patterns.geometry import Well, distance_matrix as geometry_distance_matrix


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
