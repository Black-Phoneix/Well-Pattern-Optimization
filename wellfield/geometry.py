"""
Well coordinates, constraints, and distance matrix.

This module implements the geometry for the well-field layout:
- Producer P0 at center: (0, 0)
- Injectors I1..I3 on inner ring radius R_in at angles: 0, 2π/3, 4π/3
- Outer producers P1..P4 on outer ring radius R_out with variable angles

Optimization variables:
- R_in [m]: Inner ring radius
- R_out [m]: Outer ring radius (constraint: R_out >= R_in + ΔR_min)
- θ0 [rad]: Global rotation for outer ring
- ε1, ε2, ε3 [rad]: Deviations from perfect 90° increments

Outer producer angles:
    θP1 = θ0
    θP2 = θ0 + (π/2 + ε1)
    θP3 = θP2 + (π/2 + ε2)
    θP4 = θP3 + (π/2 + ε3)
with closure: ε4 = -(ε1 + ε2 + ε3) so that total sum of increments = 2π
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import Config, DEFAULT_CONFIG


def compute_well_coordinates(
    R_in: float,
    R_out: float,
    theta0: float,
    eps1: float,
    eps2: float,
    eps3: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute well coordinates from optimization variables.
    
    Layout:
    - P0: Center producer at (0, 0)
    - I1, I2, I3: Injectors on inner ring at R_in with fixed angles (0, 2π/3, 4π/3)
    - P1, P2, P3, P4: Outer producers on outer ring at R_out with variable angles
    
    Parameters
    ----------
    R_in : float
        Inner ring radius [m] for injectors
    R_out : float
        Outer ring radius [m] for producers
    theta0 : float
        Global rotation angle [rad] for outer ring
    eps1, eps2, eps3 : float
        Deviations from π/2 increments [rad] for outer producers
    
    Returns
    -------
    dict
        Dictionary with well names as keys and (x, y) coordinates as values.
        Keys: 'P0', 'I1', 'I2', 'I3', 'P1', 'P2', 'P3', 'P4'
    """
    coords = {}
    
    # Center producer P0
    coords['P0'] = (0.0, 0.0)
    
    # Injectors I1, I2, I3 on inner ring at fixed angles
    # Angles: 0, 2π/3, 4π/3
    inj_angles = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
    for i, angle in enumerate(inj_angles, start=1):
        x = R_in * np.cos(angle)
        y = R_in * np.sin(angle)
        coords[f'I{i}'] = (x, y)
    
    # Outer producers P1, P2, P3, P4 on outer ring with variable angles
    # θP1 = θ0
    # θP2 = θ0 + (π/2 + ε1)
    # θP3 = θP2 + (π/2 + ε2)
    # θP4 = θP3 + (π/2 + ε3)
    theta_P1 = theta0
    theta_P2 = theta_P1 + (np.pi / 2.0 + eps1)
    theta_P3 = theta_P2 + (np.pi / 2.0 + eps2)
    theta_P4 = theta_P3 + (np.pi / 2.0 + eps3)
    
    prod_angles = [theta_P1, theta_P2, theta_P3, theta_P4]
    for i, angle in enumerate(prod_angles, start=1):
        x = R_out * np.cos(angle)
        y = R_out * np.sin(angle)
        coords[f'P{i}'] = (x, y)
    
    return coords


def compute_all_well_positions(
    R_in: float,
    R_out: float,
    theta0: float,
    eps1: float,
    eps2: float,
    eps3: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute arrays of well positions separated by type.
    
    Parameters
    ----------
    R_in, R_out, theta0, eps1, eps2, eps3 : float
        Optimization variables (see compute_well_coordinates)
    
    Returns
    -------
    tuple of np.ndarray
        (all_positions, injector_positions, producer_positions)
        - all_positions: (8, 2) array with all wells
        - injector_positions: (3, 2) array with injectors I1, I2, I3
        - producer_positions: (5, 2) array with producers P0, P1, P2, P3, P4
    """
    coords = compute_well_coordinates(R_in, R_out, theta0, eps1, eps2, eps3)
    
    # All positions (order: P0, I1, I2, I3, P1, P2, P3, P4)
    all_pos = np.array([
        coords['P0'], coords['I1'], coords['I2'], coords['I3'],
        coords['P1'], coords['P2'], coords['P3'], coords['P4']
    ])
    
    # Injector positions (I1, I2, I3)
    inj_pos = np.array([coords['I1'], coords['I2'], coords['I3']])
    
    # Producer positions (P0, P1, P2, P3, P4)
    prod_pos = np.array([
        coords['P0'], coords['P1'], coords['P2'], coords['P3'], coords['P4']
    ])
    
    return all_pos, inj_pos, prod_pos


def compute_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix for all wells.
    
    Parameters
    ----------
    positions : np.ndarray
        Well positions as (N, 2) array
    
    Returns
    -------
    np.ndarray
        (N, N) symmetric distance matrix where D[i, j] = distance between well i and j
    """
    N = len(positions)
    D = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            D[i, j] = dist
            D[j, i] = dist
    
    return D


def compute_minimum_spacing(positions: np.ndarray) -> float:
    """
    Compute minimum pairwise distance between all wells.
    
    Parameters
    ----------
    positions : np.ndarray
        Well positions as (N, 2) array
    
    Returns
    -------
    float
        Minimum inter-well distance [m]
    """
    D = compute_distance_matrix(positions)
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(D, np.inf)
    return float(np.min(D))


def check_geometry_constraints(
    R_in: float,
    R_out: float,
    eps1: float,
    eps2: float,
    eps3: float,
    config: Optional[Config] = None,
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Check all geometric constraints.
    
    Constraints:
    1. R_out - R_in >= DELTA_R_MIN
    2. Minimum well spacing >= S_MIN
    3. All angle increments (π/2 + εk) >= DELTA_THETA_MIN for k=1..4
       where ε4 = -(ε1 + ε2 + ε3)
    
    Parameters
    ----------
    R_in, R_out : float
        Ring radii [m]
    eps1, eps2, eps3 : float
        Angle deviations [rad]
    config : Config, optional
        Configuration object (default: DEFAULT_CONFIG)
    
    Returns
    -------
    tuple
        (is_valid, violations, metrics)
        - is_valid: bool, True if all constraints satisfied
        - violations: list of str, descriptions of violated constraints
        - metrics: dict, computed constraint metrics
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    violations = []
    metrics = {}
    
    # Compute ε4 for closure
    eps4 = -(eps1 + eps2 + eps3)
    
    # Constraint 1: Radial gap
    radial_gap = R_out - R_in
    metrics['radial_gap'] = radial_gap
    if radial_gap < config.DELTA_R_MIN:
        violations.append(
            f"Radial gap violation: {radial_gap:.1f} m < {config.DELTA_R_MIN:.1f} m"
        )
    
    # Constraint 2: Minimum well spacing
    theta0 = 0.0  # Use theta0=0 for constraint check (rotation doesn't affect spacing)
    all_pos, _, _ = compute_all_well_positions(R_in, R_out, theta0, eps1, eps2, eps3)
    min_spacing = compute_minimum_spacing(all_pos)
    metrics['min_spacing'] = min_spacing
    if min_spacing < config.S_MIN:
        violations.append(
            f"Minimum spacing violation: {min_spacing:.1f} m < {config.S_MIN:.1f} m"
        )
    
    # Constraint 3: Angle increments must be positive and >= DELTA_THETA_MIN
    increments = [
        np.pi / 2.0 + eps1,
        np.pi / 2.0 + eps2,
        np.pi / 2.0 + eps3,
        np.pi / 2.0 + eps4,
    ]
    for k, inc in enumerate(increments, start=1):
        metrics[f'angle_increment_{k}'] = inc
        if inc < config.DELTA_THETA_MIN:
            violations.append(
                f"Angle increment {k} violation: {np.rad2deg(inc):.1f}° < {config.DELTA_THETA_MIN_DEG:.1f}°"
            )
    
    is_valid = len(violations) == 0
    return is_valid, violations, metrics


def compute_constraint_penalties(
    R_in: float,
    R_out: float,
    eps1: float,
    eps2: float,
    eps3: float,
    config: Optional[Config] = None,
) -> float:
    """
    Compute smooth quadratic penalty for constraint violations.
    
    Penalty = λ * Σ max(0, violation)²
    
    Parameters
    ----------
    R_in, R_out : float
        Ring radii [m]
    eps1, eps2, eps3 : float
        Angle deviations [rad]
    config : Config, optional
        Configuration object (default: DEFAULT_CONFIG)
    
    Returns
    -------
    float
        Total penalty value (0 if all constraints satisfied)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    penalty = 0.0
    eps4 = -(eps1 + eps2 + eps3)
    
    # Radial gap penalty
    radial_violation = config.DELTA_R_MIN - (R_out - R_in)
    if radial_violation > 0:
        penalty += radial_violation**2
    
    # Minimum spacing penalty
    theta0 = 0.0
    all_pos, _, _ = compute_all_well_positions(R_in, R_out, theta0, eps1, eps2, eps3)
    min_spacing = compute_minimum_spacing(all_pos)
    spacing_violation = config.S_MIN - min_spacing
    if spacing_violation > 0:
        penalty += spacing_violation**2
    
    # Angle increment penalties
    for eps_k in [eps1, eps2, eps3, eps4]:
        increment = np.pi / 2.0 + eps_k
        angle_violation = config.DELTA_THETA_MIN - increment
        if angle_violation > 0:
            penalty += angle_violation**2
    
    return config.PENALTY_LAMBDA * penalty


def x_to_params(x: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Unpack optimization vector to named parameters.
    
    Parameters
    ----------
    x : np.ndarray
        Optimization vector [R_in, R_out, θ0, ε1, ε2, ε3]
    
    Returns
    -------
    tuple
        (R_in, R_out, theta0, eps1, eps2, eps3)
    """
    return float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])


def params_to_x(
    R_in: float,
    R_out: float,
    theta0: float,
    eps1: float,
    eps2: float,
    eps3: float,
) -> np.ndarray:
    """
    Pack named parameters to optimization vector.
    
    Returns
    -------
    np.ndarray
        Optimization vector [R_in, R_out, θ0, ε1, ε2, ε3]
    """
    return np.array([R_in, R_out, theta0, eps1, eps2, eps3])


def get_well_labels() -> Dict[str, str]:
    """
    Get well labels and types.
    
    Returns
    -------
    dict
        Dictionary mapping well name to type ('injector' or 'producer')
    """
    return {
        'P0': 'producer',   # Center producer
        'I1': 'injector',
        'I2': 'injector',
        'I3': 'injector',
        'P1': 'producer',   # Outer producers
        'P2': 'producer',
        'P3': 'producer',
        'P4': 'producer',
    }


def get_default_initial_guess(config: Optional[Config] = None) -> np.ndarray:
    """
    Get a reasonable initial guess for optimization.
    
    Returns
    -------
    np.ndarray
        Initial optimization vector [R_in, R_out, θ0, ε1, ε2, ε3]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Start with middle of allowed ranges and no deviations
    R_in = 0.5 * (config.R_IN_MIN + config.R_IN_MAX)
    R_out = R_in + config.DELTA_R_MIN + 500.0  # Some margin above minimum gap
    R_out = min(R_out, config.R_OUT_MAX)
    theta0 = np.pi / 4.0  # 45° initial rotation
    eps1, eps2, eps3 = 0.0, 0.0, 0.0  # Perfect 90° increments
    
    return np.array([R_in, R_out, theta0, eps1, eps2, eps3])
