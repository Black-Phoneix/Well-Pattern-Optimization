"""
Geometric and operational constraints for well layout optimization.

This module implements constraint checking functions for:
- Minimum well spacing
- Maximum pressure drops
- Field boundaries
- Well type placement rules
"""

import numpy as np
from typing import List, Tuple, Optional


def check_minimum_spacing(
    well_positions: List[Tuple[float, float]],
    d_min: float,
) -> bool:
    """
    Check if all wells satisfy minimum spacing constraint.
    
    Parameters
    ----------
    well_positions : list of tuples
        List of (x, y) positions [m]
    d_min : float
        Minimum required spacing [m]
    
    Returns
    -------
    bool
        True if constraint satisfied, False otherwise
    """
    n = len(well_positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = well_positions[i][0] - well_positions[j][0]
            dy = well_positions[i][1] - well_positions[j][1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < d_min:
                return False
    
    return True


def check_pressure_limits(
    well_layout: List[Tuple[Tuple[float, float], str]],
    flow_rates: np.ndarray,
    reservoir_params: dict,
    dP_max: float,
) -> bool:
    """
    Check if pressure drops are within acceptable limits.
    
    Parameters
    ----------
    well_layout : list of tuples
        Each element is ((x, y), kind) where kind is 'injector' or 'producer'
    flow_rates : np.ndarray
        Flow rates [kg/s] for each well
    reservoir_params : dict
        Reservoir properties
    dP_max : float
        Maximum allowable pressure drop [Pa]
    
    Returns
    -------
    bool
        True if all pressure drops are below limit
    
    Notes
    -----
    This is a simplified check. Full implementation would integrate
    with pressure_network module.
    """
    # Simplified check: assume uniform pressure distribution
    # In practice, would compute actual pressure field
    
    # For now, just check that configuration is reasonable
    n_inj = sum(1 for _, kind in well_layout if kind == 'injector')
    n_prod = sum(1 for _, kind in well_layout if kind == 'producer')
    
    if n_inj == 0 or n_prod == 0:
        return False
    
    # Check for reasonable flow rates
    max_flow = np.max(np.abs(flow_rates))
    if max_flow > 200.0:  # kg/s
        return False
    
    return True


def check_field_boundary(
    well_positions: List[Tuple[float, float]],
    boundary_params: dict,
) -> bool:
    """
    Check if all wells are within field boundary.
    
    Parameters
    ----------
    well_positions : list of tuples
        List of (x, y) positions [m]
    boundary_params : dict
        Boundary specification with 'type' key:
        - 'circular': requires 'radius' [m]
        - 'rectangular': requires 'x_min', 'x_max', 'y_min', 'y_max' [m]
    
    Returns
    -------
    bool
        True if all wells within boundary
    """
    boundary_type = boundary_params.get('type', 'circular')
    
    if boundary_type == 'circular':
        radius = boundary_params['radius']
        for x, y in well_positions:
            if np.sqrt(x**2 + y**2) > radius:
                return False
    
    elif boundary_type == 'rectangular':
        x_min = boundary_params['x_min']
        x_max = boundary_params['x_max']
        y_min = boundary_params['y_min']
        y_max = boundary_params['y_max']
        
        for x, y in well_positions:
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                return False
    
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    return True


def apply_all_constraints(
    x: np.ndarray,
    n_injectors: int,
    n_producers: int,
    min_spacing: float,
    field_radius: float,
    penalty_value: float = 1e9,
) -> float:
    """
    Apply all constraints and return penalty value for optimizer.
    
    Parameters
    ----------
    x : np.ndarray
        Decision variables [x1, y1, x2, y2, ..., xN, yN]
    n_injectors : int
        Number of injector wells
    n_producers : int
        Number of producer wells
    min_spacing : float
        Minimum inter-well spacing [m]
    field_radius : float
        Field boundary radius [m]
    penalty_value : float, optional
        Penalty for constraint violation (default: 1e9)
    
    Returns
    -------
    float
        0 if all constraints satisfied, penalty_value otherwise
    """
    n_wells = n_injectors + n_producers
    
    # Reshape decision variables
    if len(x) != 2 * n_wells:
        return penalty_value
    
    positions = [(x[2*i], x[2*i+1]) for i in range(n_wells)]
    
    # Check minimum spacing
    if not check_minimum_spacing(positions, min_spacing):
        return penalty_value
    
    # Check field boundary
    boundary_params = {'type': 'circular', 'radius': field_radius}
    if not check_field_boundary(positions, boundary_params):
        return penalty_value
    
    return 0.0


def spacing_penalty_smooth(
    well_positions: List[Tuple[float, float]],
    d_min: float,
    k: float = 10.0,
) -> float:
    """
    Smooth penalty function for spacing constraint.
    
    Uses smooth penalty that increases exponentially as wells get closer.
    Useful for gradient-based optimizers.
    
    Parameters
    ----------
    well_positions : list of tuples
        List of (x, y) positions [m]
    d_min : float
        Minimum desired spacing [m]
    k : float, optional
        Penalty steepness parameter (default: 10.0)
    
    Returns
    -------
    float
        Penalty value (0 if d > d_min, increases as d < d_min)
    """
    n = len(well_positions)
    penalty = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = well_positions[i][0] - well_positions[j][0]
            dy = well_positions[i][1] - well_positions[j][1]
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < d_min:
                # Exponential penalty
                violation = (d_min - dist) / d_min
                penalty += np.exp(k * violation) - 1.0
    
    return float(penalty)


def constraint_violation_count(
    well_positions: List[Tuple[float, float]],
    min_spacing: float,
    field_radius: float,
) -> int:
    """
    Count number of constraint violations.
    
    Useful for diagnostic purposes.
    
    Parameters
    ----------
    well_positions : list of tuples
        List of (x, y) positions [m]
    min_spacing : float
        Minimum inter-well spacing [m]
    field_radius : float
        Field boundary radius [m]
    
    Returns
    -------
    int
        Number of violated constraints
    """
    violations = 0
    
    # Spacing violations
    n = len(well_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = well_positions[i][0] - well_positions[j][0]
            dy = well_positions[i][1] - well_positions[j][1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_spacing:
                violations += 1
    
    # Boundary violations
    for x, y in well_positions:
        if np.sqrt(x**2 + y**2) > field_radius:
            violations += 1
    
    return violations


if __name__ == "__main__":
    # Test constraint functions
    print("Testing constraint functions...")
    
    # Test case 1: Valid spacing
    positions_valid = [(0, 0), (600, 0), (0, 600)]
    assert check_minimum_spacing(positions_valid, 500.0)
    print("✓ Minimum spacing check passed")
    
    # Test case 2: Invalid spacing
    positions_invalid = [(0, 0), (200, 0), (0, 600)]
    assert not check_minimum_spacing(positions_invalid, 500.0)
    print("✓ Minimum spacing violation detected")
    
    # Test case 3: Circular boundary
    assert check_field_boundary(positions_valid, {'type': 'circular', 'radius': 1000.0})
    assert not check_field_boundary([(2000, 0)], {'type': 'circular', 'radius': 1000.0})
    print("✓ Boundary checks passed")
    
    # Test case 4: Smooth penalty
    penalty = spacing_penalty_smooth(positions_invalid, 500.0)
    print(f"  Smooth penalty for violation: {penalty:.2f}")
    assert penalty > 0
    print("✓ Smooth penalty function working")
    
    print("✓ All constraint tests passed")
