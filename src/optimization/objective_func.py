"""
Multi-term objective function for well layout optimization.

Implements weighted cost function combining:
- Thermal breakthrough uniformity
- Pressure balance
- Spacing penalties

The objective is to minimize this cost function.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

# Add parent directory to path to import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from patterns.geometry import Well, minimum_spacing


def compute_total_cost(
    well_positions: np.ndarray,
    n_injectors: int,
    n_producers: int,
    reservoir_params: dict,
    fluid_params: dict,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    min_spacing: float = 500.0,
    flow_rate_per_well: float = 50.0,
) -> float:
    """
    Compute total weighted cost for well layout.
    
    Cost = w1*breakthrough_variance + w2*pressure_variance + w3*spacing_penalty
    
    Parameters
    ----------
    well_positions : np.ndarray
        Flattened array of well positions [x1, y1, x2, y2, ..., xN, yN]
    n_injectors : int
        Number of injector wells
    n_producers : int
        Number of producer wells
    reservoir_params : dict
        Reservoir properties
    fluid_params : dict
        Fluid properties
    weights : tuple of float, optional
        (w_breakthrough, w_pressure, w_spacing) (default: (0.5, 0.3, 0.2))
    min_spacing : float, optional
        Minimum inter-well spacing [m] (default: 500)
    flow_rate_per_well : float, optional
        Flow rate per injector [kg/s] (default: 50.0)
    
    Returns
    -------
    float
        Total cost (lower is better)
    """
    n_wells = n_injectors + n_producers
    
    # Validate input
    if len(well_positions) != 2 * n_wells:
        return 1e9  # Invalid configuration
    
    # Parse positions into Well objects
    wells = []
    for i in range(n_injectors):
        x, y = well_positions[2*i], well_positions[2*i+1]
        wells.append(Well(x=x, y=y, kind='injector'))
    
    for i in range(n_injectors, n_wells):
        x, y = well_positions[2*i], well_positions[2*i+1]
        wells.append(Well(x=x, y=y, kind='producer'))
    
    # Compute individual terms
    w1, w2, w3 = weights
    
    # Term 1: Breakthrough variance
    try:
        bt_variance = breakthrough_variance_term(
            wells, flow_rate_per_well, reservoir_params, fluid_params
        )
    except (ValueError, RuntimeError, ZeroDivisionError):
        bt_variance = 1e6  # Penalize if calculation fails
    
    # Term 2: Pressure variance
    try:
        p_variance = pressure_variance_term(
            wells, flow_rate_per_well, reservoir_params, fluid_params
        )
    except (ValueError, RuntimeError, ZeroDivisionError):
        p_variance = 1e6
    
    # Term 3: Spacing penalty
    spacing_pen = spacing_penalty_term(wells, min_spacing)
    
    # Total cost
    cost = w1 * bt_variance + w2 * p_variance + w3 * spacing_pen
    
    return float(cost)


def breakthrough_variance_term(
    wells: List[Well],
    flow_rate_per_well: float,
    reservoir_params: dict,
    fluid_params: dict,
) -> float:
    """
    Compute normalized variance of breakthrough times.
    
    Parameters
    ----------
    wells : list of Well
        Well layout
    flow_rate_per_well : float
        Flow rate per injector [kg/s]
    reservoir_params : dict
        Reservoir properties
    fluid_params : dict
        Fluid properties
    
    Returns
    -------
    float
        Normalized variance [0, 1+] (0 = perfect uniformity)
    """
    from src.reservoir.heat_depletion import calculate_breakthrough_time
    
    injectors = [w for w in wells if w.kind == 'injector']
    producers = [w for w in wells if w.kind == 'producer']
    
    if len(producers) == 0 or len(injectors) == 0:
        return 1e6
    
    breakthrough_times = []
    
    # For each producer, calculate minimum breakthrough time from any injector
    for prod in producers:
        prod_pos = (prod.x, prod.y)
        min_bt = np.inf
        
        for inj in injectors:
            inj_pos = (inj.x, inj.y)
            
            try:
                bt = calculate_breakthrough_time(
                    inj_pos, prod_pos, flow_rate_per_well,
                    reservoir_params, fluid_params
                )
                min_bt = min(min_bt, bt)
            except Exception:
                continue
        
        if min_bt < np.inf:
            breakthrough_times.append(min_bt)
    
    if len(breakthrough_times) < 2:
        return 0.0  # Can't compute variance
    
    # Coefficient of variation (normalized variance)
    mean_bt = np.mean(breakthrough_times)
    std_bt = np.std(breakthrough_times)
    
    if mean_bt < 1e-6:
        return 1e6
    
    cv = std_bt / mean_bt
    
    return float(cv)


def pressure_variance_term(
    wells: List[Well],
    flow_rate_per_well: float,
    reservoir_params: dict,
    fluid_params: dict,
) -> float:
    """
    Compute normalized variance of pressure drops.
    
    Uses simplified impedance-based calculation for efficiency.
    
    Parameters
    ----------
    wells : list of Well
        Well layout
    flow_rate_per_well : float
        Flow rate per injector [kg/s]
    reservoir_params : dict
        Reservoir properties
    fluid_params : dict
        Fluid properties
    
    Returns
    -------
    float
        Normalized pressure variance [0, 1+]
    """
    injectors = [w for w in wells if w.kind == 'injector']
    producers = [w for w in wells if w.kind == 'producer']
    
    if len(producers) == 0 or len(injectors) == 0:
        return 1e6
    
    # Simplified: compute average distance from each producer to nearest injector
    distances = []
    
    for prod in producers:
        min_dist = np.inf
        for inj in injectors:
            dx = prod.x - inj.x
            dy = prod.y - inj.y
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, dist)
        distances.append(min_dist)
    
    # Coefficient of variation of distances (proxy for pressure balance)
    if len(distances) < 2:
        return 0.0
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    if mean_dist < 1e-6:
        return 1e6
    
    cv = std_dist / mean_dist
    
    return float(cv)


def spacing_penalty_term(
    wells: List[Well],
    min_spacing: float,
) -> float:
    """
    Compute penalty for wells closer than minimum spacing.
    
    Parameters
    ----------
    wells : list of Well
        Well layout
    min_spacing : float
        Minimum required spacing [m]
    
    Returns
    -------
    float
        Penalty value (0 if no violations, >0 otherwise)
    """
    actual_min_spacing = minimum_spacing(wells)
    
    if actual_min_spacing >= min_spacing:
        return 0.0
    
    # Exponential penalty based on violation severity
    violation_ratio = (min_spacing - actual_min_spacing) / min_spacing
    penalty = np.exp(10.0 * violation_ratio) - 1.0
    
    return float(penalty)


def evaluate_layout_quality(
    wells: List[Well],
    flow_rate_per_well: float,
    reservoir_params: dict,
    fluid_params: dict,
) -> Dict[str, float]:
    """
    Comprehensive quality metrics for a well layout.
    
    Returns dictionary with individual terms and diagnostics.
    
    Parameters
    ----------
    wells : list of Well
        Well layout
    flow_rate_per_well : float
        Flow rate per injector [kg/s]
    reservoir_params : dict
        Reservoir properties
    fluid_params : dict
        Fluid properties
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'breakthrough_cv': Coefficient of variation of breakthrough times
        - 'pressure_cv': Coefficient of variation of pressure distances
        - 'min_spacing': Actual minimum spacing [m]
        - 'n_injectors': Number of injectors
        - 'n_producers': Number of producers
        - 'total_cost': Total weighted cost
    """
    bt_cv = breakthrough_variance_term(wells, flow_rate_per_well, reservoir_params, fluid_params)
    p_cv = pressure_variance_term(wells, flow_rate_per_well, reservoir_params, fluid_params)
    min_sp = minimum_spacing(wells)
    
    n_inj = sum(1 for w in wells if w.kind == 'injector')
    n_prod = sum(1 for w in wells if w.kind == 'producer')
    
    return {
        'breakthrough_cv': bt_cv,
        'pressure_cv': p_cv,
        'min_spacing': min_sp,
        'n_injectors': n_inj,
        'n_producers': n_prod,
        'total_cost': 0.5 * bt_cv + 0.3 * p_cv,
    }


if __name__ == "__main__":
    # Test objective function
    print("Testing objective function...")
    
    from patterns.geometry import generate_ring_pattern
    
    # Create test configuration
    reservoir_params = {
        'porosity': 0.10,
        'thickness': 300.0,
        'rock_density': 2650.0,
        'rock_heat_capacity': 1000.0,
    }
    
    fluid_params = {
        'density': 600.0,
        'heat_capacity': 1200.0,
    }
    
    # Generate initial pattern
    injectors, producers = generate_ring_pattern(
        n_inj=3, n_prod=5, R_inj=500.0, R_prod=1000.0
    )
    
    wells = injectors + producers
    
    # Flatten to decision variables
    x = np.array([coord for w in wells for coord in [w.x, w.y]])
    
    # Compute cost
    cost = compute_total_cost(
        x, n_injectors=3, n_producers=5,
        reservoir_params=reservoir_params,
        fluid_params=fluid_params,
    )
    
    print(f"Total cost: {cost:.4f}")
    
    # Quality metrics
    quality = evaluate_layout_quality(wells, 50.0, reservoir_params, fluid_params)
    print(f"Breakthrough CV: {quality['breakthrough_cv']:.4f}")
    print(f"Pressure CV: {quality['pressure_cv']:.4f}")
    print(f"Min spacing: {quality['min_spacing']:.1f} m")
    
    print("✓ Objective function tests complete")
