"""
Heat depletion tracker using Adams breakthrough proxy model.

This module implements simplified analytical models for thermal lifetime prediction
and breakthrough time estimation in geothermal doublet systems.

References
----------
Adams, B. M., et al. (2020): "Estimating the Geothermal Electricity Generation 
Potential of Sedimentary Basins Using genGEO", Geothermal Energy, Volume 8, Article 32.
https://doi.org/10.1186/s40517-020-00185-0
"""

import numpy as np
from typing import Tuple, List, Optional


def calculate_breakthrough_time(
    injector_pos: Tuple[float, float],
    producer_pos: Tuple[float, float],
    flow_rate: float,
    reservoir_params: dict,
    fluid_params: dict,
) -> float:
    """
    Calculate breakthrough time for a single injector-producer pair.
    
    Uses simplified advection-based model:
    
        t_breakthrough ≈ (distance * φ * ρ_fluid * cp_fluid) / (q * ρ_rock * cp_rock)
    
    This accounts for:
    - Advective transport (Darcy velocity)
    - Thermal capacity ratio (fluid vs rock)
    - Geometric spreading
    
    Parameters
    ----------
    injector_pos : tuple of float
        Injector position (x, y) [m]
    producer_pos : tuple of float
        Producer position (x, y) [m]
    flow_rate : float
        Mass flow rate [kg/s]
    reservoir_params : dict
        Dictionary containing:
        - 'porosity' : float [-]
        - 'thickness' : float [m]
        - 'rock_density' : float [kg/m³]
        - 'rock_heat_capacity' : float [J/(kg·K)]
    fluid_params : dict
        Dictionary containing:
        - 'density' : float [kg/m³]
        - 'heat_capacity' : float [J/(kg·K)]
    
    Returns
    -------
    float
        Breakthrough time [years]
    
    Notes
    -----
    This is a simplified 1D advection model. For more accurate results,
    use numerical reservoir simulation.
    """
    # Calculate distance
    dx = producer_pos[0] - injector_pos[0]
    dy = producer_pos[1] - injector_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Extract parameters
    phi = reservoir_params['porosity']
    h = reservoir_params['thickness']
    rho_rock = reservoir_params['rock_density']
    cp_rock = reservoir_params['rock_heat_capacity']
    
    rho_fluid = fluid_params['density']
    cp_fluid = fluid_params['heat_capacity']
    
    # Volumetric flow rate [m³/s]
    Q_vol = flow_rate / rho_fluid
    
    # Pore volume between injector and producer (cylindrical approximation)
    # Assume effective swept width proportional to distance
    A_cross = h * distance  # Simplified cross-sectional area
    V_pore = A_cross * distance * phi
    
    # Thermal capacity ratio
    # Heat stored in rock vs heat carried by fluid
    heat_capacity_rock = (1.0 - phi) * rho_rock * cp_rock
    heat_capacity_fluid = phi * rho_fluid * cp_fluid
    
    # Effective thermal retardation factor
    R_thermal = (heat_capacity_rock + heat_capacity_fluid) / (rho_fluid * cp_fluid)
    
    # Breakthrough time (advection with thermal retardation)
    if Q_vol <= 1e-10:
        return 1e9  # Essentially infinite for zero flow
    
    # Residence time
    t_residence = V_pore / Q_vol
    
    # Apply thermal retardation
    t_breakthrough = t_residence * R_thermal
    
    # Convert seconds to years
    t_breakthrough_years = t_breakthrough / (365.25 * 24 * 3600)
    
    return float(t_breakthrough_years)


def compute_breakthrough_variance(
    well_layout: List[Tuple[Tuple[float, float], str]],
    flow_rates: np.ndarray,
    reservoir_params: dict,
    fluid_params: dict,
) -> float:
    """
    Compute variance in breakthrough times across all producer wells.
    
    This serves as an optimization objective: minimizing variance leads
    to more uniform thermal depletion.
    
    Parameters
    ----------
    well_layout : list of tuples
        Each element is ((x, y), kind) where kind is 'injector' or 'producer'
    flow_rates : np.ndarray
        Flow rates [kg/s] for each well (positive=injection, negative=production)
    reservoir_params : dict
        Reservoir properties (see calculate_breakthrough_time)
    fluid_params : dict
        Fluid properties (see calculate_breakthrough_time)
    
    Returns
    -------
    float
        Variance of breakthrough times [years²]
    
    Notes
    -----
    For each producer, breakthrough time is calculated as the weighted average
    of contributions from all injectors.
    """
    injectors = [(pos, i) for i, (pos, kind) in enumerate(well_layout) if kind == 'injector']
    producers = [(pos, i) for i, (pos, kind) in enumerate(well_layout) if kind == 'producer']
    
    if len(producers) == 0:
        return 0.0
    
    breakthrough_times = []
    
    for prod_pos, prod_idx in producers:
        # For each producer, find minimum breakthrough time from all injectors
        min_breakthrough = np.inf
        
        for inj_pos, inj_idx in injectors:
            # Use absolute value of flow rate
            flow = abs(flow_rates[inj_idx])
            
            if flow > 1e-6:
                t_bt = calculate_breakthrough_time(
                    inj_pos, prod_pos, flow, reservoir_params, fluid_params
                )
                min_breakthrough = min(min_breakthrough, t_bt)
        
        if min_breakthrough < np.inf:
            breakthrough_times.append(min_breakthrough)
    
    if len(breakthrough_times) == 0:
        return 0.0
    
    return float(np.var(breakthrough_times))


def estimate_temperature_decline(
    injector_pos: Tuple[float, float],
    producer_pos: Tuple[float, float],
    flow_rate: float,
    T_injection: float,
    T_reservoir: float,
    time_array: np.ndarray,
    reservoir_params: dict,
    fluid_params: dict,
) -> np.ndarray:
    """
    Estimate temperature decline curve at producer well.
    
    Uses simplified heat balance model with exponential decline after breakthrough.
    
    Parameters
    ----------
    injector_pos : tuple of float
        Injector position (x, y) [m]
    producer_pos : tuple of float
        Producer position (x, y) [m]
    flow_rate : float
        Mass flow rate [kg/s]
    T_injection : float
        Injection temperature [K]
    T_reservoir : float
        Initial reservoir temperature [K]
    time_array : np.ndarray
        Time points [years]
    reservoir_params : dict
        Reservoir properties
    fluid_params : dict
        Fluid properties
    
    Returns
    -------
    np.ndarray
        Temperature at producer [K] as function of time
    
    Notes
    -----
    Before breakthrough: T = T_reservoir
    After breakthrough: T declines gradually toward T_injection
    """
    # Calculate breakthrough time
    t_bt = calculate_breakthrough_time(
        injector_pos, producer_pos, flow_rate, reservoir_params, fluid_params
    )
    
    # Temperature decline model
    T_profile = np.zeros_like(time_array)
    
    for i, t in enumerate(time_array):
        if t < t_bt:
            # Before breakthrough: reservoir temperature
            T_profile[i] = T_reservoir
        else:
            # After breakthrough: exponential decline
            # Characteristic time scale (rough approximation)
            tau = t_bt * 0.5  # Decline time scale
            
            # Temperature approaches injection temperature
            dt = t - t_bt
            T_profile[i] = T_injection + (T_reservoir - T_injection) * np.exp(-dt / tau)
    
    return T_profile


def compute_thermal_power(
    T_production: float,
    T_injection: float,
    flow_rate: float,
    cp_fluid: float,
) -> float:
    """
    Compute thermal power output.
    
    Parameters
    ----------
    T_production : float
        Production temperature [K]
    T_injection : float
        Injection temperature [K]
    flow_rate : float
        Mass flow rate [kg/s]
    cp_fluid : float
        Fluid heat capacity [J/(kg·K)]
    
    Returns
    -------
    float
        Thermal power [W]
    """
    return flow_rate * cp_fluid * (T_production - T_injection)


if __name__ == "__main__":
    # Example usage
    print("Testing heat depletion model...")
    
    # Test parameters
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
    
    # Test case: 1000m spacing, 50 kg/s flow
    inj_pos = (0.0, 0.0)
    prod_pos = (1000.0, 0.0)
    flow = 50.0  # kg/s
    
    t_bt = calculate_breakthrough_time(inj_pos, prod_pos, flow, reservoir_params, fluid_params)
    print(f"Breakthrough time: {t_bt:.1f} years")
    
    # Temperature decline
    time_years = np.linspace(0, 50, 100)
    T_profile = estimate_temperature_decline(
        inj_pos, prod_pos, flow, 323.0, 423.0, time_years, reservoir_params, fluid_params
    )
    
    print(f"Initial temperature: {T_profile[0]:.1f} K")
    print(f"Final temperature: {T_profile[-1]:.1f} K")
    print("✓ Heat depletion model test complete")
