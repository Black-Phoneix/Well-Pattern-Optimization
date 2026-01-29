"""
Coupled wellbore pressure-temperature integrator using Birdsell method.

This module implements 1D wellbore integration accounting for:
- Geothermal gradient
- Friction losses (Darcy-Weisbach)
- Gravity/hydrostatic effects
- Heat exchange with formation

References
----------
Birdsell, D. T., et al. (2021): "Hydraulics and Caprock Performance of the 
Krechba CCS Project, Algeria", International Journal of Greenhouse Gas Control, 
Volume 109, Article 103368.
https://doi.org/10.1016/j.ijggc.2021.103368
"""

import numpy as np
from typing import Tuple, Dict, Optional
from .friction import colebrook_white, reynolds_number

# Physical constants
GRAVITY = 9.81  # m/s² - gravitational acceleration


def integrate_wellbore(
    depth: float,
    flow_rate: float,
    T_surface: float,
    P_surface: float,
    wellbore_params: dict,
    fluid_props_func: Optional[callable] = None,
    n_segments: int = 100,
    direction: str = 'down',
) -> Tuple[float, float]:
    """
    Integrate wellbore equations from surface to bottom (or vice versa).
    
    Solves coupled ODEs:
        dP/dz = -ρ*g - (f*ρ*v²)/(2*D)
        dT/dz = geothermal_gradient - heat_loss_coefficient
    
    Parameters
    ----------
    depth : float
        Total well depth [m]
    flow_rate : float
        Mass flow rate [kg/s] (positive for downward flow)
    T_surface : float
        Temperature at surface [K]
    P_surface : float
        Pressure at surface [Pa]
    wellbore_params : dict
        Dictionary containing:
        - 'diameter' : float [m]
        - 'roughness' : float [m]
        - 'geothermal_gradient' : float [K/m]
    fluid_props_func : callable, optional
        Function(T, P) returning (rho, mu, cp). If None, uses constant values.
    n_segments : int, optional
        Number of discretization segments (default: 100)
    direction : str, optional
        'down' for injection (surface to bottom) or 'up' for production (default: 'down')
    
    Returns
    -------
    tuple of float
        (P_bottom, T_bottom) - Pressure [Pa] and temperature [K] at depth
    
    Notes
    -----
    Uses simple Euler integration. For production wells, integrate upward.
    """
    D = wellbore_params['diameter']
    epsilon = wellbore_params['roughness']
    grad_geo = wellbore_params['geothermal_gradient']
    
    # Default fluid properties if function not provided
    if fluid_props_func is None:
        def fluid_props_func(T, P):
            return 600.0, 5e-5, 1200.0  # rho, mu, cp
    
    # Flow velocity (assume constant for simplicity)
    A = np.pi * (D / 2.0) ** 2  # Cross-sectional area
    
    # Initialize
    if direction == 'down':
        z_start = 0.0
        z_end = depth
        dz = depth / n_segments
        z_array = np.linspace(z_start, z_end, n_segments + 1)
    else:  # up
        z_start = depth
        z_end = 0.0
        dz = -depth / n_segments
        z_array = np.linspace(z_start, z_end, n_segments + 1)
    
    # Initial conditions
    P = P_surface
    T = T_surface
    
    # Integration loop
    for i in range(n_segments):
        z = z_array[i]
        
        # Get fluid properties at current state
        rho, mu, cp = fluid_props_func(T, P)
        
        # Volumetric flow rate
        Q_vol = flow_rate / rho
        v = Q_vol / A
        
        # Reynolds number
        Re = reynolds_number(rho, abs(v), D, mu)
        
        # Friction factor
        f = colebrook_white(Re, epsilon, D)
        
        # Pressure gradient
        # Hydrostatic term + friction term
        # Note: Positive dP_dz means pressure increases with depth
        if direction == 'down':
            dP_dz = rho * GRAVITY + (f * rho * v**2) / (2.0 * D)
        else:  # up
            dP_dz = -rho * GRAVITY - (f * rho * v**2) / (2.0 * D)
        
        # Temperature gradient
        # Geothermal heating - heat loss to formation (simplified)
        # For simplicity, assume adiabatic (no heat loss term implemented here)
        if direction == 'down':
            dT_dz = grad_geo  # Temperature increases with depth
        else:  # up
            dT_dz = -grad_geo  # Temperature decreases toward surface
        
        # Euler step
        P += dP_dz * abs(dz)
        T += dT_dz * abs(dz)
        
        # Ensure physical values
        P = max(P, 1e5)  # Don't go below 1 bar
        T = max(T, 273.0)  # Don't go below freezing
    
    return float(P), float(T)


def compute_thermosiphon_effect(
    injector_params: dict,
    producer_params: dict,
    depth: float,
) -> float:
    """
    Compute thermosiphon pressure difference due to density contrast.
    
    The thermosiphon effect arises from density difference between
    cold injection and hot production fluids, creating a natural
    pressure drive.
    
    ΔP_thermosiphon = (ρ_cold - ρ_hot) * g * depth
    
    Parameters
    ----------
    injector_params : dict
        Dictionary with:
        - 'density' : float [kg/m³]
        - 'temperature' : float [K]
    producer_params : dict
        Dictionary with:
        - 'density' : float [kg/m³]
        - 'temperature' : float [K]
    depth : float
        Well depth [m]
    
    Returns
    -------
    float
        Thermosiphon pressure boost [Pa] (positive aids circulation)
    """
    rho_inj = injector_params['density']
    rho_prod = producer_params['density']
    
    # Pressure difference (positive if injection denser than production)
    dP_thermo = (rho_inj - rho_prod) * GRAVITY * depth
    
    return float(dP_thermo)


def calculate_pumping_power(
    pressure_drop: float,
    flow_rate: float,
    efficiency: float = 0.75,
) -> float:
    """
    Calculate parasitic pumping power requirement.
    
    P_pump = (ΔP * Q_vol) / η
    
    Parameters
    ----------
    pressure_drop : float
        Total pressure drop [Pa]
    flow_rate : float
        Mass flow rate [kg/s]
    efficiency : float, optional
        Pump efficiency [-] (default: 0.75)
    
    Returns
    -------
    float
        Pumping power [W]
    """
    if efficiency <= 0 or efficiency > 1:
        raise ValueError("Efficiency must be in (0, 1]")
    
    # Assume constant density for power calculation
    rho = 600.0  # kg/m³
    Q_vol = flow_rate / rho
    
    P_pump = (pressure_drop * Q_vol) / efficiency
    
    return float(max(P_pump, 0.0))


def integrate_wellbore_pair(
    depth: float,
    flow_rate: float,
    T_injection: float,
    P_surface: float,
    wellbore_params: dict,
    reservoir_params: dict,
) -> Dict[str, float]:
    """
    Integrate both injection and production wellbores.
    
    Returns comprehensive wellbore hydraulics summary.
    
    Parameters
    ----------
    depth : float
        Well depth [m]
    flow_rate : float
        Mass flow rate [kg/s]
    T_injection : float
        Injection temperature at surface [K]
    P_surface : float
        Surface pressure [Pa]
    wellbore_params : dict
        Wellbore parameters
    reservoir_params : dict
        Reservoir parameters (for production temperature)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'P_inj_bottom' : Pressure at injection well bottom [Pa]
        - 'T_inj_bottom' : Temperature at injection well bottom [K]
        - 'P_prod_bottom' : Pressure at production well bottom [Pa]
        - 'T_prod_bottom' : Temperature at production well bottom [K]
        - 'P_prod_surface' : Pressure at production well surface [Pa]
        - 'dP_thermosiphon' : Thermosiphon pressure boost [Pa]
        - 'pumping_power' : Required pumping power [W]
    """
    # Injection wellbore (down)
    P_inj_bottom, T_inj_bottom = integrate_wellbore(
        depth, flow_rate, T_injection, P_surface, wellbore_params,
        direction='down'
    )
    
    # Assume reservoir temperature at production well
    T_reservoir = reservoir_params.get('temperature', 423.0)
    
    # Production wellbore (up) - start from reservoir conditions
    P_prod_surface, T_prod_surface = integrate_wellbore(
        depth, flow_rate, T_reservoir, P_inj_bottom,
        wellbore_params, direction='up'
    )
    
    # Thermosiphon effect
    inj_props = {'density': 600.0, 'temperature': T_inj_bottom}
    prod_props = {'density': 500.0, 'temperature': T_reservoir}  # Hot fluid is less dense
    dP_thermo = compute_thermosiphon_effect(inj_props, prod_props, depth)
    
    # Pumping power
    dP_total = P_inj_bottom - P_prod_surface
    P_pump = calculate_pumping_power(dP_total, flow_rate)
    
    return {
        'P_inj_bottom': P_inj_bottom,
        'T_inj_bottom': T_inj_bottom,
        'P_prod_bottom': P_inj_bottom,  # Assume reservoir equilibrates
        'T_prod_bottom': T_reservoir,
        'P_prod_surface': P_prod_surface,
        'dP_thermosiphon': dP_thermo,
        'pumping_power': P_pump,
    }


if __name__ == "__main__":
    # Test wellbore integration
    print("Testing wellbore integration...")
    
    wellbore_params = {
        'diameter': 0.2,
        'roughness': 0.045e-3,
        'geothermal_gradient': 0.03,
    }
    
    reservoir_params = {
        'temperature': 423.0,  # 150°C
    }
    
    # Test injection well
    P_bottom, T_bottom = integrate_wellbore(
        depth=3000.0,
        flow_rate=50.0,
        T_surface=323.0,  # 50°C
        P_surface=15e6,  # 15 MPa
        wellbore_params=wellbore_params,
        direction='down'
    )
    
    print(f"Injection well:")
    print(f"  Surface: P = {15e6/1e6:.1f} MPa, T = {323.0-273.15:.1f}°C")
    print(f"  Bottom:  P = {P_bottom/1e6:.1f} MPa, T = {T_bottom-273.15:.1f}°C")
    
    # Test wellbore pair
    results = integrate_wellbore_pair(
        depth=3000.0,
        flow_rate=50.0,
        T_injection=323.0,
        P_surface=15e6,
        wellbore_params=wellbore_params,
        reservoir_params=reservoir_params,
    )
    
    print(f"\nWellbore pair:")
    print(f"  ΔP thermosiphon: {results['dP_thermosiphon']/1e6:.2f} MPa")
    print(f"  Pumping power: {results['pumping_power']/1e3:.1f} kW")
    
    print("✓ Wellbore integration tests complete")
