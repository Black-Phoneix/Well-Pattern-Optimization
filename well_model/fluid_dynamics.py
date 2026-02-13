# core\fluid_dynamics.py
"""
This module provides core functions for simulating thermodynamic and hydraulic evolution
of a working fluid (e.g., CO2) in a geothermal production well.

It implements the finite volume discretization approach and includes:
- Darcy-Weisbach-based pressure loss
- Reynolds number-based friction factor
- Initial conditions calculation

References:
- Fleming et al. (2020): "Thermodynamic modeling of CO2-based geothermal systems"
"""


import numpy as np
from GeoCaP.core.fluid_properties import get_props_T


def friction_factor(Re, D, epsilon):
    """
    Compute Darcy friction factor.

    References:
    - Schifflechner et al. (2024) "Annual Performance Profiles of CO2-Plume Geothermal (CPG) Systems:
    Impact of the Ambient Conditions."

    Parameters
    ----------
    Re : float
        Reynolds number [-].
    D : float
        Pipe diameter [m].
    epsilon : float, optional
        Pipe roughness [m], default is 55e-6.

    Returns
    -------
    float
        Darcy friction factor [-].
    """
    if Re < 2000:
        return 64 / Re  # laminar
    return 1/ (-1.8 * np.log10( (epsilon / (3.7 * D))**1.11 + 6.9 / Re))**2


def pressure_drop_friction(mass_flow, rho, D, L, f):
    """
    Compute pressure drop due to friction using the Darcy-Weisbach equation.

    References:
        - Fleming et al. (2020): "Thermodynamic modeling of CO2-based geothermal systems".

    Parameters
    ----------
    mass_flow : float
        Mass flow rate [kg/s].
    rho : float
        Fluid density [kg/m³].
    D : float
        Pipe diameter [m].
    L : float
        Pipe segment length [m].
    f : float
        Darcy friction factor [-].

    Returns
    -------
    float
        Pressure drop [Pa].
    """
    return f * (L / D) * (8 * mass_flow ** 2) / (rho * np.pi ** 2 * D ** 4)


def initial_conditions(depth: float, temp_ambient: float, gradient: float, fluid: str = "CO2"):
    """
    Set initial temperature and pressure conditions based on depth and fluid type.

    Parameters
    ----------
    depth : float
        Total well depth [m].
    temp_ambient : float
        Surface temperature [K].
    gradient : float
        Geothermal temperature gradient [K/km].
    fluid : str
        Working fluid type (default is 'CO2').

    Returns
    -------
    tuple
        Initial temperature [K], pressure [Pa], enthalpy [J], density [kg/m³], and viscosity [Pa.s].
    """
    T0 = gradient * depth / 1000 + temp_ambient  # Bulk temperature from geothermal gradient
    P0 = depth * 1000 * 9.81                     # Hydrostatic pressure at the reservoir
    h0, rho0, mu0 = get_props_T(T0, P0, fluid)  # Retrieve initial fluid properties

    return T0, P0, h0, rho0, mu0
