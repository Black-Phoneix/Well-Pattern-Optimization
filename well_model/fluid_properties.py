# core/fluid_properties.py
"""
Module for fluid properties extraction using CoolProp.
Includes functions to extract enthalpy, density, viscosity, temperature, and more.
"""

from CoolProp.CoolProp import PropsSI
import numpy as np

FLUID_CO2 = "CO2"

def get_props_T(temp, pres, fluid=FLUID_CO2):
    """
    Extract basic thermodynamic properties of a fluid.

    Parameters:
        temp (float): Fluid temperature in Kelvin [K].
        pres (float): Fluid pressure in Pascal [Pa].
        fluid (str): Fluid name (default: FLUID_CO2).

    Returns:
        tuple: (enthalpy [J/kg], density [kg/m³], dynamic viscosity [Pa·s])
    """
    h = PropsSI('H', 'T', temp, 'P', pres, fluid)
    rho = PropsSI('D', 'H', h, 'P', pres, fluid)
    mu = PropsSI('V', 'H', h, 'P', pres, fluid)

    return h, rho, mu

def get_property_h(enthalpy, pres, fluid=FLUID_CO2):
    """
    Extract basic thermodynamic properties of a fluid.

    Parameters:
        enthalpy (float): Fluid enthalpy [J/kg].
        pres (float): Fluid pressure in Pascal [Pa].
        fluid (str): Fluid name (default: FLUID_CO2).

    Returns:
        tuple: (enthalpy [J/kg], density [kg/m³], dynamic viscosity [Pa·s])
    """
    T = PropsSI('T', 'H', enthalpy, 'P', pres, fluid)
    rho = PropsSI('D', 'H', enthalpy, 'P', pres, fluid)
    mu = PropsSI('V', 'H', enthalpy, 'P', pres, fluid)

    return T, rho, mu


def wet_bulb_temperature_calc(T, RH):
    """
    Estimate the wet bulb temperature (°C) from ambient temperature (°C)
    and relative humidity (%) using Stull's approximation.

    Parameters:
    T  : float or np.ndarray - Ambient temperature in °C
    RH : float or np.ndarray - Relative humidity in %

    Returns:
    Tw : float or np.ndarray - Wet bulb temperature in °C (approximate)
    """
    Tw = (
            T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
            + 0.00391838 * RH**1.5 * np.arctan(0.023101 * RH)
            - np.arctan(RH - 1.676331)
            + np.arctan(T + RH)
            - 4.686035
          )

    return Tw
