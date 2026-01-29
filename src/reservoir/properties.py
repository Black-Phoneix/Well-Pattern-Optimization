"""
Fluid thermophysical properties for supercritical CO₂.

This module provides a CoolProp wrapper for temperature and pressure-dependent
fluid properties, with graceful fallback to constant values if CoolProp is unavailable.

References
----------
CoolProp: http://www.coolprop.org/
"""

import warnings
from typing import Optional

import numpy as np

# Try to import CoolProp
try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    warnings.warn(
        "CoolProp not available. Using constant fallback values for fluid properties. "
        "Install CoolProp for temperature/pressure-dependent properties: pip install CoolProp",
        RuntimeWarning
    )


class FluidProperties:
    """
    Wrapper for sCO₂ thermophysical properties.
    
    Uses CoolProp if available, otherwise falls back to constant values.
    
    Parameters
    ----------
    use_coolprop : bool, optional
        Force use of CoolProp (raises error if unavailable). Default: True if available.
    fallback_density : float, optional
        Fallback density [kg/m³]. Default: 600.0
    fallback_viscosity : float, optional
        Fallback dynamic viscosity [Pa·s]. Default: 5e-5
    fallback_heat_capacity : float, optional
        Fallback specific heat capacity [J/(kg·K)]. Default: 1200.0
    """
    
    def __init__(
        self,
        use_coolprop: Optional[bool] = None,
        fallback_density: float = 600.0,
        fallback_viscosity: float = 5e-5,
        fallback_heat_capacity: float = 1200.0,
    ):
        if use_coolprop is None:
            self.use_coolprop = COOLPROP_AVAILABLE
        else:
            self.use_coolprop = use_coolprop
            if use_coolprop and not COOLPROP_AVAILABLE:
                raise ImportError("CoolProp requested but not available")
        
        self.fallback_density = fallback_density
        self.fallback_viscosity = fallback_viscosity
        self.fallback_heat_capacity = fallback_heat_capacity
        
        # Fluid identifier for CoolProp
        self.fluid = "CO2"
    
    def get_density(self, T: float, P: float) -> float:
        """
        Calculate fluid density.
        
        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        
        Returns
        -------
        float
            Density [kg/m³]
        """
        if self.use_coolprop:
            try:
                return float(PropsSI('D', 'T', T, 'P', P, self.fluid))
            except Exception as e:
                warnings.warn(f"CoolProp failed: {e}. Using fallback value.", RuntimeWarning)
                return self.fallback_density
        return self.fallback_density
    
    def get_viscosity(self, T: float, P: float) -> float:
        """
        Calculate fluid dynamic viscosity.
        
        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        
        Returns
        -------
        float
            Dynamic viscosity [Pa·s]
        """
        if self.use_coolprop:
            try:
                return float(PropsSI('V', 'T', T, 'P', P, self.fluid))
            except Exception as e:
                warnings.warn(f"CoolProp failed: {e}. Using fallback value.", RuntimeWarning)
                return self.fallback_viscosity
        return self.fallback_viscosity
    
    def get_enthalpy(self, T: float, P: float) -> float:
        """
        Calculate fluid specific enthalpy.
        
        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        
        Returns
        -------
        float
            Specific enthalpy [J/kg]
        """
        if self.use_coolprop:
            try:
                return float(PropsSI('H', 'T', T, 'P', P, self.fluid))
            except Exception as e:
                warnings.warn(f"CoolProp failed: {e}. Using cp*T approximation.", RuntimeWarning)
                return self.fallback_heat_capacity * T
        return self.fallback_heat_capacity * T
    
    def get_heat_capacity(self, T: float, P: float) -> float:
        """
        Calculate fluid specific heat capacity at constant pressure.
        
        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Pressure [Pa]
        
        Returns
        -------
        float
            Specific heat capacity [J/(kg·K)]
        """
        if self.use_coolprop:
            try:
                return float(PropsSI('C', 'T', T, 'P', P, self.fluid))
            except Exception as e:
                warnings.warn(f"CoolProp failed: {e}. Using fallback value.", RuntimeWarning)
                return self.fallback_heat_capacity
        return self.fallback_heat_capacity


# Convenience functions using default FluidProperties instance
_default_properties = FluidProperties()


def get_density(T: float, P: float) -> float:
    """Get density [kg/m³] at temperature T [K] and pressure P [Pa]."""
    return _default_properties.get_density(T, P)


def get_viscosity(T: float, P: float) -> float:
    """Get dynamic viscosity [Pa·s] at temperature T [K] and pressure P [Pa]."""
    return _default_properties.get_viscosity(T, P)


def get_enthalpy(T: float, P: float) -> float:
    """Get specific enthalpy [J/kg] at temperature T [K] and pressure P [Pa]."""
    return _default_properties.get_enthalpy(T, P)


def get_heat_capacity(T: float, P: float) -> float:
    """Get specific heat capacity [J/(kg·K)] at temperature T [K] and pressure P [Pa]."""
    return _default_properties.get_heat_capacity(T, P)
