"""
Centralized configuration for well layout optimization.

This module defines parameter dataclasses for all physics models,
ensuring consistent units (SI) and documentation throughout the codebase.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ReservoirParams:
    """
    Reservoir properties for geothermal system modeling.
    
    All parameters in SI units.
    
    Attributes
    ----------
    permeability : float
        Intrinsic permeability [m²] (default: 5e-14 = 50 mD)
    porosity : float
        Porosity [-] (default: 0.10)
    thickness : float
        Reservoir thickness [m] (default: 300)
    temperature : float
        Initial reservoir temperature [K] (default: 423 K = 150°C)
    compressibility : float
        Total compressibility [Pa⁻¹] (default: 1e-9)
    rock_density : float
        Rock matrix density [kg/m³] (default: 2650)
    rock_heat_capacity : float
        Rock specific heat capacity [J/(kg·K)] (default: 1000)
    """
    permeability: float = 5e-14  # m² (50 mD)
    porosity: float = 0.10
    thickness: float = 300.0  # m
    temperature: float = 423.0  # K (150°C)
    compressibility: float = 1e-9  # Pa⁻¹
    rock_density: float = 2650.0  # kg/m³
    rock_heat_capacity: float = 1000.0  # J/(kg·K)


@dataclass
class WellboreParams:
    """
    Wellbore geometry and properties.
    
    All parameters in SI units.
    
    Attributes
    ----------
    depth : float
        Well depth [m] (default: 3000)
    diameter : float
        Wellbore inner diameter [m] (default: 0.2 = 8 inches)
    roughness : float
        Absolute roughness [m] (default: 0.045e-3 = commercial steel)
    geothermal_gradient : float
        Temperature increase with depth [K/m] (default: 0.03 = 30°C/km)
    """
    depth: float = 3000.0  # m
    diameter: float = 0.2  # m (8 inches)
    roughness: float = 0.045e-3  # m (commercial steel)
    geothermal_gradient: float = 0.03  # K/m (30°C/km)


@dataclass
class FluidParams:
    """
    Fluid properties for sCO₂.
    
    All parameters in SI units. Used as fallback when CoolProp unavailable.
    
    Attributes
    ----------
    injection_temperature : float
        Injection temperature [K] (default: 323 K = 50°C)
    operating_pressure : float
        Operating pressure [Pa] (default: 15 MPa)
    density : float
        Fluid density [kg/m³] (default: 600)
    viscosity : float
        Dynamic viscosity [Pa·s] (default: 5e-5)
    heat_capacity : float
        Specific heat capacity [J/(kg·K)] (default: 1200)
    """
    injection_temperature: float = 323.0  # K (50°C)
    operating_pressure: float = 15e6  # Pa (15 MPa)
    density: float = 600.0  # kg/m³
    viscosity: float = 5e-5  # Pa·s
    heat_capacity: float = 1200.0  # J/(kg·K)


@dataclass
class OptimizationParams:
    """
    Parameters for well layout optimization.
    
    Attributes
    ----------
    n_producers : int
        Number of producer wells (default: 5)
    n_injectors : int
        Number of injector wells (default: 3)
    min_spacing : float
        Minimum inter-well spacing [m] (default: 500)
    max_pressure_drop : float
        Maximum allowable pressure drop [Pa] (default: 5 MPa)
    weights : Tuple[float, float, float]
        Objective function weights (breakthrough, pressure, spacing)
        (default: (0.5, 0.3, 0.2))
    field_radius : float
        Maximum field radius [m] (default: 2000)
    """
    n_producers: int = 5
    n_injectors: int = 3
    min_spacing: float = 500.0  # m
    max_pressure_drop: float = 5e6  # Pa (5 MPa)
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    field_radius: float = 2000.0  # m


# Default instances for easy import
DEFAULT_RESERVOIR = ReservoirParams()
DEFAULT_WELLBORE = WellboreParams()
DEFAULT_FLUID = FluidParams()
DEFAULT_OPTIMIZATION = OptimizationParams()
