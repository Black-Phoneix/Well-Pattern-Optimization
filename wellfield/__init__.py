"""
CO2-based closed-loop geothermal well-field optimization package.

This package provides modules for:
- config: Centralized configuration with all constants
- geometry: Well coordinates, constraints, distance matrix
- hydraulics: Pressure potential, Δp at wells, pressure/velocity field
- thermal: Frustum heat content, extraction rate, lifetime τ
- breakthrough: Streamline plotting, time-of-flight (TOF) proxy
- objective: Objective function J with penalties
- optimize: SciPy differential_evolution wrapper
- plots: Publication-style visualization
"""

from .config import Config
from .geometry import (
    compute_well_coordinates,
    compute_all_well_positions,
    compute_distance_matrix,
    compute_minimum_spacing,
)
from .hydraulics import (
    compute_pressure_field,
    compute_pressure_at_wells,
    compute_velocity_field,
    get_fluid_properties,
)
from .thermal import (
    compute_reservoir_volume,
    compute_thermal_lifetime,
    compute_heat_extraction_rate,
)
from .breakthrough import (
    compute_streamlines,
    compute_tof_proxy,
)
from .objective import compute_objective
from .optimize import run_optimization
from . import plots

__version__ = "1.0.0"
__all__ = [
    "Config",
    "compute_well_coordinates",
    "compute_all_well_positions",
    "compute_distance_matrix",
    "compute_minimum_spacing",
    "compute_pressure_field",
    "compute_pressure_at_wells",
    "compute_velocity_field",
    "get_fluid_properties",
    "compute_reservoir_volume",
    "compute_thermal_lifetime",
    "compute_heat_extraction_rate",
    "compute_streamlines",
    "compute_tof_proxy",
    "compute_objective",
    "run_optimization",
    "plots",
]
