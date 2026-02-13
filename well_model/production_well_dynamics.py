# models/production_well/dry_CO2.py
"""
This module provides core functions for simulating thermodynamic and hydraulic evolution
of a working fluid (e.g., CO2) in a geothermal production well.

It implements the finite volume discretization approach and includes:
- Darcy-Weisbach pressure loss
- Enthalpy and temperature evolution
- Reynolds number-based friction factor

References:
- Fleming et al. (2020): "Thermodynamic modeling of CO2-based geothermal systems"
"""

import numpy as np
from .fluid_properties import get_property_h, get_props_T
from .fluid_dynamics import pressure_drop_friction, friction_factor

g = 9.81

def dry_CO2_model(mass_flow: float, inlet_pressure: float, inlet_temperature: float, depth: float,
                    fluid: str = 'CO2', D: float = 0.27, L_segment: float = 100, roughness: float = 55e-6,
                    debug: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Simulates vertical fluid evolution in a geothermal well using finite volume method assuming only dried CO2.

    The function solves for temperature, pressure, and enthalpy profiles along the well depth using a
    finite volume approach with a constant mass flow rate. The simulation accounts for:
    - Gravitational pressure loss
    - Frictional pressure loss based on the Darcy-Weisbach equation
    - Enthalpy and temperature evolution based on the fluid's thermodynamic properties

    Parameters
    ----------
    mass_flow : float
        Mass flow rate [kg/s].
    inlet_pressure : float
        Inlet pressure at the bottom of the well [Pa].
    inlet_temperature : float
        Inlet temperature at the bottom of the well [K].
    depth : float
        Total vertical depth of the well [m].
    fluid : str, optional
        Working fluid type (default is 'CO2').
    D : float, optional
        Pipe diameter [m] (default is 0.27).
    L_segment : float, optional
        Height of each vertical segment used for discretization [m] (default is 100).
    roughness : float, optional
        Pipe roughness [m] (default is 55e-6).
    debug : bool, optional
        If True, prints debug information for each segment (default is False).

    Returns
    -------
    T : np.ndarray
        Temperature profile [K] along the well depth, from bottom to top.
    P : np.ndarray
        Pressure profile [Pa] along the well depth, from bottom to top.
    h : np.ndarray
        Enthalpy profile [J/kg] along the well depth, from bottom to top.

    References:
        - Fleming et al. (2020): "Thermodynamic modeling of CO2-based geothermal systems".

    Notes
    -----
    - Assumes single-phase, incompressible flow with constant mass flow.
    - Thermodynamic changes are modeled via enthalpy evolution.
    - Momentum balance includes gravitational and frictional losses.
    """
    # --- Input validation ---
    assert inlet_temperature > 0, "Inlet temperature must be positive [K]"
    assert inlet_pressure > 0, "Inlet pressure must be positive [m]"
    assert mass_flow > 0, "Mass flow must be positive [kg/s]"
    assert depth > 0, "Well depth must be positive [m]"
    assert L_segment > 0, "Segment height must be positive [m]"
    assert D > 0, "Pipe diameter must be positive [m]"
    assert isinstance(fluid, str), "Fluid type must be a string"

    # --- Computation setup ---
    n_segments = int(np.ceil(depth / L_segment))  # Number of segments along the well
    A_pipe = np.pi * D**2 / 4  # Pipe cross-sectional area [m²]

    # Preallocate arrays
    T = np.zeros(n_segments+1)
    P = np.zeros(n_segments+1)
    h = np.zeros(n_segments+1)
    rho = np.zeros(n_segments+1)
    mu = np.zeros(n_segments+1)
    v = np.zeros(n_segments+1)

    # Set initial state
    T[0], P[0] = inlet_temperature, inlet_pressure
    h[0], rho[0], mu[0] = get_props_T(T[0], P[0])

    v[0] = mass_flow / (A_pipe * rho[0])

    # Solve for fluid properties at each segment
    for i in range(n_segments):
        # Fluid and flow properties at segment i
        Re = rho[i] * v[i] * D / mu[i]  # Reynolds number
        f = friction_factor(Re, D, roughness)

        # Momentum balance (from: Eq 6 + 8, Fleming2020)
        dp_friction = pressure_drop_friction(mass_flow, rho[i], D, L_segment, f)
        dp_gravity = rho[i] * g * L_segment  # Gravitational pressure loss
        P[i + 1] = P[i] - dp_friction - dp_gravity  # Total pressure loss

        # Energy balance (from: Eq 3, Fleming2020)
        h[i + 1] = h[i] - g * L_segment

        # Update fluid properties for the next segment
        T[i + 1], rho[i + 1], mu[i + 1] = get_property_h(h[i + 1], P[i + 1], fluid)
        v[i + 1] = mass_flow / (A_pipe * rho[i + 1])

        # --- Debugging output ---
        if debug:
            print(f"[Segment {i+1}] P={P[i+1]/1e5:.2f} bar, T={T[i+1]-273.15:.2f} °C, Re={Re:.2e}, f={f:.4f}, v={v[i+1]:.2f} m/s")

    return T, P, h, 0, 0