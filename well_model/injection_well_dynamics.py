# models/production_well/injection_well_dynamics.py

import numpy as np

from .fluid_properties import get_property_h
from .fluid_dynamics import pressure_drop_friction, friction_factor

g = 9.81

def injection_model(mass_flow: float, h_in: float, depth: float, P_in: float,
                    fluid: str = 'CO2', D: float = 0.27, L_segment: float = 100, roughness: float = 55e-6,
                    debug: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This module provides core functions for simulating thermodynamic and hydraulic evolution
    of a working fluid (e.g., CO2) in a geothermal injection well. Based on the production well model,
    but the iteration is reversed from 0 to depth.

    The function solves for temperature, pressure, and enthalpy profiles along the well depth using a
    finite volume approach with a constant mass flow rate. The simulation accounts for:
    - Gravitational pressure loss
    - Frictional pressure loss based on the Darcy-Weisbach equation
    - Enthalpy and temperature evolution based on the fluid's thermodynamic properties

    Parameters
    ----------
    mass_flow : float
        Mass flow rate [kg/s].
    h_in : float
        Injection well inlet enthalpy [J/kg].
    P_in : float
        Injection well inlet pressure [Pa].
    depth : float
        Total well depth [m].
    fluid : str, optional
        Working fluid type (default is 'CO2').
    D : float, optional
        Pipe diameter [m] (default is 0.27).
    L_segment : float, optional
        Vertical discretization segment height [m] (default is 100).
    roughness : float, optional
        Pipe roughness [m] (default is 55e-6).
    debug : bool, optional
        If True, prints debug information at each segment (default is False).

    References:
        - Fleming et al. (2020): "Thermodynamic modeling of CO2-based geothermal systems".

    Returns
    -------
    T : np.ndarray
        Temperature profile [K] along the well.
    P : np.ndarray
        Pressure profile [Pa] along the well.
    h : np.ndarray
        Enthalpy profile [J] along the well.

    Notes
    -----
    - Assumes single-phase, incompressible flow with constant mass flow.
    - Thermodynamic changes are modeled via enthalpy evolution.
    - Momentum balance includes gravitational and frictional losses.
    """
    # --- Input validation ---
    assert h_in > 0, "Inlet enthalpy must be positive [J/kg]"
    assert P_in > 0, "Inlet pressure must be positive [m]"
    assert mass_flow > 0, "Mass flow must be positive [kg/s]"
    assert depth > 0, "Well depth must be positive [m]"
    assert L_segment > 0, "Segment height must be positive [m]"
    assert D > 0, "Pipe diameter must be positive [m]"


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
    h[0], P[0] = h_in, P_in
    T[0], rho[0], mu[0] = get_property_h(h_in, P_in, fluid)
    v[0] = mass_flow / (A_pipe * rho[0])

    # Solve for fluid properties at each segment
    for i in range(n_segments):
        # Fluid and flow properties at segment i
        Re = rho[i] * v[i] * D / mu[i]  # Reynolds number
        f = friction_factor(Re, D, roughness)

        # Momentum balance (from: Eq 6 + 8, Fleming2020)
        dp_friction = pressure_drop_friction(mass_flow, rho[i], D, L_segment, f)
        dp_gravity = rho[i] * g * L_segment  # Gravitational pressure loss
        P[i + 1] = P[i] - dp_friction + dp_gravity  # Total pressure loss

        # Energy balance (from: Eq 3, Fleming2020)
        h[i + 1] = h[i] + g * L_segment

        # Update fluid properties for the next segment
        T[i + 1], rho[i + 1], mu[i + 1] = get_property_h(h[i + 1], P[i + 1], fluid)
        v[i + 1] = mass_flow / (A_pipe * rho[i + 1])

        # --- Debugging output ---
        if debug:
            print(f"[Segment {i+1}] P={P[i+1]/1e5:.2f} bar, T={T[i+1]-273.15:.2f} °C, Rho={rho[i+1]:.2f}, v={v[i+1]:.2f} m/s")

    return T, P, h



class Well_Injection:
    """
    Interface class for solving the thermodynamic evolution of CO₂ during descent
    through a geothermal injection well.

    Based on inverted Dry_CO2 method from the production well model.

    Attributes:
        m_total (float): Total mass flow rate [kg/s].
        h_in (float): Inlet enthalpy [J/kg].
        P_in (float): Inlet pressure [K].
        depth (float): Total well depth [m].
        D (float): Inner diameter of the production well [m] (default 0.27 m).
        L_segment (float): Vertical discretization length [m] (default 100 m).
        roughness (float): Pipe wall roughness [m] (default 55e-6 m).

    Usage Example:
        >>> from GeoCaP import Well_Injection
        >>> well = Well_Injection(m_total=100, T_in=273, P_in=10, depth=3000, D=0.41)
        >>> results = well.solve()

        The `results` will contain the computed temperature, pressure and enthalpy profiles.
    """
    def __init__(self, m_total, P_in, h_in, depth, D=0.27, L_segment=100, roughness=55e-6):
        """
        Initializes the well injection dynamics parameters.

        Args:
            m_total (float): Total mass flow rate [kg/s].
            h_in (float): Inlet enthalpy [J/kg].
            P_in (float): Inlet pressure [K].
            depth (float): Total well depth [m].
            D (float): Inner diameter of the production well [m] (default 0.27 m).
            L_segment (float): Vertical discretization length [m] (default 100 m).
            roughness (float): Pipe wall roughness [m] (default 55e-6 m).
        """
        self.m_total = m_total
        self.P_in = P_in
        self.h_in = h_in
        self.depth = depth
        self.D = D
        self.L_segment = L_segment
        self.roughness = roughness


    def solve(self, debug=False):
        """
        Solves the fluid dynamics along the geothermal injection well with only dry CO2.

        Args:
            debug (bool, optional): If True, prints additional information at each step. Defaults to False.

        Returns:
            dict: Dictionary containing results such as:
                - Temperature profile [K],
                - Pressure profile [Pa],
                - Enthalpy profile [J/kg].
        """
        res = injection_model(mass_flow=self.m_total, h_in=self.h_in, P_in=self.P_in, depth=self.depth,
                                D=self.D, L_segment=self.L_segment, roughness=self.roughness, debug=debug)

        return res




