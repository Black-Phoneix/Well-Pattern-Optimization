"""
Thermal model and lifetime estimation.

Uses a conical frustum reservoir volume model tied to geometry (R_in, R_out).

Key equations:
    Reservoir volume:
        V_res = (1/3) π H (R_out² + R_out*R_in + R_in²)
    
    Mass of rock + residual water:
        m_rock = V_res * (1-φ) * ρ_rock
        m_wat = V_res * φ * ρ_wat * S_wirr
    
    Total available sensible heat:
        Q_res = (m_rock * c_p,rock + m_wat * c_p,wat) * (T_res - T_working)
    
    Heat extraction rate:
        Q̇_CO2 = ṁ_total * c_p,CO2 * (T_prod - T_inj)
    
    Lifetime:
        τ_years = Q_res / (Q̇_CO2 * 3600 * 24 * 365)
"""

from typing import Dict, Optional, Tuple
import numpy as np

from .config import Config, DEFAULT_CONFIG
from .hydraulics import get_mean_fluid_properties


def compute_reservoir_volume(
    R_in: float,
    R_out: float,
    config: Optional[Config] = None,
) -> float:
    """
    Compute conical frustum reservoir volume.
    
    Equation:
        V_res = (1/3) π H (R_out² + R_out*R_in + R_in²)
    
    Parameters
    ----------
    R_in : float
        Inner radius (injector ring) [m]
    R_out : float
        Outer radius (producer ring) [m]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    float
        Reservoir volume [m³]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    H = config.H_THICK
    
    V_res = (1.0 / 3.0) * np.pi * H * (R_out**2 + R_out * R_in + R_in**2)
    
    return float(V_res)


def compute_rock_water_masses(
    R_in: float,
    R_out: float,
    config: Optional[Config] = None,
) -> Tuple[float, float]:
    """
    Compute mass of rock and residual water in the reservoir.
    
    Equations:
        m_rock = V_res * (1 - φ) * ρ_rock
        m_wat = V_res * φ * ρ_wat * S_wirr
    
    Parameters
    ----------
    R_in, R_out : float
        Ring radii [m]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    tuple
        (m_rock, m_wat) - Masses [kg]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    V_res = compute_reservoir_volume(R_in, R_out, config)
    
    phi = config.POROSITY
    rho_rock = config.RHO_ROCK
    rho_wat = config.RHO_WAT
    S_wirr = config.S_WIRR
    
    m_rock = V_res * (1.0 - phi) * rho_rock
    m_wat = V_res * phi * rho_wat * S_wirr
    
    return float(m_rock), float(m_wat)


def compute_total_heat_content(
    R_in: float,
    R_out: float,
    config: Optional[Config] = None,
) -> float:
    """
    Compute total available sensible heat in the reservoir.
    
    Equation:
        Q_res = (m_rock * c_p,rock + m_wat * c_p,wat) * (T_res - T_working)
    
    Parameters
    ----------
    R_in, R_out : float
        Ring radii [m]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    float
        Total heat content [J]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    m_rock, m_wat = compute_rock_water_masses(R_in, R_out, config)
    
    cp_rock = config.CP_ROCK
    cp_wat = config.CP_WAT
    T_res = config.T_RES_K
    T_work = config.T_WORK_K
    
    Q_res = (m_rock * cp_rock + m_wat * cp_wat) * (T_res - T_work)
    
    return float(Q_res)


def compute_heat_extraction_rate(config: Optional[Config] = None) -> float:
    """
    Compute heat extraction rate from circulating CO2.
    
    Equation:
        Q̇_CO2 = ṁ_total * c_p,CO2 * (T_prod - T_inj)
    
    Uses CoolProp for c_p,CO2 at mean conditions.
    
    Parameters
    ----------
    config : Config, optional
        Configuration object
    
    Returns
    -------
    float
        Heat extraction rate [W]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Get CO2 heat capacity from CoolProp
    props = get_mean_fluid_properties(config)
    cp_co2 = props['cp']  # J/(kg·K)
    
    m_dot = config.M_DOT_TOTAL  # kg/s
    T_prod = config.T_PROD_K    # K
    T_inj = config.T_INJ_K      # K
    
    # Ensure positive temperature difference
    dT = T_prod - T_inj
    if dT <= 0:
        raise ValueError(
            f"T_prod ({config.T_PROD_C}°C) must be > T_inj ({config.T_INJ_C}°C) "
            "for positive heat extraction."
        )
    
    Q_dot = m_dot * cp_co2 * dT
    
    return float(Q_dot)


def compute_thermal_lifetime(
    R_in: float,
    R_out: float,
    config: Optional[Config] = None,
) -> float:
    """
    Compute thermal lifetime of the reservoir.
    
    Equation:
        τ_years = Q_res / (Q̇_CO2 * 3600 * 24 * 365)
    
    Parameters
    ----------
    R_in, R_out : float
        Ring radii [m]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    float
        Thermal lifetime [years]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    Q_res = compute_total_heat_content(R_in, R_out, config)  # J
    Q_dot = compute_heat_extraction_rate(config)  # W = J/s
    
    # Seconds per year
    seconds_per_year = 3600.0 * 24.0 * 365.0
    
    tau_years = Q_res / (Q_dot * seconds_per_year)
    
    return float(tau_years)


def get_thermal_metrics(
    R_in: float,
    R_out: float,
    config: Optional[Config] = None,
) -> Dict[str, float]:
    """
    Get comprehensive thermal metrics for reporting.
    
    Parameters
    ----------
    R_in, R_out : float
        Ring radii [m]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    dict
        Dictionary with thermal metrics:
        - 'V_res': Reservoir volume [m³]
        - 'm_rock': Rock mass [kg]
        - 'm_water': Water mass [kg]
        - 'Q_res': Total heat content [J]
        - 'Q_dot': Heat extraction rate [W]
        - 'Q_dot_MW': Heat extraction rate [MW]
        - 'tau_years': Thermal lifetime [years]
        - 'cp_co2': CO2 heat capacity [J/(kg·K)]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    V_res = compute_reservoir_volume(R_in, R_out, config)
    m_rock, m_water = compute_rock_water_masses(R_in, R_out, config)
    Q_res = compute_total_heat_content(R_in, R_out, config)
    Q_dot = compute_heat_extraction_rate(config)
    tau_years = compute_thermal_lifetime(R_in, R_out, config)
    
    props = get_mean_fluid_properties(config)
    
    return {
        'V_res': V_res,
        'm_rock': m_rock,
        'm_water': m_water,
        'Q_res': Q_res,
        'Q_res_GJ': Q_res / 1e9,
        'Q_dot': Q_dot,
        'Q_dot_MW': Q_dot / 1e6,
        'tau_years': tau_years,
        'cp_co2': props['cp'],
    }
