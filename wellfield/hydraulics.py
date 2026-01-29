"""
Hydraulics forward model using logarithmic potential superposition.

This module implements steady Darcy flow in a homogeneous, isotropic reservoir:
- Pressure as superposition of 2D logarithmic potentials of point sources/sinks
- Per-well mass rates fixed by equal allocation (closure B from prompt)
- Analytical velocity field computation

Key equations:
    p(x) = P_ref - F_two_sided * (μ / (2π κ H)) * Σ_j Q_j * ln( max(r_j(x), r_w) / R_b )
    
    q(x) = -(κ/μ) ∇p(x)   [Darcy flux]
    
    ∂/∂x ln r = (x-xj)/r²
    ∂/∂y ln r = (y-yj)/r²

Fluid properties (μ, ρ) computed using CoolProp at mean conditions.
"""

from typing import Dict, Tuple, Optional
import numpy as np

from .config import Config, DEFAULT_CONFIG


# Check for CoolProp availability
try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False


def get_fluid_properties(
    T_K: float,
    P_Pa: float,
    config: Optional[Config] = None,
) -> Dict[str, float]:
    """
    Get CO2 fluid properties at given temperature and pressure using CoolProp.
    
    Parameters
    ----------
    T_K : float
        Temperature [K]
    P_Pa : float
        Pressure [Pa]
    config : Config, optional
        Configuration (used for fallback values if CoolProp fails)
    
    Returns
    -------
    dict
        Dictionary with:
        - 'mu': Dynamic viscosity [Pa·s]
        - 'rho': Density [kg/m³]
        - 'cp': Specific heat capacity [J/(kg·K)]
    
    Raises
    ------
    ImportError
        If CoolProp is not installed and no fallback is available
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if not COOLPROP_AVAILABLE:
        raise ImportError(
            "CoolProp is required for fluid property calculations.\n"
            "Install with: pip install CoolProp\n"
            "Or: conda install -c conda-forge coolprop"
        )
    
    try:
        mu = PropsSI('V', 'T', T_K, 'P', P_Pa, 'CO2')      # Viscosity [Pa·s]
        rho = PropsSI('D', 'T', T_K, 'P', P_Pa, 'CO2')     # Density [kg/m³]
        cp = PropsSI('C', 'T', T_K, 'P', P_Pa, 'CO2')      # Heat capacity [J/(kg·K)]
    except Exception as e:
        raise RuntimeError(
            f"CoolProp property calculation failed at T={T_K:.1f} K, P={P_Pa/1e6:.2f} MPa: {e}"
        )
    
    return {'mu': mu, 'rho': rho, 'cp': cp}


def get_mean_fluid_properties(config: Optional[Config] = None) -> Dict[str, float]:
    """
    Get fluid properties at mean operating conditions.
    
    Uses: P_mean = 0.5*(P_inj + P_prod), T_mean = 0.5*(T_inj + T_prod)
    
    Parameters
    ----------
    config : Config, optional
        Configuration object
    
    Returns
    -------
    dict
        Fluid properties at mean conditions
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    return get_fluid_properties(config.T_MEAN_K, config.P_MEAN, config)


def compute_volumetric_flow_rates(config: Optional[Config] = None) -> Dict[str, float]:
    """
    Compute volumetric flow rates for each well type.
    
    Conversion from mass rate to volumetric rate:
        Q_j [m³/s] = sign_j * ṁ_j / ρ
    
    Injection has positive Q (source), production has negative Q (sink).
    
    Parameters
    ----------
    config : Config, optional
        Configuration object
    
    Returns
    -------
    dict
        - 'Q_inj': Volumetric rate per injector [m³/s] (positive)
        - 'Q_prod': Volumetric rate per producer [m³/s] (negative)
        - 'Q_center': Volumetric rate for center producer [m³/s] (negative)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    props = get_mean_fluid_properties(config)
    rho = props['rho']
    
    # Mass rates
    m_inj = config.m_dot_inj_each   # kg/s per injector
    m_prod = config.m_dot_prod_each  # kg/s per producer
    
    # Volumetric rates (positive for injection, negative for production)
    Q_inj = m_inj / rho    # Positive (source)
    Q_prod = -m_prod / rho  # Negative (sink)
    
    return {
        'Q_inj': Q_inj,
        'Q_prod': Q_prod,
        'Q_center': Q_prod,  # Center producer same rate as outer producers
    }


def compute_pressure_field(
    X: np.ndarray,
    Y: np.ndarray,
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
    P_ref: float = 0.0,
) -> np.ndarray:
    """
    Compute pressure field on a 2D grid using logarithmic potential superposition.
    
    Equation:
        p(x) = P_ref - F_two_sided * (μ / (2π κ H)) * Σ_j Q_j * ln( max(r_j(x), r_w) / R_b )
    
    Parameters
    ----------
    X, Y : np.ndarray
        2D mesh grids of x and y coordinates [m]
    inj_pos : np.ndarray
        Injector positions as (N_inj, 2) array [m]
    prod_pos : np.ndarray
        Producer positions as (N_prod, 2) array [m]
    config : Config, optional
        Configuration object
    P_ref : float, optional
        Reference pressure [Pa] (default: 0)
    
    Returns
    -------
    np.ndarray
        Pressure field p(x,y) [Pa] with same shape as X
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Get fluid properties
    props = get_mean_fluid_properties(config)
    mu = props['mu']
    
    # Get flow rates
    Q_rates = compute_volumetric_flow_rates(config)
    
    # Physical parameters
    kappa = config.K_PERM    # Permeability [m²]
    H = config.H_THICK       # Thickness [m]
    r_w = config.R_WELL      # Well radius [m]
    R_b = config.R_B         # Reference radius [m]
    F = config.F_TWO_SIDED   # Two-sided factor
    
    # Coefficient: F * μ / (2π κ H)
    coeff = F * mu / (2.0 * np.pi * kappa * H)
    
    # Initialize pressure field
    p = np.full_like(X, P_ref, dtype=float)
    
    # Add contribution from each injector
    for i in range(len(inj_pos)):
        xj, yj = inj_pos[i]
        Q_j = Q_rates['Q_inj']  # Positive (source)
        
        # Distance from grid points to well
        r = np.sqrt((X - xj)**2 + (Y - yj)**2)
        r = np.maximum(r, r_w)  # Cutoff at well radius
        
        # Pressure contribution: -coeff * Q * ln(r / R_b)
        p -= coeff * Q_j * np.log(r / R_b)
    
    # Add contribution from center producer (P0)
    xj, yj = prod_pos[0]
    Q_j = Q_rates['Q_center']  # Negative (sink)
    r = np.sqrt((X - xj)**2 + (Y - yj)**2)
    r = np.maximum(r, r_w)
    p -= coeff * Q_j * np.log(r / R_b)
    
    # Add contribution from outer producers (P1-P4)
    for i in range(1, len(prod_pos)):
        xj, yj = prod_pos[i]
        Q_j = Q_rates['Q_prod']  # Negative (sink)
        
        r = np.sqrt((X - xj)**2 + (Y - yj)**2)
        r = np.maximum(r, r_w)
        p -= coeff * Q_j * np.log(r / R_b)
    
    return p


def compute_pressure_at_wells(
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
    P_ref: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pressure at each well location (bottom-hole pressure proxy).
    
    For self-distance, use r_j = r_w (well radius).
    
    Parameters
    ----------
    inj_pos : np.ndarray
        Injector positions as (N_inj, 2) array [m]
    prod_pos : np.ndarray
        Producer positions as (N_prod, 2) array [m]
    config : Config, optional
        Configuration object
    P_ref : float, optional
        Reference pressure [Pa]
    
    Returns
    -------
    tuple
        (p_inj, p_prod) - Pressure at injectors and producers [Pa]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    props = get_mean_fluid_properties(config)
    mu = props['mu']
    
    Q_rates = compute_volumetric_flow_rates(config)
    
    kappa = config.K_PERM
    H = config.H_THICK
    r_w = config.R_WELL
    R_b = config.R_B
    F = config.F_TWO_SIDED
    
    coeff = F * mu / (2.0 * np.pi * kappa * H)
    
    # All wells with their flow rates
    n_inj = len(inj_pos)
    n_prod = len(prod_pos)
    
    all_pos = np.vstack([inj_pos, prod_pos])
    all_Q = np.array(
        [Q_rates['Q_inj']] * n_inj +
        [Q_rates['Q_center']] +  # Center producer
        [Q_rates['Q_prod']] * (n_prod - 1)  # Outer producers
    )
    
    n_wells = len(all_pos)
    p_wells = np.full(n_wells, P_ref, dtype=float)
    
    # Compute pressure at each well from all sources
    for i in range(n_wells):
        xi, yi = all_pos[i]
        
        for j in range(n_wells):
            xj, yj = all_pos[j]
            Q_j = all_Q[j]
            
            # Distance (use r_w for self-interaction)
            if i == j:
                r = r_w
            else:
                r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                r = max(r, r_w)
            
            p_wells[i] -= coeff * Q_j * np.log(r / R_b)
    
    p_inj = p_wells[:n_inj]
    p_prod = p_wells[n_inj:]
    
    return p_inj, p_prod


def compute_pressure_drops(
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
    P_ref: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pressure drops Δp_i = P_ref - p(x_i) for each well.
    
    Parameters
    ----------
    inj_pos, prod_pos : np.ndarray
        Well positions
    config : Config, optional
        Configuration object
    P_ref : float, optional
        Reference pressure [Pa]
    
    Returns
    -------
    tuple
        (dp_inj, dp_prod) - Pressure drops at injectors and producers [Pa]
    """
    p_inj, p_prod = compute_pressure_at_wells(inj_pos, prod_pos, config, P_ref)
    
    dp_inj = P_ref - p_inj
    dp_prod = P_ref - p_prod
    
    return dp_inj, dp_prod


def compute_cv_pressure(
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
) -> Tuple[float, float]:
    """
    Compute coefficient of variation of pressure drops for injectors and producers.
    
    CV = std(Δp) / mean(Δp)
    
    Parameters
    ----------
    inj_pos, prod_pos : np.ndarray
        Well positions
    config : Config, optional
        Configuration object
    
    Returns
    -------
    tuple
        (CV_inj, CV_prod)
    """
    dp_inj, dp_prod = compute_pressure_drops(inj_pos, prod_pos, config)
    
    # Handle sign: use absolute values for CV calculation
    dp_inj = np.abs(dp_inj)
    dp_prod = np.abs(dp_prod)
    
    mean_inj = np.mean(dp_inj)
    mean_prod = np.mean(dp_prod)
    
    # Avoid division by zero
    if mean_inj > 1e-10:
        cv_inj = np.std(dp_inj) / mean_inj
    else:
        cv_inj = 0.0
    
    if mean_prod > 1e-10:
        cv_prod = np.std(dp_prod) / mean_prod
    else:
        cv_prod = 0.0
    
    return float(cv_inj), float(cv_prod)


def compute_velocity_field(
    X: np.ndarray,
    Y: np.ndarray,
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Darcy velocity (flux) field analytically.
    
    Equations:
        q(x) = -(κ/μ) ∇p(x)
        
        For logarithmic potential from well j:
        ∂/∂x ln(r) = (x - xj) / r²
        ∂/∂y ln(r) = (y - yj) / r²
    
    Parameters
    ----------
    X, Y : np.ndarray
        2D mesh grids of coordinates [m]
    inj_pos : np.ndarray
        Injector positions as (N_inj, 2) array [m]
    prod_pos : np.ndarray
        Producer positions as (N_prod, 2) array [m]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    tuple
        (qx, qy) - Darcy flux components [m/s]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    props = get_mean_fluid_properties(config)
    mu = props['mu']
    
    Q_rates = compute_volumetric_flow_rates(config)
    
    kappa = config.K_PERM
    H = config.H_THICK
    r_w = config.R_WELL
    F = config.F_TWO_SIDED
    
    # Coefficient for ∇p: F * μ / (2π κ H)
    coeff_p = F * mu / (2.0 * np.pi * kappa * H)
    
    # Coefficient for q = -(κ/μ) ∇p
    # ∇p = -coeff_p * Σ Q_j * (x-xj)/r² (for x-component)
    # q_x = -(κ/μ) * (-coeff_p) * Σ Q_j * (x-xj)/r² = (κ/μ) * coeff_p * Σ Q_j * (x-xj)/r²
    # = (F / (2π H)) * Σ Q_j * (x-xj)/r²
    coeff_q = F / (2.0 * np.pi * H)
    
    # Initialize velocity components
    qx = np.zeros_like(X, dtype=float)
    qy = np.zeros_like(Y, dtype=float)
    
    # All wells with their flow rates
    n_inj = len(inj_pos)
    n_prod = len(prod_pos)
    
    all_pos = np.vstack([inj_pos, prod_pos])
    all_Q = np.array(
        [Q_rates['Q_inj']] * n_inj +
        [Q_rates['Q_center']] +
        [Q_rates['Q_prod']] * (n_prod - 1)
    )
    
    for j in range(len(all_pos)):
        xj, yj = all_pos[j]
        Q_j = all_Q[j]
        
        # Distance squared
        dx = X - xj
        dy = Y - yj
        r_sq = dx**2 + dy**2
        r_sq = np.maximum(r_sq, r_w**2)  # Cutoff
        
        # Velocity contribution
        qx += coeff_q * Q_j * dx / r_sq
        qy += coeff_q * Q_j * dy / r_sq
    
    return qx, qy


def compute_pore_velocity_field(
    X: np.ndarray,
    Y: np.ndarray,
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pore velocity field.
    
    v_p(x) = q(x) / φ
    
    Parameters
    ----------
    X, Y : np.ndarray
        2D mesh grids
    inj_pos, prod_pos : np.ndarray
        Well positions
    config : Config, optional
        Configuration object
    
    Returns
    -------
    tuple
        (vx, vy) - Pore velocity components [m/s]
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    qx, qy = compute_velocity_field(X, Y, inj_pos, prod_pos, config)
    
    phi = config.POROSITY
    vx = qx / phi
    vy = qy / phi
    
    return vx, vy
