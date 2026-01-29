"""
Friction factor calculation for wellbore flow.

This module implements the Colebrook-White equation for turbulent flow
friction factor calculation using Newton-Raphson iteration.

References
----------
Colebrook, C. F. (1939): Turbulent flow in pipes
Moody, L. F. (1944): Friction factors for pipe flow
"""

import numpy as np
from typing import Optional
import warnings


def friction_factor_laminar(Re: float) -> float:
    """
    Calculate friction factor for laminar flow.
    
    For Re < 2300:
        f = 64 / Re
    
    Parameters
    ----------
    Re : float
        Reynolds number [-]
    
    Returns
    -------
    float
        Darcy friction factor [-]
    """
    if Re <= 0:
        raise ValueError("Reynolds number must be positive")
    
    return 64.0 / Re


def friction_factor_turbulent(
    Re: float,
    epsilon: float,
    D: float,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> float:
    """
    Calculate friction factor for turbulent flow using Colebrook-White equation.
    
    Solves the implicit equation using Newton-Raphson:
    
        1/√f = -2*log₁₀(ε/(3.7*D) + 2.51/(Re*√f))
    
    Parameters
    ----------
    Re : float
        Reynolds number [-]
    epsilon : float
        Absolute roughness [m]
    D : float
        Pipe diameter [m]
    max_iter : int, optional
        Maximum iterations for Newton-Raphson (default: 50)
    tol : float, optional
        Convergence tolerance (default: 1e-6)
    
    Returns
    -------
    float
        Darcy friction factor [-]
    
    Raises
    ------
    ValueError
        If Newton-Raphson fails to converge
    
    Notes
    -----
    For initialization, uses Swamee-Jain approximation:
        f ≈ 0.25 / [log₁₀(ε/(3.7*D) + 5.74/Re^0.9)]²
    """
    if Re <= 0:
        raise ValueError("Reynolds number must be positive")
    if D <= 0:
        raise ValueError("Diameter must be positive")
    if epsilon < 0:
        raise ValueError("Roughness must be non-negative")
    
    # Relative roughness
    eps_D = epsilon / D
    
    # For smooth pipes, use Blasius or Prandtl-von Karman equation directly
    if eps_D < 1e-6:  # Effectively smooth
        # Prandtl-von Karman smooth pipe equation (implicit)
        # 1/√f = 2.0*log10(Re*√f) - 0.8
        # Use iterative solution
        f = 0.316 / (Re ** 0.25)  # Initial guess (Blasius)
        for i in range(max_iter):
            sqrt_f = np.sqrt(f)
            f_new = 1.0 / (2.0 * np.log10(Re * sqrt_f) - 0.8) ** 2
            if abs(f_new - f) < tol:
                return float(f_new)
            f = f_new
        return float(f)  # Return last value if didn't converge
    
    # Initial guess using Swamee-Jain approximation
    if eps_D > 1e-10:
        log_term = np.log10(eps_D / 3.7 + 5.74 / (Re ** 0.9))
        f = 0.25 / (log_term ** 2)
    else:
        # Smooth pipe approximation
        f = 0.316 / (Re ** 0.25)
    
    # Newton-Raphson iteration
    for i in range(max_iter):
        # Current value of 1/√f
        inv_sqrt_f = 1.0 / np.sqrt(f)
        
        # Colebrook-White equation: F(f) = 0
        # F(f) = 1/√f + 2*log₁₀(ε/(3.7*D) + 2.51/(Re*√f))
        term1 = eps_D / 3.7
        term2 = 2.51 / (Re * np.sqrt(f))
        F = inv_sqrt_f + 2.0 * np.log10(term1 + term2)
        
        # Derivative dF/df
        # dF/df = -0.5 * f^(-3/2) - (2/(ln(10))) * (2.51/(Re*√f)) * (-0.5*f^(-3/2)) / (term1 + term2)
        dF_df = -0.5 * (f ** (-1.5))
        dF_df -= (2.0 / np.log(10)) * (2.51 / (Re * np.sqrt(f))) * (-0.5 * f ** (-1.5)) / (term1 + term2)
        
        # Newton-Raphson update
        f_new = f - F / dF_df
        
        # Ensure f stays positive
        f_new = max(f_new, 1e-6)
        
        # Check convergence
        if abs(f_new - f) < tol:
            return float(f_new)
        
        f = f_new
    
    # Failed to converge
    warnings.warn(
        f"Newton-Raphson did not converge after {max_iter} iterations. "
        f"Returning last value: f={f:.6f}",
        RuntimeWarning
    )
    return float(f)


def colebrook_white(
    Re: float,
    epsilon: float,
    D: float,
    transition_range: tuple = (2300, 4000),
) -> float:
    """
    Calculate friction factor using appropriate regime.
    
    Automatically selects between laminar, transitional, and turbulent flow.
    
    Parameters
    ----------
    Re : float
        Reynolds number [-]
    epsilon : float
        Absolute roughness [m]
    D : float
        Pipe diameter [m]
    transition_range : tuple of float, optional
        (Re_lower, Re_upper) for transition regime (default: (2300, 4000))
    
    Returns
    -------
    float
        Darcy friction factor [-]
    
    Notes
    -----
    - Re < 2300: Laminar flow (f = 64/Re)
    - 2300 < Re < 4000: Transition regime (linear interpolation)
    - Re > 4000: Turbulent flow (Colebrook-White)
    """
    Re_lower, Re_upper = transition_range
    
    if Re < Re_lower:
        # Laminar flow
        return friction_factor_laminar(Re)
    
    elif Re > Re_upper:
        # Fully turbulent flow
        return friction_factor_turbulent(Re, epsilon, D)
    
    else:
        # Transition regime: interpolate between laminar and turbulent
        f_lam = friction_factor_laminar(Re_lower)
        f_turb = friction_factor_turbulent(Re_upper, epsilon, D)
        
        # Linear interpolation
        alpha = (Re - Re_lower) / (Re_upper - Re_lower)
        return f_lam * (1 - alpha) + f_turb * alpha


def reynolds_number(
    rho: float,
    v: float,
    D: float,
    mu: float,
) -> float:
    """
    Calculate Reynolds number.
    
    Re = ρ*v*D / μ
    
    Parameters
    ----------
    rho : float
        Fluid density [kg/m³]
    v : float
        Flow velocity [m/s]
    D : float
        Pipe diameter [m]
    mu : float
        Dynamic viscosity [Pa·s]
    
    Returns
    -------
    float
        Reynolds number [-]
    """
    if mu <= 0:
        raise ValueError("Viscosity must be positive")
    
    return (rho * v * D) / mu


def pressure_drop_darcy_weisbach(
    f: float,
    L: float,
    D: float,
    rho: float,
    v: float,
) -> float:
    """
    Calculate pressure drop using Darcy-Weisbach equation.
    
    ΔP = f * (L/D) * (ρ*v²/2)
    
    Parameters
    ----------
    f : float
        Darcy friction factor [-]
    L : float
        Pipe length [m]
    D : float
        Pipe diameter [m]
    rho : float
        Fluid density [kg/m³]
    v : float
        Flow velocity [m/s]
    
    Returns
    -------
    float
        Pressure drop [Pa]
    """
    return f * (L / D) * (rho * v**2 / 2.0)


if __name__ == "__main__":
    # Test friction factor calculation
    print("Testing friction factor calculations...")
    
    # Test case 1: Laminar flow
    Re_lam = 1000
    f_lam = colebrook_white(Re_lam, 0.045e-3, 0.2)
    print(f"Laminar (Re={Re_lam}): f = {f_lam:.6f} (expected ≈ {64/Re_lam:.6f})")
    
    # Test case 2: Turbulent flow (smooth pipe)
    Re_turb = 1e5
    f_turb_smooth = colebrook_white(Re_turb, 0.0, 0.2)
    print(f"Turbulent smooth (Re={Re_turb:.0e}): f = {f_turb_smooth:.6f}")
    
    # Test case 3: Turbulent flow (rough pipe)
    epsilon = 0.045e-3  # Commercial steel
    D = 0.2
    f_turb_rough = colebrook_white(Re_turb, epsilon, D)
    print(f"Turbulent rough (Re={Re_turb:.0e}, ε/D={epsilon/D:.2e}): f = {f_turb_rough:.6f}")
    
    # Test case 4: Transition regime
    Re_trans = 3000
    f_trans = colebrook_white(Re_trans, epsilon, D)
    print(f"Transition (Re={Re_trans}): f = {f_trans:.6f}")
    
    print("✓ Friction factor tests complete")
