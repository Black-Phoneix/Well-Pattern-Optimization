"""
Objective function for well-field optimization.

Primary objective (closure B):
    J = w1*CV_inj + w2*CV_prod + w4*CV_tof - w3*(τ_years / τ_ref) + Penalty

Where:
    - CV_inj: Coefficient of variation of injector pressure drops
    - CV_prod: Coefficient of variation of producer pressure drops
    - CV_tof: Coefficient of variation of breakthrough times
    - τ_years: Thermal lifetime [years]
    - τ_ref: Reference lifetime for normalization [years]
    - Penalty: Smooth quadratic penalties for constraint violations
"""

from typing import Dict, Optional, Tuple
import numpy as np

from .config import Config, DEFAULT_CONFIG
from .geometry import (
    x_to_params,
    compute_all_well_positions,
    compute_constraint_penalties,
)
from .hydraulics import compute_cv_pressure, compute_pressure_drops
from .thermal import compute_thermal_lifetime
from .breakthrough import compute_tof_simple


def compute_objective(
    x: np.ndarray,
    config: Optional[Config] = None,
    use_simple_tof: bool = True,
    return_components: bool = False,
) -> float:
    """
    Compute objective function value for optimization.
    
    Objective:
        J = w1*CV_inj + w2*CV_prod + w4*CV_tof - w3*(τ_years / τ_ref) + Penalty
    
    Parameters
    ----------
    x : np.ndarray
        Optimization variables [R_in, R_out, θ0, ε1, ε2, ε3]
    config : Config, optional
        Configuration object
    use_simple_tof : bool, optional
        Use simplified TOF calculation (faster, default: True)
    return_components : bool, optional
        If True, return dict with component breakdown
    
    Returns
    -------
    float or dict
        Objective value J, or dict with components if return_components=True
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Unpack variables
    R_in, R_out, theta0, eps1, eps2, eps3 = x_to_params(x)
    
    # Compute well positions
    _, inj_pos, prod_pos = compute_all_well_positions(
        R_in, R_out, theta0, eps1, eps2, eps3
    )
    
    # Compute constraint penalties
    penalty = compute_constraint_penalties(R_in, R_out, eps1, eps2, eps3, config)
    
    # If major constraint violation, return penalty only (skip expensive calculations)
    if penalty > config.PENALTY_LAMBDA * 100:
        if return_components:
            return {
                'J': penalty,
                'CV_inj': np.nan,
                'CV_prod': np.nan,
                'CV_tof': np.nan,
                'tau_years': np.nan,
                'penalty': penalty,
            }
        return penalty
    
    try:
        # Compute pressure uniformity
        CV_inj, CV_prod = compute_cv_pressure(inj_pos, prod_pos, config)
        
        # Compute thermal lifetime
        tau_years = compute_thermal_lifetime(R_in, R_out, config)
        
        # Compute TOF uniformity
        if use_simple_tof:
            _, CV_tof = compute_tof_simple(inj_pos, prod_pos, config)
        else:
            from .breakthrough import compute_tof_proxy
            _, CV_tof = compute_tof_proxy(inj_pos, prod_pos, config)
        
    except Exception as e:
        # If any calculation fails, return large penalty
        if return_components:
            return {
                'J': config.PENALTY_LAMBDA,
                'CV_inj': np.nan,
                'CV_prod': np.nan,
                'CV_tof': np.nan,
                'tau_years': np.nan,
                'penalty': config.PENALTY_LAMBDA,
                'error': str(e),
            }
        return config.PENALTY_LAMBDA
    
    # Objective function
    # J = w1*CV_inj + w2*CV_prod + w4*CV_tof - w3*(τ_years / τ_ref) + Penalty
    w1 = config.W1
    w2 = config.W2
    w3 = config.W3
    w4 = config.W4
    tau_ref = config.TAU_REF
    
    J = (
        w1 * CV_inj +
        w2 * CV_prod +
        w4 * CV_tof -
        w3 * (tau_years / tau_ref) +
        penalty
    )
    
    if return_components:
        return {
            'J': float(J),
            'CV_inj': float(CV_inj),
            'CV_prod': float(CV_prod),
            'CV_tof': float(CV_tof),
            'tau_years': float(tau_years),
            'tau_normalized': float(tau_years / tau_ref),
            'penalty': float(penalty),
            'w1_term': float(w1 * CV_inj),
            'w2_term': float(w2 * CV_prod),
            'w4_term': float(w4 * CV_tof),
            'w3_term': float(-w3 * (tau_years / tau_ref)),
        }
    
    return float(J)


def evaluate_solution(
    x: np.ndarray,
    config: Optional[Config] = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a solution.
    
    Parameters
    ----------
    x : np.ndarray
        Optimization variables [R_in, R_out, θ0, ε1, ε2, ε3]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    dict
        Dictionary with all metrics:
        - Objective components (J, CV_inj, CV_prod, CV_tof, tau_years, penalty)
        - Geometry (R_in, R_out, theta0, eps1, eps2, eps3, radial_gap, min_spacing)
        - Thermal (V_res, Q_res, Q_dot_MW)
        - Pressure (dp_inj, dp_prod)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Unpack variables
    R_in, R_out, theta0, eps1, eps2, eps3 = x_to_params(x)
    
    # Get objective components
    obj_components = compute_objective(x, config, return_components=True)
    
    # Compute well positions
    all_pos, inj_pos, prod_pos = compute_all_well_positions(
        R_in, R_out, theta0, eps1, eps2, eps3
    )
    
    # Geometry metrics
    from .geometry import compute_minimum_spacing, check_geometry_constraints
    min_spacing = compute_minimum_spacing(all_pos)
    _, violations, geo_metrics = check_geometry_constraints(R_in, R_out, eps1, eps2, eps3, config)
    
    # Thermal metrics
    from .thermal import get_thermal_metrics
    try:
        thermal = get_thermal_metrics(R_in, R_out, config)
    except Exception:
        thermal = {}
    
    # Pressure drops
    try:
        dp_inj, dp_prod = compute_pressure_drops(inj_pos, prod_pos, config)
    except Exception:
        dp_inj, dp_prod = np.array([np.nan] * 3), np.array([np.nan] * 5)
    
    # Combine all results
    eps4 = -(eps1 + eps2 + eps3)
    
    result = {
        # Optimization variables
        'R_in': R_in,
        'R_out': R_out,
        'theta0': theta0,
        'theta0_deg': np.rad2deg(theta0),
        'eps1': eps1,
        'eps2': eps2,
        'eps3': eps3,
        'eps4': eps4,
        'eps1_deg': np.rad2deg(eps1),
        'eps2_deg': np.rad2deg(eps2),
        'eps3_deg': np.rad2deg(eps3),
        'eps4_deg': np.rad2deg(eps4),
        
        # Geometry
        'radial_gap': R_out - R_in,
        'min_spacing': min_spacing,
        'n_violations': len(violations),
        'violations': violations,
        
        # Pressure
        'dp_inj': dp_inj,
        'dp_prod': dp_prod,
        'dp_inj_MPa': dp_inj / 1e6,
        'dp_prod_MPa': dp_prod / 1e6,
    }
    
    # Add objective components
    result.update(obj_components)
    
    # Add thermal metrics
    for key, value in thermal.items():
        result[f'thermal_{key}'] = value
    
    return result


def print_solution_summary(
    x: np.ndarray,
    config: Optional[Config] = None,
):
    """
    Print formatted summary of solution.
    
    Parameters
    ----------
    x : np.ndarray
        Optimization variables
    config : Config, optional
        Configuration object
    """
    metrics = evaluate_solution(x, config)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULT SUMMARY")
    print("=" * 60)
    
    print("\n--- Optimized Variables ---")
    print(f"  R_in  = {metrics['R_in']:.1f} m")
    print(f"  R_out = {metrics['R_out']:.1f} m")
    print(f"  θ0    = {metrics['theta0_deg']:.1f}°")
    print(f"  ε1    = {metrics['eps1_deg']:.2f}°")
    print(f"  ε2    = {metrics['eps2_deg']:.2f}°")
    print(f"  ε3    = {metrics['eps3_deg']:.2f}°")
    print(f"  (ε4   = {metrics['eps4_deg']:.2f}° - derived)")
    
    print("\n--- Constraint Status ---")
    print(f"  Radial gap:    {metrics['radial_gap']:.1f} m (min: 500 m)")
    print(f"  Min spacing:   {metrics['min_spacing']:.1f} m (min: 500 m)")
    if metrics['n_violations'] == 0:
        print("  ✓ All constraints satisfied")
    else:
        print(f"  ✗ {metrics['n_violations']} constraint violation(s):")
        for v in metrics['violations']:
            print(f"    - {v}")
    
    print("\n--- Objective Breakdown ---")
    print(f"  Total J    = {metrics['J']:.4f}")
    print(f"  CV_inj     = {metrics['CV_inj']:.4f}")
    print(f"  CV_prod    = {metrics['CV_prod']:.4f}")
    print(f"  CV_tof     = {metrics['CV_tof']:.4f}")
    print(f"  τ_years    = {metrics['tau_years']:.1f} years")
    print(f"  Penalty    = {metrics['penalty']:.2e}")
    
    if 'thermal_Q_dot_MW' in metrics:
        print("\n--- Thermal Metrics ---")
        print(f"  V_res      = {metrics['thermal_V_res']/1e9:.2f} × 10⁹ m³")
        print(f"  Q_res      = {metrics['thermal_Q_res_GJ']:.1f} GJ")
        print(f"  Q̇_CO2     = {metrics['thermal_Q_dot_MW']:.2f} MW")
        print(f"  τ          = {metrics['tau_years']:.1f} years")
    
    print("\n--- Pressure Drops (MPa) ---")
    print("  Injectors:  ", end="")
    for i, dp in enumerate(metrics['dp_inj_MPa'], 1):
        print(f"I{i}={dp:.2f}  ", end="")
    print()
    print("  Producers:  ", end="")
    for i, dp in enumerate(metrics['dp_prod_MPa']):
        label = f"P{i}" if i > 0 else "P0"
        print(f"{label}={dp:.2f}  ", end="")
    print()
    
    print("\n" + "=" * 60)
