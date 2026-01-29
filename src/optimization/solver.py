"""
Differential Evolution solver for well layout optimization.

Wraps scipy.optimize.differential_evolution with domain-specific
features for geothermal well field optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
from scipy.optimize import differential_evolution
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from patterns.geometry import Well
from src.optimization.objective_func import compute_total_cost, evaluate_layout_quality
from src.optimization.constraints import apply_all_constraints


def optimize_layout(
    initial_pattern: List[Well],
    reservoir_params: dict,
    fluid_params: dict,
    bounds: Optional[List[Tuple[float, float]]] = None,
    constraints: Optional[dict] = None,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    de_params: Optional[dict] = None,
    callback: Optional[Callable] = None,
) -> Tuple[List[Well], dict]:
    """
    Optimize well layout using Differential Evolution.
    
    Parameters
    ----------
    initial_pattern : list of Well
        Initial well configuration
    reservoir_params : dict
        Reservoir properties
    fluid_params : dict
        Fluid properties
    bounds : list of tuples, optional
        Bounds for each decision variable [(x1_min, x1_max), (y1_min, y1_max), ...]
        If None, uses circular field with radius from constraints
    constraints : dict, optional
        Constraint parameters:
        - 'min_spacing': float [m]
        - 'field_radius': float [m]
        - 'max_pressure_drop': float [Pa]
    weights : tuple of float, optional
        Objective function weights (default: (0.5, 0.3, 0.2))
    de_params : dict, optional
        Differential Evolution parameters:
        - 'popsize': int (default: 15)
        - 'maxiter': int (default: 100)
        - 'mutation': float (default: 0.8)
        - 'recombination': float (default: 0.7)
        - 'tol': float (default: 1e-3)
        - 'workers': int (default: 1)
    callback : callable, optional
        Callback function(xk, convergence) called each iteration
    
    Returns
    -------
    tuple
        (optimized_wells, optimization_info)
        - optimized_wells: list of Well objects
        - optimization_info: dict with optimization statistics
    """
    # Parse initial pattern
    n_injectors = sum(1 for w in initial_pattern if w.kind == 'injector')
    n_producers = sum(1 for w in initial_pattern if w.kind == 'producer')
    n_wells = len(initial_pattern)
    
    # Default constraints
    if constraints is None:
        constraints = {
            'min_spacing': 500.0,
            'field_radius': 2000.0,
            'max_pressure_drop': 5e6,
        }
    
    min_spacing = constraints.get('min_spacing', 500.0)
    field_radius = constraints.get('field_radius', 2000.0)
    
    # Default bounds: circular field
    if bounds is None:
        bounds = [(-field_radius, field_radius) for _ in range(2 * n_wells)]
    
    # Default DE parameters
    if de_params is None:
        de_params = {}
    
    popsize = de_params.get('popsize', 15)
    maxiter = de_params.get('maxiter', 100)
    mutation = de_params.get('mutation', 0.8)
    recombination = de_params.get('recombination', 0.7)
    tol = de_params.get('tol', 1e-3)
    workers = de_params.get('workers', 1)
    
    # Initial guess from pattern
    x0 = np.array([coord for w in initial_pattern for coord in [w.x, w.y]])
    
    # Optimization history
    history = {
        'iteration': [],
        'best_cost': [],
        'population_mean': [],
        'population_std': [],
    }
    
    def callback_wrapper(xk, convergence):
        """Wrapper to track optimization history."""
        cost = objective_function(xk)
        history['iteration'].append(len(history['iteration']))
        history['best_cost'].append(cost)
        
        if callback is not None:
            callback(xk, convergence)
        
        # Print progress
        if len(history['iteration']) % 10 == 0:
            print(f"  Iteration {len(history['iteration'])}: cost = {cost:.6f}")
    
    def objective_function(x):
        """Objective function with constraints."""
        # Check hard constraints
        penalty = apply_all_constraints(
            x, n_injectors, n_producers, min_spacing, field_radius
        )
        
        if penalty > 0:
            return penalty
        
        # Compute actual cost
        try:
            return compute_total_cost(
                x, n_injectors, n_producers,
                reservoir_params, fluid_params,
                weights, min_spacing
            )
        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            # Return penalty for physics calculation failures
            return 1e9
    
    print(f"Starting optimization with {n_injectors} injectors and {n_producers} producers...")
    print(f"DE parameters: popsize={popsize}, maxiter={maxiter}")
    
    # Run optimization
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        tol=tol,
        workers=workers,
        callback=callback_wrapper,
        polish=True,
        init='latinhypercube',
    )
    
    # Parse result
    optimized_x = result.x
    optimized_wells = []
    
    for i in range(n_injectors):
        x, y = optimized_x[2*i], optimized_x[2*i+1]
        optimized_wells.append(Well(x=x, y=y, kind='injector'))
    
    for i in range(n_injectors, n_wells):
        x, y = optimized_x[2*i], optimized_x[2*i+1]
        optimized_wells.append(Well(x=x, y=y, kind='producer'))
    
    # Optimization info
    info = {
        'success': result.success,
        'message': result.message,
        'n_iterations': result.nit,
        'n_evaluations': result.nfev,
        'final_cost': result.fun,
        'initial_cost': objective_function(x0),
        'improvement': (objective_function(x0) - result.fun) / objective_function(x0) * 100,
        'history': history,
    }
    
    print(f"\nOptimization complete:")
    print(f"  Initial cost: {info['initial_cost']:.6f}")
    print(f"  Final cost: {info['final_cost']:.6f}")
    print(f"  Improvement: {info['improvement']:.2f}%")
    print(f"  Iterations: {info['n_iterations']}")
    
    return optimized_wells, info


def run_sensitivity_analysis(
    base_wells: List[Well],
    param_ranges: Dict[str, Tuple[float, float, int]],
    reservoir_params: dict,
    fluid_params: dict,
) -> pd.DataFrame:
    """
    Run sensitivity analysis by varying parameters.
    
    Parameters
    ----------
    base_wells : list of Well
        Base well configuration
    param_ranges : dict
        Dictionary mapping parameter names to (min, max, n_points):
        - 'permeability': (min, max, n) in m²
        - 'spacing': (min, max, n) in m
        - 'flow_rate': (min, max, n) in kg/s
    reservoir_params : dict
        Base reservoir parameters (will be varied)
    fluid_params : dict
        Fluid parameters
    
    Returns
    -------
    pd.DataFrame
        Results with columns for each parameter and objective terms
    """
    results = []
    
    for param_name, (pmin, pmax, n_points) in param_ranges.items():
        param_values = np.linspace(pmin, pmax, n_points)
        
        for val in param_values:
            # Modify parameters
            res_params = reservoir_params.copy()
            fluid_par = fluid_params.copy()
            flow_rate = 50.0
            
            if param_name == 'permeability':
                res_params['permeability'] = val
            elif param_name == 'flow_rate':
                flow_rate = val
            # Add more parameters as needed
            
            # Evaluate layout
            quality = evaluate_layout_quality(base_wells, flow_rate, res_params, fluid_par)
            
            results.append({
                'parameter': param_name,
                'value': val,
                'breakthrough_cv': quality['breakthrough_cv'],
                'pressure_cv': quality['pressure_cv'],
                'total_cost': quality['total_cost'],
            })
    
    return pd.DataFrame(results)


def visualize_convergence(
    optimization_history: dict,
    save_path: Optional[str] = None,
):
    """
    Visualize optimization convergence.
    
    Parameters
    ----------
    optimization_history : dict
        History dictionary from optimize_layout
    save_path : str, optional
        Path to save figure. If None, displays interactively.
    
    Returns
    -------
    matplotlib.figure.Figure
        Convergence plot
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = optimization_history['iteration']
    costs = optimization_history['best_cost']
    
    ax.plot(iterations, costs, 'b-', linewidth=2, label='Best cost')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add improvement annotation
    if len(costs) > 0:
        improvement = (costs[0] - costs[-1]) / costs[0] * 100
        ax.text(
            0.95, 0.95, f'Improvement: {improvement:.1f}%',
            transform=ax.transAxes,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test optimization
    print("Testing optimization solver...")
    
    from patterns.geometry import generate_ring_pattern
    
    # Test parameters
    reservoir_params = {
        'porosity': 0.10,
        'thickness': 300.0,
        'rock_density': 2650.0,
        'rock_heat_capacity': 1000.0,
    }
    
    fluid_params = {
        'density': 600.0,
        'heat_capacity': 1200.0,
    }
    
    # Generate initial pattern
    injectors, producers = generate_ring_pattern(
        n_inj=3, n_prod=5, R_inj=500.0, R_prod=1000.0
    )
    initial_wells = injectors + producers
    
    # Run short optimization (reduced iterations for testing)
    de_params = {'maxiter': 5, 'popsize': 10}
    
    optimized_wells, info = optimize_layout(
        initial_wells,
        reservoir_params,
        fluid_params,
        de_params=de_params,
    )
    
    print(f"\n✓ Optimization test complete")
    print(f"  Moved wells by average: {np.mean([abs(w1.x - w2.x) + abs(w1.y - w2.y) for w1, w2 in zip(initial_wells, optimized_wells)]):.1f} m")
