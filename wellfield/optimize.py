"""
Differential Evolution optimizer wrapper for well-field optimization.

Uses scipy.optimize.differential_evolution with:
- Bounds from config
- Fixed random seed for reproducibility
- Configurable popsize and maxiter
- Optional parallel workers
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import differential_evolution

from .config import Config, DEFAULT_CONFIG
from .geometry import x_to_params, compute_well_coordinates, get_default_initial_guess
from .objective import compute_objective, evaluate_solution, print_solution_summary


def run_optimization(
    config: Optional[Config] = None,
    callback: Optional[Callable] = None,
    verbose: bool = True,
    use_simple_tof: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Run well-field layout optimization using Differential Evolution.
    
    Parameters
    ----------
    config : Config, optional
        Configuration object with bounds, weights, and DE parameters
    callback : callable, optional
        Callback function(xk, convergence) called each iteration
    verbose : bool, optional
        Print progress messages (default: True)
    use_simple_tof : bool, optional
        Use simplified TOF calculation (default: True, faster)
    
    Returns
    -------
    tuple
        (x_best, info)
        - x_best: Best optimization variables [R_in, R_out, θ0, ε1, ε2, ε3]
        - info: Dictionary with optimization information
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Get bounds
    bounds = config.get_bounds()
    
    # Optimization history
    history = {
        'iterations': [],
        'best_J': [],
        'convergence': [],
    }
    
    iteration_count = [0]  # Use list for mutable closure
    
    def callback_wrapper(xk: np.ndarray, convergence: float = 0.0) -> bool:
        """Track optimization progress."""
        iteration_count[0] += 1
        J = compute_objective(xk, config, use_simple_tof)
        
        history['iterations'].append(iteration_count[0])
        history['best_J'].append(J)
        history['convergence'].append(convergence)
        
        if verbose and iteration_count[0] % 10 == 0:
            print(f"  Iteration {iteration_count[0]}: J = {J:.4f}")
        
        if callback is not None:
            callback(xk, convergence)
        
        return False  # Don't stop
    
    def objective_wrapper(x: np.ndarray) -> float:
        """Wrapper for objective function."""
        return compute_objective(x, config, use_simple_tof)
    
    if verbose:
        print("\n" + "=" * 60)
        print("WELL-FIELD OPTIMIZATION")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Population size: {config.DE_POPSIZE}")
        print(f"  Max iterations:  {config.DE_MAXITER}")
        print(f"  Random seed:     {config.DE_SEED}")
        print(f"  Weights: w1={config.W1}, w2={config.W2}, w3={config.W3}, w4={config.W4}")
        print("\nRunning optimization...")
    
    # Run differential evolution
    result = differential_evolution(
        objective_wrapper,
        bounds=bounds,
        strategy='best1bin',
        maxiter=config.DE_MAXITER,
        popsize=config.DE_POPSIZE,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=config.DE_SEED,
        callback=callback_wrapper,
        polish=True,
        init='latinhypercube',
        workers=1,  # Single process for reproducibility
        updating='deferred',
        tol=0.01,
    )
    
    x_best = result.x
    
    # Compute initial guess for comparison
    x0 = get_default_initial_guess(config)
    J_initial = compute_objective(x0, config, use_simple_tof)
    J_final = result.fun
    
    info = {
        'success': result.success,
        'message': result.message,
        'n_iterations': result.nit,
        'n_evaluations': result.nfev,
        'J_initial': J_initial,
        'J_final': J_final,
        'improvement': (J_initial - J_final) / abs(J_initial) * 100 if J_initial != 0 else 0,
        'history': history,
        'x_initial': x0,
    }
    
    if verbose:
        print(f"\nOptimization complete!")
        print(f"  Status: {'Success' if result.success else 'Did not converge'}")
        print(f"  Message: {result.message}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  J (initial): {J_initial:.4f}")
        print(f"  J (final):   {J_final:.4f}")
        print(f"  Improvement: {info['improvement']:.1f}%")
    
    return x_best, info


def optimize_and_report(
    config: Optional[Config] = None,
    plot: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Run optimization and generate complete report with plots.
    
    Parameters
    ----------
    config : Config, optional
        Configuration object
    plot : bool, optional
        Generate plots (default: True)
    
    Returns
    -------
    tuple
        (x_best, metrics) - Best solution and complete metrics
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Run optimization
    x_best, opt_info = run_optimization(config, verbose=True)
    
    # Print detailed summary
    print_solution_summary(x_best, config)
    
    # Get full evaluation
    metrics = evaluate_solution(x_best, config)
    metrics['optimization'] = opt_info
    
    # Generate plots if requested
    if plot:
        from . import plots
        
        R_in, R_out, theta0, eps1, eps2, eps3 = x_to_params(x_best)
        coords = compute_well_coordinates(R_in, R_out, theta0, eps1, eps2, eps3)
        
        # Well layout plot
        fig1 = plots.plot_well_layout(coords, config)
        
        # Pressure map
        fig2 = plots.plot_pressure_map(coords, config)
        
        # Streamlines
        fig3 = plots.plot_streamlines(coords, config)
        
        # Metrics bars
        fig4 = plots.plot_metrics_bars(metrics)
        
        metrics['figures'] = {
            'layout': fig1,
            'pressure': fig2,
            'streamlines': fig3,
            'metrics': fig4,
        }
    
    return x_best, metrics


def run_parameter_study(
    param_name: str,
    param_values: List[float],
    base_config: Optional[Config] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run parameter sensitivity study.
    
    Parameters
    ----------
    param_name : str
        Name of config parameter to vary
    param_values : list
        Values to test
    base_config : Config, optional
        Base configuration
    verbose : bool, optional
        Print progress
    
    Returns
    -------
    dict
        Results with arrays for each metric
    """
    if base_config is None:
        base_config = DEFAULT_CONFIG
    
    results = {
        'param_values': np.array(param_values),
        'J_best': [],
        'CV_inj': [],
        'CV_prod': [],
        'CV_tof': [],
        'tau_years': [],
        'R_in': [],
        'R_out': [],
    }
    
    for i, val in enumerate(param_values):
        if verbose:
            print(f"\nParameter study {i+1}/{len(param_values)}: {param_name} = {val}")
        
        # Create modified config
        config_dict = {
            'M_DOT_TOTAL': base_config.M_DOT_TOTAL,
            'K_PERM': base_config.K_PERM,
            'H_THICK': base_config.H_THICK,
            'W1': base_config.W1,
            'W2': base_config.W2,
            'W3': base_config.W3,
            'W4': base_config.W4,
            'DE_MAXITER': 50,  # Faster for parameter study
        }
        config_dict[param_name] = val
        
        config = Config(**config_dict)
        
        # Run optimization
        x_best, opt_info = run_optimization(config, verbose=False)
        
        # Evaluate
        metrics = evaluate_solution(x_best, config)
        
        results['J_best'].append(metrics['J'])
        results['CV_inj'].append(metrics['CV_inj'])
        results['CV_prod'].append(metrics['CV_prod'])
        results['CV_tof'].append(metrics['CV_tof'])
        results['tau_years'].append(metrics['tau_years'])
        results['R_in'].append(metrics['R_in'])
        results['R_out'].append(metrics['R_out'])
    
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results
