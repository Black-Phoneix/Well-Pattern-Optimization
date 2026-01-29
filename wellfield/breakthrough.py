"""
Breakthrough and plume uniformity using streamlines and time-of-flight (TOF) proxy.

This module implements physics-based breakthrough uniformity indicators:
- Streamline computation using velocity field integration
- Time-of-flight (TOF) proxy for cold breakthrough estimation

Equations:
    v_p(x) = q(x) / φ    [pore velocity]
    
    dX/dt = v_p(X)       [trajectory equation]
    
    TOF proxy:
        t_bt,k = median TOF of trajectories reaching producer k
        CV_tof = std(t_bt) / mean(t_bt) over 5 producers
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from .config import Config, DEFAULT_CONFIG
from .hydraulics import compute_velocity_field


def _interpolate_velocity(
    x: float,
    y: float,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    porosity: float,
) -> Tuple[float, float]:
    """
    Interpolate pore velocity at arbitrary point using bilinear interpolation.
    
    Parameters
    ----------
    x, y : float
        Point coordinates [m]
    X_grid, Y_grid : np.ndarray
        Grid coordinates
    qx, qy : np.ndarray
        Darcy flux components on grid
    porosity : float
        Porosity for pore velocity
    
    Returns
    -------
    tuple
        (vx, vy) pore velocity at (x, y) [m/s]
    """
    # Get grid bounds
    x_min, x_max = X_grid[0, 0], X_grid[0, -1]
    y_min, y_max = Y_grid[0, 0], Y_grid[-1, 0]
    
    # Check bounds
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return 0.0, 0.0
    
    # Grid spacing
    dx = X_grid[0, 1] - X_grid[0, 0]
    dy = Y_grid[1, 0] - Y_grid[0, 0]
    
    # Find grid cell indices
    i = int((x - x_min) / dx)
    j = int((y - y_min) / dy)
    
    # Clamp to valid range
    i = max(0, min(i, qx.shape[1] - 2))
    j = max(0, min(j, qx.shape[0] - 2))
    
    # Local coordinates in cell [0, 1]
    s = (x - X_grid[j, i]) / dx
    t = (y - Y_grid[j, i]) / dy
    s = max(0.0, min(1.0, s))
    t = max(0.0, min(1.0, t))
    
    # Bilinear interpolation
    qx_val = (
        (1 - s) * (1 - t) * qx[j, i] +
        s * (1 - t) * qx[j, i + 1] +
        (1 - s) * t * qx[j + 1, i] +
        s * t * qx[j + 1, i + 1]
    )
    
    qy_val = (
        (1 - s) * (1 - t) * qy[j, i] +
        s * (1 - t) * qy[j, i + 1] +
        (1 - s) * t * qy[j + 1, i] +
        s * t * qy[j + 1, i + 1]
    )
    
    # Pore velocity
    vx = qx_val / porosity
    vy = qy_val / porosity
    
    return float(vx), float(vy)


def compute_streamlines(
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
    n_grid: int = 100,
    margin: float = 1.2,
) -> Dict[str, np.ndarray]:
    """
    Compute streamlines from injectors to producers.
    
    For each injector, releases N_SEED particles on a small circle around the well.
    Integrates particle trajectories using solve_ivp until they reach a producer
    or time exceeds T_MAX.
    
    Parameters
    ----------
    inj_pos : np.ndarray
        Injector positions (N_inj, 2) [m]
    prod_pos : np.ndarray
        Producer positions (N_prod, 2) [m]
    config : Config, optional
        Configuration object
    n_grid : int, optional
        Grid resolution for velocity interpolation (default: 100)
    margin : float, optional
        Margin factor for grid extent (default: 1.2)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'trajectories': List of (N_points, 2) arrays for each streamline
        - 'times': List of time arrays for each streamline
        - 'tof': List of final times (TOF) for each streamline
        - 'endpoints': List of final endpoint indices (-1 if no producer reached)
        - 'source_inj': List of source injector indices
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Determine grid extent
    all_pos = np.vstack([inj_pos, prod_pos])
    R_max = np.max(np.sqrt(all_pos[:, 0]**2 + all_pos[:, 1]**2))
    extent = margin * R_max
    
    # Create velocity grid
    x_grid = np.linspace(-extent, extent, n_grid)
    y_grid = np.linspace(-extent, extent, n_grid)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Compute velocity field on grid
    qx, qy = compute_velocity_field(X_grid, Y_grid, inj_pos, prod_pos, config)
    
    # Parameters
    phi = config.POROSITY
    r_seed = config.r_seed
    r_capture = config.r_capture
    t_max = config.T_MAX
    n_seed = config.N_SEED
    
    # Results storage
    trajectories = []
    times = []
    tof_values = []
    endpoints = []
    source_injectors = []
    
    # For each injector
    for inj_idx, (x_inj, y_inj) in enumerate(inj_pos):
        # Generate seed points on circle around injector
        seed_angles = np.linspace(0, 2 * np.pi, n_seed, endpoint=False)
        
        for angle in seed_angles:
            x0 = x_inj + r_seed * np.cos(angle)
            y0 = y_inj + r_seed * np.sin(angle)
            
            # Define ODE for streamline
            def streamline_ode(t: float, state: np.ndarray) -> np.ndarray:
                x, y = state
                vx, vy = _interpolate_velocity(x, y, X_grid, Y_grid, qx, qy, phi)
                return np.array([vx, vy])
            
            # Event function: check if reached any producer
            def capture_event(t: float, state: np.ndarray) -> float:
                x, y = state
                min_dist = float('inf')
                for px, py in prod_pos:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    min_dist = min(min_dist, dist)
                return min_dist - r_capture
            
            capture_event.terminal = True
            capture_event.direction = -1
            
            # Integrate streamline
            try:
                sol = solve_ivp(
                    streamline_ode,
                    [0, t_max],
                    [x0, y0],
                    method='RK45',
                    events=capture_event,
                    max_step=t_max / 1000,
                    dense_output=True,
                )
                
                trajectory = sol.y.T  # (N_points, 2)
                time_array = sol.t
                final_time = sol.t[-1]
                
                # Determine which producer was reached
                x_final, y_final = trajectory[-1]
                endpoint_idx = -1
                
                for prod_idx, (px, py) in enumerate(prod_pos):
                    dist = np.sqrt((x_final - px)**2 + (y_final - py)**2)
                    if dist <= r_capture * 1.5:  # Slightly relaxed for detection
                        endpoint_idx = prod_idx
                        break
                
                trajectories.append(trajectory)
                times.append(time_array)
                tof_values.append(final_time)
                endpoints.append(endpoint_idx)
                source_injectors.append(inj_idx)
                
            except Exception:
                # Integration failed - skip this streamline
                continue
    
    return {
        'trajectories': trajectories,
        'times': times,
        'tof': tof_values,
        'endpoints': endpoints,
        'source_inj': source_injectors,
    }


def compute_tof_proxy(
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute time-of-flight breakthrough proxy for each producer.
    
    For each producer k:
        t_bt,k = median TOF of trajectories that end at producer k
    
    CV_tof = std(t_bt) / mean(t_bt) over all producers
    
    Parameters
    ----------
    inj_pos : np.ndarray
        Injector positions (N_inj, 2) [m]
    prod_pos : np.ndarray
        Producer positions (N_prod, 2) [m]
    config : Config, optional
        Configuration object
    
    Returns
    -------
    tuple
        (t_bt, cv_tof)
        - t_bt: Array of median breakthrough times for each producer [s]
        - cv_tof: Coefficient of variation of breakthrough times
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Compute streamlines
    streamline_data = compute_streamlines(inj_pos, prod_pos, config)
    
    tof_values = streamline_data['tof']
    endpoints = streamline_data['endpoints']
    
    n_prod = len(prod_pos)
    t_bt = np.zeros(n_prod)
    
    # Collect TOF for each producer
    for prod_idx in range(n_prod):
        tof_to_producer = [
            tof for tof, end in zip(tof_values, endpoints)
            if end == prod_idx
        ]
        
        if len(tof_to_producer) > 0:
            t_bt[prod_idx] = np.median(tof_to_producer)
        else:
            # No streamlines reached this producer - use maximum TOF
            t_bt[prod_idx] = config.T_MAX
    
    # Compute CV
    mean_tbt = np.mean(t_bt)
    if mean_tbt > 1e-10:
        cv_tof = np.std(t_bt) / mean_tbt
    else:
        cv_tof = 0.0
    
    return t_bt, float(cv_tof)


def compute_tof_simple(
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
) -> Tuple[np.ndarray, float]:
    """
    Simplified TOF computation based on distance and average velocity.
    
    This is a faster approximation when full streamline integration is too slow.
    Uses characteristic velocity from average flow rate and reservoir dimensions.
    
    Parameters
    ----------
    inj_pos : np.ndarray
        Injector positions
    prod_pos : np.ndarray
        Producer positions
    config : Config, optional
        Configuration object
    
    Returns
    -------
    tuple
        (t_bt, cv_tof) - Breakthrough times [s] and CV
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    from .hydraulics import get_mean_fluid_properties, compute_volumetric_flow_rates
    
    # Get flow rates
    Q_rates = compute_volumetric_flow_rates(config)
    Q_total = 3 * Q_rates['Q_inj']  # Total injection rate [m³/s]
    
    # Characteristic cross-sectional area
    H = config.H_THICK
    phi = config.POROSITY
    
    # Characteristic velocity
    # v ≈ Q / (2π r H φ) where r is typical distance
    
    n_prod = len(prod_pos)
    t_bt = np.zeros(n_prod)
    
    for prod_idx, (px, py) in enumerate(prod_pos):
        # Find minimum distance to any injector
        min_dist = float('inf')
        for ix, iy in inj_pos:
            dist = np.sqrt((px - ix)**2 + (py - iy)**2)
            min_dist = min(min_dist, dist)
        
        # Characteristic velocity at this distance
        if min_dist > config.R_WELL:
            v_char = Q_total / (2 * np.pi * min_dist * H * phi)
            t_bt[prod_idx] = min_dist / max(v_char, 1e-10)
        else:
            t_bt[prod_idx] = 0.0
    
    # Compute CV
    mean_tbt = np.mean(t_bt)
    if mean_tbt > 1e-10:
        cv_tof = np.std(t_bt) / mean_tbt
    else:
        cv_tof = 0.0
    
    return t_bt, float(cv_tof)


def get_breakthrough_metrics(
    inj_pos: np.ndarray,
    prod_pos: np.ndarray,
    config: Optional[Config] = None,
    use_simple: bool = True,
) -> Dict[str, float]:
    """
    Get comprehensive breakthrough metrics.
    
    Parameters
    ----------
    inj_pos, prod_pos : np.ndarray
        Well positions
    config : Config, optional
        Configuration object
    use_simple : bool, optional
        Use simplified TOF calculation (faster but less accurate)
    
    Returns
    -------
    dict
        Dictionary with breakthrough metrics:
        - 't_bt_mean': Mean breakthrough time [s]
        - 't_bt_std': Standard deviation of breakthrough times [s]
        - 't_bt_min': Minimum breakthrough time [s]
        - 't_bt_max': Maximum breakthrough time [s]
        - 'cv_tof': Coefficient of variation
        - 't_bt': Array of breakthrough times per producer
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if use_simple:
        t_bt, cv_tof = compute_tof_simple(inj_pos, prod_pos, config)
    else:
        t_bt, cv_tof = compute_tof_proxy(inj_pos, prod_pos, config)
    
    return {
        't_bt_mean': float(np.mean(t_bt)),
        't_bt_std': float(np.std(t_bt)),
        't_bt_min': float(np.min(t_bt)),
        't_bt_max': float(np.max(t_bt)),
        'cv_tof': cv_tof,
        't_bt': t_bt,
    }
