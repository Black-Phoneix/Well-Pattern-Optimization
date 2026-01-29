"""
Publication-style plots for well-field optimization.

Required plots:
1. Well layout plot with labels (P0, I1..I3, P1..P4)
2. Pressure map: contour plot of p(x,y)
3. Streamlines + seeds + capture circles
4. Bar/marker plots for Δp and t_bt
"""

from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from .config import Config, DEFAULT_CONFIG


def plot_well_layout(
    coords: Dict[str, Tuple[float, float]],
    config: Optional[Config] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> plt.Figure:
    """
    Plot well layout with labels.
    
    Parameters
    ----------
    coords : dict
        Dictionary with well names as keys and (x, y) coordinates as values
    config : Config, optional
        Configuration object
    ax : plt.Axes, optional
        Axes to plot on (creates new figure if None)
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extract positions
    injectors = ['I1', 'I2', 'I3']
    producers = ['P0', 'P1', 'P2', 'P3', 'P4']
    
    # Plot injectors
    for name in injectors:
        x, y = coords[name]
        ax.scatter(x, y, c='blue', s=200, marker='^', edgecolors='black', linewidths=1.5, zorder=3)
        ax.annotate(name, (x, y), xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold', color='blue')
    
    # Plot producers
    for name in producers:
        x, y = coords[name]
        marker = 'o' if name == 'P0' else 's'  # Circle for center, square for outer
        ax.scatter(x, y, c='red', s=200, marker=marker, edgecolors='black', linewidths=1.5, zorder=3)
        ax.annotate(name, (x, y), xytext=(10, -15), textcoords='offset points',
                   fontsize=12, fontweight='bold', color='red')
    
    # Draw inner and outer rings
    inj_pos = np.array([coords[n] for n in injectors])
    prod_outer_pos = np.array([coords[n] for n in ['P1', 'P2', 'P3', 'P4']])
    
    R_in = np.sqrt(inj_pos[0, 0]**2 + inj_pos[0, 1]**2)
    R_out = np.sqrt(prod_outer_pos[0, 0]**2 + prod_outer_pos[0, 1]**2)
    
    circle_in = Circle((0, 0), R_in, fill=False, linestyle='--', color='blue', alpha=0.5, label=f'R_in = {R_in:.0f} m')
    circle_out = Circle((0, 0), R_out, fill=False, linestyle='--', color='red', alpha=0.5, label=f'R_out = {R_out:.0f} m')
    ax.add_patch(circle_in)
    ax.add_patch(circle_out)
    
    # Formatting
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Well Layout: 1 Center + 3 Injectors + 4 Outer Producers', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set limits with margin
    margin = max(R_out * 0.3, 500)
    ax.set_xlim(-R_out - margin, R_out + margin)
    ax.set_ylim(-R_out - margin, R_out + margin)
    
    plt.tight_layout()
    return fig


def plot_pressure_map(
    coords: Dict[str, Tuple[float, float]],
    config: Optional[Config] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    n_grid: int = 100,
) -> plt.Figure:
    """
    Plot pressure contour map.
    
    Parameters
    ----------
    coords : dict
        Well coordinates
    config : Config, optional
        Configuration object
    ax : plt.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size
    n_grid : int, optional
        Grid resolution
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    from .geometry import compute_all_well_positions
    from .hydraulics import compute_pressure_field
    
    # Get positions
    injectors = ['I1', 'I2', 'I3']
    producers = ['P0', 'P1', 'P2', 'P3', 'P4']
    
    inj_pos = np.array([coords[n] for n in injectors])
    prod_pos = np.array([coords[n] for n in producers])
    
    # Grid extent
    R_out = np.sqrt(prod_pos[1, 0]**2 + prod_pos[1, 1]**2)
    margin = R_out * 0.3
    extent = R_out + margin
    
    # Create grid
    x_grid = np.linspace(-extent, extent, n_grid)
    y_grid = np.linspace(-extent, extent, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute pressure field
    P = compute_pressure_field(X, Y, inj_pos, prod_pos, config)
    
    # Convert to MPa for plotting
    P_MPa = P / 1e6
    
    # Plot contours
    levels = np.linspace(np.nanmin(P_MPa), np.nanmax(P_MPa), 20)
    contour = ax.contourf(X, Y, P_MPa, levels=levels, cmap='RdYlBu_r', extend='both')
    ax.contour(X, Y, P_MPa, levels=levels[::2], colors='black', linewidths=0.5, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='Pressure [MPa]', pad=0.02)
    
    # Plot wells
    for name in injectors:
        x, y = coords[name]
        ax.scatter(x, y, c='blue', s=150, marker='^', edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='white')
    
    for name in producers:
        x, y = coords[name]
        marker = 'o' if name == 'P0' else 's'
        ax.scatter(x, y, c='red', s=150, marker=marker, edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(name, (x, y), xytext=(5, -15), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='white')
    
    # Formatting
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Pressure Field (Logarithmic Potential Superposition)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def plot_streamlines(
    coords: Dict[str, Tuple[float, float]],
    config: Optional[Config] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 8),
    n_grid: int = 80,
    density: float = 1.5,
) -> plt.Figure:
    """
    Plot streamlines with velocity field.
    
    Parameters
    ----------
    coords : dict
        Well coordinates
    config : Config, optional
        Configuration object
    ax : plt.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size
    n_grid : int, optional
        Grid resolution
    density : float, optional
        Streamline density
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    from .hydraulics import compute_velocity_field
    
    # Get positions
    injectors = ['I1', 'I2', 'I3']
    producers = ['P0', 'P1', 'P2', 'P3', 'P4']
    
    inj_pos = np.array([coords[n] for n in injectors])
    prod_pos = np.array([coords[n] for n in producers])
    
    # Grid extent
    R_out = np.sqrt(prod_pos[1, 0]**2 + prod_pos[1, 1]**2)
    margin = R_out * 0.3
    extent = R_out + margin
    
    # Create grid
    x_grid = np.linspace(-extent, extent, n_grid)
    y_grid = np.linspace(-extent, extent, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute velocity field
    qx, qy = compute_velocity_field(X, Y, inj_pos, prod_pos, config)
    
    # Velocity magnitude
    speed = np.sqrt(qx**2 + qy**2)
    
    # Plot streamlines
    strm = ax.streamplot(X, Y, qx, qy, color=np.log10(speed + 1e-15), cmap='viridis',
                         density=density, linewidth=1.0, arrowsize=1.2)
    
    # Colorbar for velocity
    cbar = plt.colorbar(strm.lines, ax=ax, label='log₁₀(|q|) [m/s]', pad=0.02)
    
    # Plot seed circles around injectors
    r_seed = config.r_seed
    for name in injectors:
        x, y = coords[name]
        circle = Circle((x, y), r_seed, fill=False, linestyle=':', color='cyan', linewidth=2)
        ax.add_patch(circle)
        ax.scatter(x, y, c='cyan', s=150, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(name, (x, y), xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='cyan')
    
    # Plot capture circles around producers
    r_capture = config.r_capture
    for name in producers:
        x, y = coords[name]
        circle = Circle((x, y), r_capture, fill=False, linestyle=':', color='orange', linewidth=2)
        ax.add_patch(circle)
        marker = 'o' if name == 'P0' else 's'
        ax.scatter(x, y, c='orange', s=150, marker=marker, edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(name, (x, y), xytext=(8, -15), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='orange')
    
    # Formatting
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Streamlines and Velocity Field', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    
    plt.tight_layout()
    return fig


def plot_metrics_bars(
    metrics: Dict,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """
    Plot bar charts of pressure drops and breakthrough times.
    
    Parameters
    ----------
    metrics : dict
        Evaluation metrics from evaluate_solution()
    ax : plt.Axes, optional
        Axes to plot on (creates 2 subplots if None)
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Pressure drops
    dp_inj = metrics.get('dp_inj_MPa', metrics.get('dp_inj', [0, 0, 0]) / 1e6)
    dp_prod = metrics.get('dp_prod_MPa', metrics.get('dp_prod', [0, 0, 0, 0, 0]) / 1e6)
    
    # Take absolute values for visualization
    dp_inj = np.abs(dp_inj)
    dp_prod = np.abs(dp_prod)
    
    # Injector bars
    x_inj = np.arange(len(dp_inj))
    bars_inj = ax1.bar(x_inj - 0.2, dp_inj, 0.35, label='Injectors', color='blue', alpha=0.7)
    
    # Producer bars
    x_prod = np.arange(len(dp_prod))
    bars_prod = ax1.bar(x_prod[:len(dp_prod)] + len(dp_inj) + 0.5, dp_prod, 0.35, 
                        label='Producers', color='red', alpha=0.7)
    
    # Labels
    inj_labels = [f'I{i+1}' for i in range(len(dp_inj))]
    prod_labels = ['P0'] + [f'P{i}' for i in range(1, len(dp_prod))]
    all_labels = inj_labels + prod_labels
    all_x = list(x_inj - 0.2) + list(x_prod[:len(dp_prod)] + len(dp_inj) + 0.5)
    
    ax1.set_xticks(all_x)
    ax1.set_xticklabels(all_labels)
    ax1.set_ylabel('|Δp| [MPa]', fontsize=12)
    ax1.set_title('Pressure Drops at Wells', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add CV annotation
    if len(dp_inj) > 1:
        cv_inj = np.std(dp_inj) / np.mean(dp_inj) if np.mean(dp_inj) > 0 else 0
        ax1.text(0.02, 0.98, f'CV_inj = {cv_inj:.3f}', transform=ax1.transAxes,
                verticalalignment='top', fontsize=10, color='blue')
    if len(dp_prod) > 1:
        cv_prod = np.std(dp_prod) / np.mean(dp_prod) if np.mean(dp_prod) > 0 else 0
        ax1.text(0.02, 0.90, f'CV_prod = {cv_prod:.3f}', transform=ax1.transAxes,
                verticalalignment='top', fontsize=10, color='red')
    
    # Plot 2: Breakthrough times (if available)
    if 't_bt' in metrics or 't_bt_mean' in metrics:
        t_bt = metrics.get('t_bt', None)
        if t_bt is None:
            # Create dummy data
            t_bt = np.array([metrics.get('t_bt_mean', 1e8)] * 5)
        
        # Convert to years for readability
        t_bt_years = t_bt / (365.25 * 24 * 3600)
        
        x_prod = np.arange(len(t_bt_years))
        bars = ax2.bar(x_prod, t_bt_years, 0.6, color='green', alpha=0.7)
        
        prod_labels = ['P0'] + [f'P{i}' for i in range(1, len(t_bt_years))]
        ax2.set_xticks(x_prod)
        ax2.set_xticklabels(prod_labels)
        ax2.set_ylabel('Breakthrough Time [years]', fontsize=12)
        ax2.set_title('Time-of-Flight Breakthrough Proxy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add CV annotation
        cv_tof = metrics.get('CV_tof', np.std(t_bt_years) / np.mean(t_bt_years) if np.mean(t_bt_years) > 0 else 0)
        ax2.text(0.02, 0.98, f'CV_tof = {cv_tof:.3f}', transform=ax2.transAxes,
                verticalalignment='top', fontsize=10, color='green')
    else:
        ax2.text(0.5, 0.5, 'TOF data not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
    
    plt.tight_layout()
    return fig


def plot_convergence(
    history: Dict,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot optimization convergence history.
    
    Parameters
    ----------
    history : dict
        Optimization history with 'iterations' and 'best_J' keys
    ax : plt.Axes, optional
        Axes to plot on
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        The figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    iterations = history['iterations']
    best_J = history['best_J']
    
    ax.plot(iterations, best_J, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective J', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(best_J) > 1:
        improvement = (best_J[0] - best_J[-1]) / abs(best_J[0]) * 100 if best_J[0] != 0 else 0
        ax.text(0.95, 0.95, f'Improvement: {improvement:.1f}%', transform=ax.transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def create_summary_figure(
    x: np.ndarray,
    metrics: Dict,
    config: Optional[Config] = None,
    figsize: Tuple[float, float] = (16, 12),
) -> plt.Figure:
    """
    Create comprehensive summary figure with all plots.
    
    Parameters
    ----------
    x : np.ndarray
        Optimization variables
    metrics : dict
        Evaluation metrics
    config : Config, optional
        Configuration object
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    plt.Figure
        The figure object with 4 subplots
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    from .geometry import x_to_params, compute_well_coordinates
    
    R_in, R_out, theta0, eps1, eps2, eps3 = x_to_params(x)
    coords = compute_well_coordinates(R_in, R_out, theta0, eps1, eps2, eps3)
    
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2x2 grid
    ax1 = fig.add_subplot(2, 2, 1)  # Well layout
    ax2 = fig.add_subplot(2, 2, 2)  # Pressure map
    ax3 = fig.add_subplot(2, 2, 3)  # Streamlines
    ax4 = fig.add_subplot(2, 2, 4)  # Metrics
    
    # Create individual plots on the axes
    # Note: These functions will use the provided axes
    plot_well_layout(coords, config, ax1)
    plot_pressure_map(coords, config, ax2)
    plot_streamlines(coords, config, ax3)
    
    # Metrics plot (custom for this layout)
    dp_inj = np.abs(metrics.get('dp_inj_MPa', np.zeros(3)))
    dp_prod = np.abs(metrics.get('dp_prod_MPa', np.zeros(5)))
    
    x_wells = np.arange(8)
    labels = ['I1', 'I2', 'I3', 'P0', 'P1', 'P2', 'P3', 'P4']
    values = np.concatenate([dp_inj, dp_prod])
    colors = ['blue'] * 3 + ['red'] * 5
    
    bars = ax4.bar(x_wells, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xticks(x_wells)
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('|Δp| [MPa]', fontsize=12)
    ax4.set_title('Pressure Drops', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add summary text
    summary_text = (
        f"J = {metrics.get('J', 0):.4f}\n"
        f"CV_inj = {metrics.get('CV_inj', 0):.4f}\n"
        f"CV_prod = {metrics.get('CV_prod', 0):.4f}\n"
        f"CV_tof = {metrics.get('CV_tof', 0):.4f}\n"
        f"τ = {metrics.get('tau_years', 0):.1f} years"
    )
    ax4.text(0.98, 0.98, summary_text, transform=ax4.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    plt.suptitle('Well-Field Optimization Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig
