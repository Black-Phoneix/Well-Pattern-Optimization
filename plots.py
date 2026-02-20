"""
Plotting utilities for the well-field optimization.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import GRID_MARGIN, GRID_N, POROSITY, R_CAPTURE
from geometry import Well
from hydraulics import pressure_field, velocity_field


def plot_layout(ax: plt.Axes, injectors: List[Well], producers: List[Well]) -> None:
    """Plot well layout with labels."""

    ax.scatter([w.x for w in injectors], [w.y for w in injectors], c="tab:blue", label="Injectors")
    ax.scatter([w.x for w in producers], [w.y for w in producers], c="tab:red", label="Producers")
    for w in injectors + producers:
        ax.text(w.x, w.y, w.label, fontsize=9, ha="center", va="center")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()


def plot_pressure_contours(
    ax: plt.Axes,
    wells: List[Well],
    q_rates: np.ndarray,
    viscosity: float,
    r_out: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot pressure contours and return grid arrays."""

    span = r_out + GRID_MARGIN
    x = np.linspace(-span, span, GRID_N)
    y = np.linspace(-span, span, GRID_N)
    xx, yy = np.meshgrid(x, y)
    p = pressure_field(xx, yy, wells, q_rates, viscosity)
    cs = ax.contourf(xx, yy, p, levels=20, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="Pressure (relative)")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return xx, yy, p


def plot_streamlines(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    wells: List[Well],
    q_rates: np.ndarray,
    viscosity: float,
) -> None:
    """Plot streamlines based on velocity field."""

    qx, qy = velocity_field(xx, yy, wells, q_rates, viscosity)
    ax.streamplot(xx, yy, qx / POROSITY, qy / POROSITY, density=1.2, color="white", linewidth=0.8)
    for w in wells:
        if w.kind == "producer":
            circ = plt.Circle((w.x, w.y), R_CAPTURE, color="white", fill=False, linestyle="--", alpha=0.7)
            ax.add_artist(circ)


def plot_bar_metrics(
    ax: plt.Axes,
    labels: List[str],
    values: List[float],
    ylabel: str,
) -> None:
    """Plot bar chart for metrics."""

    ax.bar(labels, values, color="tab:gray")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
