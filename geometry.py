from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Well:
    """
    2D map-view well definition.

    Attributes
    ----------
    x, y : float
        Coordinates [m].
    kind : str
        "injector" or "producer".
    label : str
        Well label (P0, I1.., P1..).
    """

    x: float
    y: float
    kind: str
    label: str


def outer_producer_angles(theta0: float, eps: Iterable[float]) -> np.ndarray:
    """
    Compute outer producer angles from (theta0, eps1..eps3).

    θP1 = θ0
    θP2 = θ0 + (π/2 + ε1)
    θP3 = θP2 + (π/2 + ε2)
    θP4 = θP3 + (π/2 + ε3)
    ε4 = -(ε1+ε2+ε3) enforces closure
    """

    eps = np.asarray(list(eps), dtype=float)
    if eps.size != 3:
        raise ValueError("eps must have length 3 (ε1, ε2, ε3).")
    inc = np.array([np.pi / 2.0 + eps[0], np.pi / 2.0 + eps[1], np.pi / 2.0 + eps[2]])
    theta = np.zeros(4, dtype=float)
    theta[0] = theta0
    theta[1] = theta[0] + inc[0]
    theta[2] = theta[1] + inc[1]
    theta[3] = theta[2] + inc[2]
    return theta


def generate_wells(
    r_in: float,
    r_out: float,
    theta0: float,
    eps: Iterable[float],
) -> Tuple[List[Well], List[Well]]:
    """
    Generate the 1 center producer + 3 injectors + 4 outer producers layout.

    Injectors are fixed at 0, 2π/3, 4π/3 on the inner ring.
    """

    injectors = []
    producers = []

    # Center producer P0
    producers.append(Well(0.0, 0.0, "producer", "P0"))

    # Injectors on inner ring (fixed angles)
    inj_angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    for i, ang in enumerate(inj_angles, start=1):
        x = r_in * np.cos(ang)
        y = r_in * np.sin(ang)
        injectors.append(Well(x, y, "injector", f"I{i}"))

    # Outer producers with variable angles
    angles = outer_producer_angles(theta0, eps)
    for i, ang in enumerate(angles, start=1):
        x = r_out * np.cos(ang)
        y = r_out * np.sin(ang)
        producers.append(Well(x, y, "producer", f"P{i}"))

    return injectors, producers


def distance_matrix(injectors: List[Well], producers: List[Well]) -> np.ndarray:
    """
    Compute injector–producer distance matrix.

    D[j, i] = distance between producer j and injector i
    """

    n_inj = len(injectors)
    n_prod = len(producers)
    dmat = np.zeros((n_prod, n_inj), dtype=float)
    for j, prod in enumerate(producers):
        for i, inj in enumerate(injectors):
            dmat[j, i] = np.hypot(prod.x - inj.x, prod.y - inj.y)
    return dmat


def minimum_spacing(wells: List[Well]) -> float:
    """Return the minimum pairwise distance between wells."""

    coords = np.array([[w.x, w.y] for w in wells], dtype=float)
    d_min = np.inf
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d_min = min(d_min, np.linalg.norm(coords[i] - coords[j]))
    return float(d_min)


def geometry_violations(
    r_in: float,
    r_out: float,
    eps: Iterable[float],
    s_min: float,
    delta_r_min: float,
    delta_theta_min: float,
    wells: List[Well],
) -> dict:
    """
    Compute constraint violations for geometry.

    Returns a dictionary of nonnegative violation magnitudes.
    """

    eps = np.asarray(list(eps), dtype=float)
    eps4 = -(eps[0] + eps[1] + eps[2])
    increments = np.array(
        [np.pi / 2.0 + eps[0], np.pi / 2.0 + eps[1], np.pi / 2.0 + eps[2], np.pi / 2.0 + eps4]
    )

    angle_violation = float(np.sum(np.maximum(0.0, delta_theta_min - increments)))
    violations = {
        "min_spacing": max(0.0, s_min - minimum_spacing(wells)),
        "radial_gap": max(0.0, delta_r_min - (r_out - r_in)),
        "angle_increments": angle_violation,
    }
    return violations
