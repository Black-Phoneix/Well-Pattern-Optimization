"""
Hydraulic forward model using steady Darcy flow and logarithmic potentials.

p(x) = P_ref - F_two_sided * (μ / (2π κ H)) * Σ_j Q_j ln(max(r_j, r_w) / R_b)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import importlib.util
import numpy as np

from config import (
    D_WELL,
    F_TWO_SIDED,
    H_THICK,
    K_PERM,
    M_DOT_TOTAL,
    P_INJ,
    P_PROD,
    R_B,
    R_WELL,
    T_INJ_C,
    T_PROD_C,
)
from geometry import Well


if importlib.util.find_spec("CoolProp") is None:
    raise ImportError("CoolProp is required. Install via: pip install CoolProp")

from CoolProp.CoolProp import PropsSI


@dataclass(frozen=True)
class FlowAllocation:
    """Per-well mass and volumetric rate allocation."""

    m_dot_inj_each: float
    m_dot_prod_each: float
    q_inj_each: float
    q_prod_each: float
    density: float
    viscosity: float


def mean_conditions() -> Tuple[float, float]:
    """Return mean (T, P) in SI units for property evaluation."""

    t_mean = 0.5 * (T_INJ_C + T_PROD_C) + 273.15
    p_mean = 0.5 * (P_INJ + P_PROD)
    return t_mean, p_mean


def fluid_properties() -> Tuple[float, float]:
    """Return (density, viscosity) from CoolProp at mean conditions."""

    t_mean, p_mean = mean_conditions()
    density = float(PropsSI("D", "T", t_mean, "P", p_mean, "CO2"))
    viscosity = float(PropsSI("V", "T", t_mean, "P", p_mean, "CO2"))
    return density, viscosity


def allocate_flows() -> FlowAllocation:
    """
    Allocate fixed per-well mass and volumetric rates.

    ṁ_inj_each = ṁ_total / 3
    ṁ_prod_each = ṁ_total / 5 (negative sign in Q for production)
    Q = sign * ṁ / ρ
    """

    density, viscosity = fluid_properties()
    m_dot_inj_each = M_DOT_TOTAL / 3.0
    m_dot_prod_each = M_DOT_TOTAL / 5.0
    q_inj_each = m_dot_inj_each / density
    q_prod_each = -m_dot_prod_each / density
    return FlowAllocation(m_dot_inj_each, m_dot_prod_each, q_inj_each, q_prod_each, density, viscosity)


def well_rates(wells: Iterable[Well], allocation: FlowAllocation) -> np.ndarray:
    """Return volumetric rates Q_j [m^3/s] for each well."""

    rates = []
    for well in wells:
        if well.kind == "injector":
            rates.append(allocation.q_inj_each)
        else:
            rates.append(allocation.q_prod_each)
    return np.asarray(rates, dtype=float)


def pressure_field(
    x: np.ndarray,
    y: np.ndarray,
    wells: Iterable[Well],
    q_rates: np.ndarray,
    viscosity: float,
    permeability: float = K_PERM,
    thickness: float = H_THICK,
    r_well: float = R_WELL,
    r_b: float = R_B,
    f_two_sided: float = F_TWO_SIDED,
    p_ref: float = 0.0,
) -> np.ndarray:
    """
    Compute pressure field for arrays x, y.

    p(x) = P_ref - F * (μ/(2π κ H)) * Σ_j Q_j ln(max(r_j, r_w)/R_b)
    """

    coeff = f_two_sided * (viscosity / (2.0 * np.pi * permeability * thickness))
    p = np.full_like(x, p_ref, dtype=float)
    for (well, qj) in zip(wells, q_rates):
        r = np.hypot(x - well.x, y - well.y)
        r_eff = np.maximum(r, r_well)
        p -= coeff * qj * np.log(r_eff / r_b)
    return p


def pressure_at_wells(
    wells: Iterable[Well],
    q_rates: np.ndarray,
    viscosity: float,
    permeability: float = K_PERM,
    thickness: float = H_THICK,
    r_well: float = R_WELL,
    r_b: float = R_B,
    f_two_sided: float = F_TWO_SIDED,
) -> np.ndarray:
    """
    Evaluate Δp_i = P_ref - p(x_i).

    With ΣQ=0, Δp_i is independent of P_ref.
    """

    wells = list(wells)
    x = np.array([w.x for w in wells])
    y = np.array([w.y for w in wells])
    p = pressure_field(
        x,
        y,
        wells,
        q_rates,
        viscosity,
        permeability=permeability,
        thickness=thickness,
        r_well=r_well,
        r_b=r_b,
        f_two_sided=f_two_sided,
        p_ref=0.0,
    )
    delta_p = -p
    return delta_p


def velocity_field(
    x: np.ndarray,
    y: np.ndarray,
    wells: Iterable[Well],
    q_rates: np.ndarray,
    viscosity: float,
    permeability: float = K_PERM,
    thickness: float = H_THICK,
    r_well: float = R_WELL,
    r_b: float = R_B,
    f_two_sided: float = F_TWO_SIDED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Darcy flux q = -(κ/μ) ∇p for the log-potential field.

    ∂/∂x ln r = (x-xj)/r^2, ∂/∂y ln r = (y-yj)/r^2
    """

    coeff = f_two_sided * (viscosity / (2.0 * np.pi * permeability * thickness))
    grad_x = np.zeros_like(x, dtype=float)
    grad_y = np.zeros_like(y, dtype=float)
    for well, qj in zip(wells, q_rates):
        dx = x - well.x
        dy = y - well.y
        r2 = np.maximum(dx * dx + dy * dy, r_well * r_well)
        grad_x += qj * (dx / r2)
        grad_y += qj * (dy / r2)
    dpdx = -coeff * grad_x
    dpdy = -coeff * grad_y
    qx = -(permeability / viscosity) * dpdx
    qy = -(permeability / viscosity) * dpdy
    return qx, qy
