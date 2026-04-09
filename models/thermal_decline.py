"""Analytical thermal decline model for CPG pattern-level evaluation.

This module implements a reduced-order equivalent-volume heat depletion model
for producer-wise temperature decline in homogeneous reservoirs.

Model summary
-------------
For each producer ``i`` with effective swept volume ``V_eff,i`` [m³] and flow
rate ``m_dot,i`` [kg/s], the no-conduction temperature fraction is:

    G_i(t) = (1 + t * m_dot,i / (V_eff,i * rho_eff * c_eff / c_co2))^-1

Equivalently, using characteristic time ``tau_i`` [s]:

    tau_i = V_eff,i * rho_eff * c_eff / (m_dot,i * c_co2)
    G_i(t) = 1 / (1 + t / tau_i)

and producer temperature is:

    T_i(t) = T_inj + G_i(t) * (T0_i - T_inj)

Units
-----
- Time: seconds internally (helpers for years provided)
- Temperature: K
- Power: W
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


@dataclass(frozen=True)
class ThermalMaterialProperties:
    """Effective material properties for the swept region.

    Parameters
    ----------
    rho_eff : float
        Effective bulk density of swept region [kg/m³].
    c_eff : float
        Effective bulk heat capacity [J/(kg·K)].
    c_co2 : float
        Produced CO2 heat capacity [J/(kg·K)].
    """

    rho_eff: float
    c_eff: float
    c_co2: float



def effective_bulk_heat_capacity(
    porosity: float,
    rho_fluid: float,
    c_fluid: float,
    rho_rock: float,
    c_rock: float,
) -> float:
    """Return effective volumetric heat capacity ``rho_eff * c_eff`` [J/(m³·K)]."""
    return (
        porosity * rho_fluid * c_fluid
        + (1.0 - porosity) * rho_rock * c_rock
    )



def thermal_time_constant(
    m_dot: np.ndarray,
    v_eff: np.ndarray,
    rho_eff: float,
    c_eff: float,
    c_co2: float,
) -> np.ndarray:
    """Return producer-wise thermal time constants ``tau`` [s]."""
    m_dot = np.asarray(m_dot, dtype=float)
    v_eff = np.asarray(v_eff, dtype=float)

    if m_dot.shape != v_eff.shape:
        raise ValueError("m_dot and v_eff must have the same shape.")
    if np.any(m_dot <= 0.0):
        raise ValueError("m_dot must be strictly positive.")
    if np.any(v_eff <= 0.0):
        raise ValueError("v_eff must be strictly positive.")

    return (v_eff * rho_eff * c_eff) / (m_dot * c_co2)



def g_decline_no_conduction(time_s: np.ndarray | float, tau_s: np.ndarray | float) -> np.ndarray:
    """Compute dimensionless temperature fraction ``G(t) = 1 / (1 + t/tau)``."""
    t = np.asarray(time_s, dtype=float)
    tau = np.asarray(tau_s, dtype=float)
    if np.any(t < 0.0):
        raise ValueError("time_s must be non-negative.")
    if np.any(tau <= 0.0):
        raise ValueError("tau_s must be strictly positive.")
    return 1.0 / (1.0 + t / tau)



def producer_temperature_from_g(g: np.ndarray, t_inj_k: float, t0_k: np.ndarray | float) -> np.ndarray:
    """Return producer temperature ``T = T_inj + G * (T0 - T_inj)`` [K]."""
    g = np.asarray(g, dtype=float)
    t0 = np.asarray(t0_k, dtype=float)
    return t_inj_k + g * (t0 - t_inj_k)



def producer_thermal_power(
    m_dot: np.ndarray,
    c_co2: float,
    t_prod_k: np.ndarray,
    t_inj_k: float,
) -> np.ndarray:
    """Return producer thermal power ``P = m_dot * c_co2 * (T_prod - T_inj)`` [W]."""
    m_dot = np.asarray(m_dot, dtype=float)
    t_prod_k = np.asarray(t_prod_k, dtype=float)
    return m_dot * c_co2 * np.maximum(t_prod_k - t_inj_k, 0.0)



def breakthrough_time_proxy(
    tau_s: np.ndarray,
    g_threshold: float = 0.5,
) -> np.ndarray:
    """Return threshold crossing proxy time [s] for ``G(t)=g_threshold``.

    For ``G(t)=1/(1+t/tau)``, ``t_bt = tau * (1/g_threshold - 1)``.
    """
    tau_s = np.asarray(tau_s, dtype=float)
    if not (0.0 < g_threshold < 1.0):
        raise ValueError("g_threshold must be in (0, 1).")
    return tau_s * (1.0 / g_threshold - 1.0)



def time_average_power(
    time_s: np.ndarray,
    power_w: np.ndarray,
) -> float:
    """Return time-averaged total power [W] via trapezoidal integration."""
    t = np.asarray(time_s, dtype=float)
    p = np.asarray(power_w, dtype=float)
    if t.ndim != 1:
        raise ValueError("time_s must be 1D.")
    if p.shape[0] != t.shape[0]:
        raise ValueError("First axis of power_w must match time_s length.")
    if t[-1] <= t[0]:
        raise ValueError("time_s must span a positive interval.")
    total = np.trapz(p, t, axis=0)
    return float(np.sum(total) / (t[-1] - t[0]))



def evaluate_thermal_performance(
    m_dot_i: Iterable[float],
    v_eff_i: Iterable[float],
    t_inj_k: float,
    t0_i_k: Iterable[float] | float,
    props: ThermalMaterialProperties,
    horizon_years: float = 30.0,
    n_time_steps: int = 200,
    g_breakthrough_threshold: float = 0.5,
) -> Dict[str, np.ndarray | float]:
    """Evaluate producer-wise and aggregated thermal performance.

    Returns a dictionary containing time axis, ``G_i(t)``, ``T_i(t)``,
    ``P_i(t)``, pattern totals, breakthrough proxies, and horizon-averaged power.
    """
    m_dot = np.asarray(list(m_dot_i), dtype=float)
    v_eff = np.asarray(list(v_eff_i), dtype=float)
    if np.isscalar(t0_i_k):
        t0 = np.full_like(m_dot, float(t0_i_k), dtype=float)
    else:
        t0 = np.asarray(list(t0_i_k), dtype=float)

    tau = thermal_time_constant(m_dot, v_eff, props.rho_eff, props.c_eff, props.c_co2)
    t_end_s = horizon_years * SECONDS_PER_YEAR
    time_s = np.linspace(0.0, t_end_s, n_time_steps)

    g_ti = g_decline_no_conduction(time_s[:, None], tau[None, :])
    t_ti = producer_temperature_from_g(g_ti, t_inj_k=t_inj_k, t0_k=t0[None, :])
    p_ti = producer_thermal_power(m_dot[None, :], props.c_co2, t_ti, t_inj_k=t_inj_k)
    p_total_t = np.sum(p_ti, axis=1)
    g_avg_t = np.average(g_ti, axis=1, weights=m_dot)

    p_avg = float(np.trapz(p_total_t, time_s) / t_end_s)
    t_bt = breakthrough_time_proxy(tau, g_threshold=g_breakthrough_threshold)

    return {
        "time_s": time_s,
        "time_years": time_s / SECONDS_PER_YEAR,
        "tau_s": tau,
        "G_ti": g_ti,
        "G_avg_t": g_avg_t,
        "T_ti_k": t_ti,
        "P_ti_w": p_ti,
        "P_total_t_w": p_total_t,
        "P_avg_w": p_avg,
        "breakthrough_time_proxy_s": t_bt,
        "breakthrough_time_proxy_years": t_bt / SECONDS_PER_YEAR,
    }
