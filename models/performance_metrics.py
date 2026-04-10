"""Coupled hydraulic + thermal performance metrics for CPG well patterns."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from models.pressure_only import (
    compute_pairwise_impedance,
    producer_rates_from_volume,
    solve_producer_bhp_variable_rate,
    swept_volumes_3inj5prod,
)
from models.thermal_decline import (
    ThermalMaterialProperties,
    evaluate_thermal_performance,
)


def coefficient_of_variation(values: np.ndarray) -> float:
    """Return CV = std/mean with numerical safeguard."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if abs(mean) < 1e-30:
        return float("inf")
    return float(np.std(arr) / abs(mean))



def spacing_metrics(inj_xy: np.ndarray, prod_xy: np.ndarray) -> Dict[str, float]:
    """Compute minimum spacing metrics in meters."""
    inj_xy = np.asarray(inj_xy, dtype=float)
    prod_xy = np.asarray(prod_xy, dtype=float)

    d_ip = np.linalg.norm(prod_xy[:, None, :] - inj_xy[None, :, :], axis=2)
    d_pp = np.linalg.norm(prod_xy[:, None, :] - prod_xy[None, :, :], axis=2)
    np.fill_diagonal(d_pp, np.inf)

    center = np.mean(inj_xy, axis=0)
    r_inj = np.linalg.norm(inj_xy - center[None, :], axis=1)
    r_prod = np.linalg.norm(prod_xy - center[None, :], axis=1)

    return {
        "min_ip_spacing_m": float(np.min(d_ip)),
        "min_pp_spacing_m": float(np.min(d_pp)),
        "min_inj_radius_m": float(np.min(r_inj)),
        "max_inj_radius_m": float(np.max(r_inj)),
        "min_prod_radius_m": float(np.min(r_prod)),
        "max_prod_radius_m": float(np.max(r_prod)),
    }



def default_objective_builder(
    w_power: float = 1.0,
    w_flow_cv: float = 0.2,
    w_pressure: float = 0.15,
    w_penalty: float = 10.0,
) -> Callable[[Dict[str, float]], float]:
    """Build scalar objective from thermal, balance, pressure, and penalties.

    Objective is minimized:
        J = thermal_term + balance_term + pressure_term + penalty_term
          = -w_power * P_avg_norm
            + w_flow_cv * CV_prod
            + w_pressure * max_pressure_drop_norm
            + w_penalty * penalty
    """

    def _objective(m: Dict[str, float]) -> float:
        return (
            -w_power * m["P_avg_norm"]
            + w_flow_cv * m["cv_prod_rates"]
            + w_pressure * m["max_pressure_drop_norm"]
            + w_penalty * m["constraint_penalty"]
        )

    return _objective



def evaluate_layout_performance(
    inj_xy: np.ndarray,
    prod_xy: np.ndarray,
    pressure_params: dict,
    thermal_props: ThermalMaterialProperties,
    p_inj_pa: float,
    q_total_kg_s: float,
    t_inj_k: float,
    t0_k: float,
    horizon_years: float = 30.0,
    pressure_drop_max_pa: float | None = None,
    spacing_min_ip_m: float | None = None,
    spacing_min_pp_m: float | None = None,
    producer_radius_bounds_m: tuple[float, float] | None = None,
    objective_fn: Callable[[Dict[str, float]], float] | None = None,
    p_avg_reference_w: float | None = None,
    depth_m: float | None = None,
) -> Dict[str, object]:
    """Evaluate coupled metrics and scalar objective for one candidate layout."""
    inj_xy = np.asarray(inj_xy, dtype=float)
    prod_xy = np.asarray(prod_xy, dtype=float)
    n_prod = prod_xy.shape[0]

    center = np.mean(inj_xy, axis=0)
    r_inj = float(np.mean(np.linalg.norm(inj_xy - center[None, :], axis=1)))
    r_prod = float(np.mean(np.linalg.norm(prod_xy[1:] - center[None, :], axis=1))) if n_prod > 1 else r_inj * 2.0
    r_top = max(3.0 * r_inj, 1.05 * r_prod)
    v_eff = swept_volumes_3inj5prod(Rin=r_inj, Rout=r_prod, Rtop=r_top, height=float(pressure_params["b"]))
    if len(v_eff) != n_prod:
        # fallback for non 3/5 patterns
        v_eff = np.full(n_prod, np.sum(v_eff) / n_prod)

    q_prod_vec = producer_rates_from_volume(q_total=q_total_kg_s, volumes=v_eff)
    z = compute_pairwise_impedance(inj_xy, prod_xy, pressure_params)
    p_prod, q_ij, q_inj = solve_producer_bhp_variable_rate(p_inj_pa, q_prod_vec, z)

    thermal = evaluate_thermal_performance(
        m_dot_i=q_prod_vec,
        v_eff_i=v_eff,
        t_inj_k=t_inj_k,
        t0_i_k=np.full(n_prod, t0_k),
        props=thermal_props,
        horizon_years=horizon_years,
    )

    p_drop = p_inj_pa - p_prod
    spacing = spacing_metrics(inj_xy, prod_xy)

    pressure_penalty = 0.0
    spacing_penalty = 0.0
    thermal_penalty = 0.0
    if pressure_drop_max_pa is not None:
        pressure_penalty = max(0.0, float(np.max(p_drop) - pressure_drop_max_pa) / pressure_drop_max_pa)
    if spacing_min_ip_m is not None:
        spacing_penalty += max(0.0, (spacing_min_ip_m - spacing["min_ip_spacing_m"]) / spacing_min_ip_m)
    if spacing_min_pp_m is not None:
        spacing_penalty += max(0.0, (spacing_min_pp_m - spacing["min_pp_spacing_m"]) / spacing_min_pp_m)
    if producer_radius_bounds_m is not None:
        rmin, rmax = producer_radius_bounds_m
        spacing_penalty += max(0.0, (rmin - spacing["min_prod_radius_m"]) / max(rmin, 1e-9))
        spacing_penalty += max(0.0, (spacing["max_prod_radius_m"] - rmax) / max(rmax, 1e-9))
    penalty = pressure_penalty + spacing_penalty + thermal_penalty

    p_avg = float(thermal["P_avg_w"])
    # Normalize against a reservoir-side reference scale independent of T_inj.
    # Using a T_inj-dependent normalization masked true T_inj sensitivity in the objective.
    # The reference here anchors to reservoir absolute temperature above 0°C.
    if p_avg_reference_w is None:
        p_ref = max(q_total_kg_s * thermal_props.c_co2 * max(t0_k - 273.15, 1.0), 1.0)
    else:
        p_ref = p_avg_reference_w

    pressure_scale = float(pressure_drop_max_pa) if pressure_drop_max_pa is not None else max(float(p_inj_pa), 1.0)
    scalar_metrics = {
        "P_avg_w": p_avg,
        "P_avg_norm": p_avg / p_ref,
        "cv_prod_rates": coefficient_of_variation(q_prod_vec),
        "cv_inj_rates": coefficient_of_variation(q_inj),
        "mean_pressure_drop_pa": float(np.mean(p_drop)),
        "max_pressure_drop_pa": float(np.max(p_drop)),
        "max_pressure_drop_norm": float(np.max(p_drop)) / pressure_scale,
        "pressure_penalty": float(pressure_penalty),
        "spacing_penalty": float(spacing_penalty),
        "thermal_penalty": float(thermal_penalty),
        "constraint_penalty": float(penalty),
        "breakthrough_mean_years": float(np.mean(thermal["breakthrough_time_proxy_years"])),
        "breakthrough_min_years": float(np.min(thermal["breakthrough_time_proxy_years"])),
        "breakthrough_max_years": float(np.max(thermal["breakthrough_time_proxy_years"])),
        "reservoir_reference_temperature_k": float(t0_k),
        "depth_m": None if depth_m is None else float(depth_m),
        "depth_is_active_in_current_model": False,
        **spacing,
    }

    if objective_fn is None:
        objective_fn = default_objective_builder()
    objective_value = float(objective_fn(scalar_metrics))
    scalar_metrics["objective"] = objective_value
    scalar_metrics["objective_components"] = {
        "thermal_term": float(-scalar_metrics["P_avg_norm"]),
        "balance_term": float(scalar_metrics["cv_prod_rates"]),
        "pressure_term": float(scalar_metrics["max_pressure_drop_norm"]),
        "constraint_penalty_term": float(scalar_metrics["constraint_penalty"]),
    }

    return {
        "objective": objective_value,
        "metrics": scalar_metrics,
        "hydraulics": {
            "P_prod_pa": p_prod,
            "q_prod_kg_s": q_prod_vec,
            "q_inj_kg_s": q_inj,
            "q_ij_kg_s": q_ij,
            "pressure_drop_pa": p_drop,
            "Z_pa_per_kg_s": z,
        },
        "thermal": thermal,
        "geometry": {
            "inj_xy": inj_xy,
            "prod_xy": prod_xy,
            "v_eff_m3": v_eff,
        },
    }
