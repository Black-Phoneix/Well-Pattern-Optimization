"""Pattern optimization workflows for thesis-ready CPG studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import differential_evolution

from models.performance_metrics import evaluate_layout_performance
from models.thermal_decline import ThermalMaterialProperties


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for differential evolution search."""

    popsize: int = 12
    maxiter: int = 40
    seed: int = 42
    polish: bool = True



def build_3inj5prod_layout(
    r_inj: float,
    r_prod: float,
    phi_inj0: float,
    phi_prod0: float,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create low-dimensional 3-injector / 5-producer layout arrays."""
    inj_angles = phi_inj0 + np.arange(3) * 2.0 * np.pi / 3.0
    prod_angles = phi_prod0 + np.arange(4) * 2.0 * np.pi / 4.0

    inj_xy = np.column_stack([r_inj * np.cos(inj_angles), r_inj * np.sin(inj_angles)])
    prod_outer = np.column_stack([r_prod * np.cos(prod_angles), r_prod * np.sin(prod_angles)])
    prod_center = np.array([[center_offset_x, center_offset_y]], dtype=float)
    prod_xy = np.vstack([prod_center, prod_outer])
    return inj_xy, prod_xy



def run_inverted_five_spot_spacing_study(
    r_inj: float,
    r_prod_values: np.ndarray,
    pressure_params: dict,
    thermal_props: ThermalMaterialProperties,
    p_inj_pa: float,
    q_total_kg_s: float,
    t_inj_k: float,
    t0_k: float,
) -> Dict[str, np.ndarray]:
    """Sweep producer ring spacing to expose optimum in coupled objective."""
    objs, pavg = [], []
    for r_prod in r_prod_values:
        inj_xy, prod_xy = build_3inj5prod_layout(r_inj, float(r_prod), 0.0, np.pi / 4.0)
        evald = evaluate_layout_performance(
            inj_xy,
            prod_xy,
            pressure_params=pressure_params,
            thermal_props=thermal_props,
            p_inj_pa=p_inj_pa,
            q_total_kg_s=q_total_kg_s,
            t_inj_k=t_inj_k,
            t0_k=t0_k,
            spacing_min_ip_m=0.6 * r_inj,
            spacing_min_pp_m=0.75 * r_inj,
        )
        objs.append(evald["objective"])
        pavg.append(evald["metrics"]["P_avg_w"])

    objs = np.asarray(objs)
    pavg = np.asarray(pavg)
    best_idx = int(np.argmin(objs))
    return {
        "r_prod_values_m": np.asarray(r_prod_values, dtype=float),
        "objective": objs,
        "P_avg_w": pavg,
        "best_index": best_idx,
        "best_r_prod_m": float(r_prod_values[best_idx]),
    }



def optimize_3inj5prod_layout(
    pressure_params: dict,
    thermal_props: ThermalMaterialProperties,
    p_inj_pa: float,
    q_total_kg_s: float,
    t_inj_k: float,
    t0_k: float,
    cfg: OptimizationConfig = OptimizationConfig(),
) -> Dict[str, object]:
    """Optimize low-dimensional 3 injector / 5 producer geometry."""

    bounds = [
        (120.0, 350.0),   # r_inj
        (260.0, 900.0),   # r_prod
        (0.0, 2.0 * np.pi),  # phi_inj0
        (0.0, 2.0 * np.pi),  # phi_prod0
        (-80.0, 80.0),    # center x
        (-80.0, 80.0),    # center y
    ]

    def obj(x: np.ndarray) -> float:
        r_inj, r_prod, phi_i, phi_p, cx, cy = x
        if r_prod <= r_inj:
            return 1e6 + (r_inj - r_prod) ** 2
        inj_xy, prod_xy = build_3inj5prod_layout(r_inj, r_prod, phi_i, phi_p, cx, cy)
        result = evaluate_layout_performance(
            inj_xy,
            prod_xy,
            pressure_params=pressure_params,
            thermal_props=thermal_props,
            p_inj_pa=p_inj_pa,
            q_total_kg_s=q_total_kg_s,
            t_inj_k=t_inj_k,
            t0_k=t0_k,
            pressure_drop_max_pa=8e6,
            spacing_min_ip_m=0.6 * r_inj,
            spacing_min_pp_m=0.7 * r_inj,
            producer_radius_bounds_m=(0.0, 1200.0),
        )
        return float(result["objective"])

    result = differential_evolution(
        obj,
        bounds,
        popsize=cfg.popsize,
        maxiter=cfg.maxiter,
        seed=cfg.seed,
        polish=cfg.polish,
        updating="deferred",
    )

    x_best = result.x
    inj_xy, prod_xy = build_3inj5prod_layout(*x_best)
    eval_best = evaluate_layout_performance(
        inj_xy,
        prod_xy,
        pressure_params=pressure_params,
        thermal_props=thermal_props,
        p_inj_pa=p_inj_pa,
        q_total_kg_s=q_total_kg_s,
        t_inj_k=t_inj_k,
        t0_k=t0_k,
        pressure_drop_max_pa=8e6,
        spacing_min_ip_m=0.6 * x_best[0],
        spacing_min_pp_m=0.7 * x_best[0],
        producer_radius_bounds_m=(0.0, 1200.0),
    )

    return {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "x_best": x_best,
        "objective_best": float(result.fun),
        "evaluation": eval_best,
    }
