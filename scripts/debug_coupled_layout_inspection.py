#!/usr/bin/env python
"""Debug inspection for coupled hydraulic-thermal 3-inj/5-prod layout runs."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.performance_metrics import evaluate_layout_performance
from models.thermal_decline import ThermalMaterialProperties
from optimizers.pattern_optimization import build_3inj5prod_layout, optimize_3inj5prod_layout


def _dist_matrix(prod_xy: np.ndarray, inj_xy: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prod_xy[:, None, :] - inj_xy[None, :, :], axis=2)


def _fmt_arr(arr: np.ndarray, precision: int = 3) -> str:
    return np.array2string(np.asarray(arr), precision=precision, suppress_small=False)


def print_case(title: str, evald: dict, params: dict, thermal_props: ThermalMaterialProperties, p_inj: float, q_total: float, t_inj: float, t_res: float, horizon_years: float) -> None:
    g = evald["geometry"]
    h = evald["hydraulics"]
    t = evald["thermal"]
    m = evald["metrics"]

    inj_xy = np.asarray(g["inj_xy"])
    prod_xy = np.asarray(g["prod_xy"])
    center = np.mean(inj_xy, axis=0)
    d_ip = _dist_matrix(prod_xy, inj_xy)
    d_pp = np.linalg.norm(prod_xy[:, None, :] - prod_xy[None, :, :], axis=2)
    np.fill_diagonal(d_pp, np.inf)

    frac_support = h["q_ij_kg_s"] / np.sum(h["q_ij_kg_s"], axis=1, keepdims=True)

    print(f"\n{'='*90}\n{title}\n{'='*90}")
    print("[1] INPUTS")
    print(f"p_inj [Pa]: {p_inj:.3e}")
    print(f"q_total [kg/s]: {q_total:.3f}")
    print(f"t_inj [K]: {t_inj:.2f}")
    print(f"t_res/t0 [K]: {t_res:.2f}")
    print(f"pressure params: {params}")
    print(f"thermal props: {thermal_props}")

    print("\n[2] GEOMETRY")
    print("injector coordinates [m]:")
    print(_fmt_arr(inj_xy))
    print("producer coordinates [m]:")
    print(_fmt_arr(prod_xy))
    print(f"center producer coordinate [m]: {_fmt_arr(prod_xy[0])}")
    print("outer producer coordinates [m]:")
    print(_fmt_arr(prod_xy[1:]))
    print(f"injector-ring geometric center [m]: {_fmt_arr(center)}")
    print("distance matrix producer->injector [m] (rows producers, cols injectors):")
    print(_fmt_arr(d_ip))
    print(f"mean injector radius [m]: {float(np.mean(np.linalg.norm(inj_xy-center[None,:], axis=1))):.3f}")
    print(f"mean producer radius [m]: {float(np.mean(np.linalg.norm(prod_xy-center[None,:], axis=1))):.3f}")
    print(f"minimum injector-producer spacing [m]: {float(np.min(d_ip)):.3f}")
    print(f"minimum producer-producer spacing [m]: {float(np.min(d_pp)):.3f}")

    print("\n[3] HYDRAULICS")
    print(f"producer BHPs [Pa]: {_fmt_arr(h['P_prod_pa'])}")
    print(f"producer pressure drops [Pa]: {_fmt_arr(h['pressure_drop_pa'])}")
    print(f"injector rates q_inj [kg/s]: {_fmt_arr(h['q_inj_kg_s'])}")
    print(f"producer rates q_prod [kg/s]: {_fmt_arr(h['q_prod_kg_s'])}")
    print("pairwise flow matrix q_ij [kg/s]:")
    print(_fmt_arr(h["q_ij_kg_s"]))
    print("fractional injector support q_ij/sum_j q_ij by producer:")
    print(_fmt_arr(frac_support, precision=4))

    print("\n[4] THERMAL")
    print(f"effective swept volumes v_eff [m^3]: {_fmt_arr(g['v_eff_m3'])}")
    print(f"thermal time constants tau_i [s]: {_fmt_arr(t['tau_s'])}")
    print(f"breakthrough proxies [years]: {_fmt_arr(t['breakthrough_time_proxy_years'])}")
    print(f"initial producer temperatures T_i(t=0) [K]: {_fmt_arr(t['T_ti_k'][0])}")
    print(f"final producer temperatures T_i(t=end) [K]: {_fmt_arr(t['T_ti_k'][-1])}")
    print(f"average thermal power P_avg_w [W]: {m['P_avg_w']:.6e}")
    print(f"initial total thermal power [W]: {float(t['P_total_t_w'][0]):.6e}")
    print(f"final total thermal power [W]: {float(t['P_total_t_w'][-1]):.6e}")
    print(f"time horizon used [years]: {horizon_years:.3f}")

    print("\n[5] OBJECTIVE BREAKDOWN")
    print(f"total objective: {evald['objective']:.8f}")
    print(f"objective components: {m['objective_components']}")
    print(f"thermal term: {m['objective_components']['thermal_term']:.8f}")
    print(f"balance term: {m['objective_components']['balance_term']:.8f}")
    print(f"pressure term: {m['objective_components']['pressure_term']:.8f}")
    print(f"constraint penalty term: {m['objective_components']['constraint_penalty_term']:.8f}")
    print(f"pressure penalty: {m['pressure_penalty']:.8f}")
    print(f"spacing penalty: {m['spacing_penalty']:.8f}")
    print(f"thermal penalty: {m['thermal_penalty']:.8f}")


if __name__ == "__main__":
    params = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    thermal_props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)
    p_inj = 30e6
    q_total = 126.8
    t_inj = 313.15
    t_res = 393.15
    horizon_years = 30.0

    base_inj, base_prod = build_3inj5prod_layout(250.0, 600.0, 0.0, np.pi / 4.0)
    base = evaluate_layout_performance(
        base_inj,
        base_prod,
        pressure_params=params,
        thermal_props=thermal_props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
        horizon_years=horizon_years,
        pressure_drop_max_pa=8e6,
        spacing_min_ip_m=0.6 * 250.0,
        spacing_min_pp_m=0.7 * 250.0,
        producer_radius_bounds_m=(0.0, 1200.0),
    )

    opt = optimize_3inj5prod_layout(
        pressure_params=params,
        thermal_props=thermal_props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
    )
    best = opt["evaluation"]

    print_case("BASELINE LAYOUT", base, params, thermal_props, p_inj, q_total, t_inj, t_res, horizon_years)
    print_case("OPTIMIZED LAYOUT", best, params, thermal_props, p_inj, q_total, t_inj, t_res, horizon_years)
    print("\nOptimization x_best = [R_inj, R_prod, phi_inj0, phi_prod0, cx, cy]:")
    print(_fmt_arr(opt["x_best"], precision=6))
