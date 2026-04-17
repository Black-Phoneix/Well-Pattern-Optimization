#!/usr/bin/env python
"""Compare free-center vs fixed-center optimization diagnostics."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.performance_metrics import evaluate_layout_performance
from models.thermal_decline import ThermalMaterialProperties
from optimizers.pattern_optimization import (
    OptimizationConfig,
    build_3inj5prod_layout,
    optimize_3inj5prod_layout,
    optimize_3inj5prod_layout_fixed_center,
)


def hit_upper_bound(value: float, upper_bound: float, tol: float = 1e-6) -> bool:
    return bool(abs(float(value) - float(upper_bound)) <= tol)


def cv(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    return float(np.std(arr) / abs(mean)) if abs(mean) > 1e-30 else float("inf")


def summarize_case(name: str, evald: dict, r_inj: float, r_prod: float, r_prod_ub: float) -> dict:
    m = evald["metrics"]
    h = evald["hydraulics"]
    t = evald["thermal"]
    g = evald["geometry"]

    center_offset = float(np.linalg.norm(np.asarray(g["prod_xy"])[0] - np.mean(np.asarray(g["inj_xy"]), axis=0)))
    bt = np.asarray(t["breakthrough_time_proxy_years"], dtype=float)
    out = {
        "case": name,
        "R_inj_m": float(r_inj),
        "R_prod_m": float(r_prod),
        "center_offset_m": center_offset,
        "R_prod_hit_upper_bound": hit_upper_bound(r_prod, r_prod_ub, tol=1e-3),
        "min_ip_spacing_m": float(m["min_ip_spacing_m"]),
        "min_pp_spacing_m": float(m["min_pp_spacing_m"]),
        "objective": float(evald["objective"]),
        "P_avg_MW": float(m["P_avg_w"]) / 1e6,
        "max_dp_MPa": float(m["max_pressure_drop_pa"]) / 1e6,
        "mean_dp_MPa": float(m["mean_pressure_drop_pa"]) / 1e6,
        "cv_inj": float(m["cv_inj_rates"]),
        "cv_prod": float(m["cv_prod_rates"]),
        "q_inj": np.asarray(h["q_inj_kg_s"]),
        "q_prod": np.asarray(h["q_prod_kg_s"]),
        "v_eff": np.asarray(g["v_eff_m3"]),
        "bt_mean_years": float(np.mean(bt)),
        "bt_cv": cv(bt),
        "T_final_k": np.asarray(t["T_ti_k"][-1]),
        "objective_components": dict(m["objective_components"]),
        "weighted_thermal_term": float(-m["P_avg_norm"]),
        "weighted_balance_term": float(0.2 * m["cv_prod_rates"]),
        "weighted_pressure_term": float(0.15 * m["max_pressure_drop_norm"]),
        "weighted_penalty_term": float(10.0 * m["constraint_penalty"]),
    }
    return out


def print_case_row(s: dict) -> None:
    print(
        f"{s['case']:<34}"
        f"R_inj={s['R_inj_m']:7.2f}  R_prod={s['R_prod_m']:8.2f}  "
        f"offset={s['center_offset_m']:7.2f}  HIT_BOUND={str(s['R_prod_hit_upper_bound']):<5}  "
        f"obj={s['objective']: .6f}  P_avg={s['P_avg_MW']:6.3f} MW  "
        f"maxDP={s['max_dp_MPa']:6.4f} MPa  meanDP={s['mean_dp_MPa']:6.4f} MPa"
    )


def print_details(s: dict, evald: dict) -> None:
    h = evald["hydraulics"]
    q_ij = np.asarray(h["q_ij_kg_s"])
    support = q_ij / np.sum(q_ij, axis=1, keepdims=True)
    print(f"\n--- {s['case']} details ---")
    print(f"min_ip_spacing_m={s['min_ip_spacing_m']:.3f}, min_pp_spacing_m={s['min_pp_spacing_m']:.3f}")
    print(f"injector rate CV={s['cv_inj']:.6f}, producer rate CV={s['cv_prod']:.6f}")
    print(f"injector rates [kg/s]: {np.array2string(s['q_inj'], precision=3)}")
    print(f"producer rates [kg/s]: {np.array2string(s['q_prod'], precision=3)}")
    print(f"swept volumes [m^3]: {np.array2string(s['v_eff'], precision=2)}")
    print(f"breakthrough mean [years]: {s['bt_mean_years']:.3f}, CV={s['bt_cv']:.6f}")
    print(f"final producer temperatures [K]: {np.array2string(s['T_final_k'], precision=3)}")
    print(f"objective components (unweighted): {s['objective_components']}")
    print(
        "weighted components: "
        f"thermal={s['weighted_thermal_term']:.6f}, "
        f"balance={s['weighted_balance_term']:.6f}, "
        f"pressure={s['weighted_pressure_term']:.6f}, "
        f"penalty={s['weighted_penalty_term']:.6f}"
    )
    print("q_ij [kg/s]:")
    print(np.array2string(q_ij, precision=4))
    print("injector support fractions q_ij/sum_j q_ij:")
    print(np.array2string(support, precision=4))


def main() -> None:
    params = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)
    p_inj = 30e6
    q_total = 126.8
    t_inj = 313.15
    t_res = 393.15
    cfg = OptimizationConfig(popsize=8, maxiter=25, seed=42, polish=True)

    base_inj, base_prod = build_3inj5prod_layout(250.0, 600.0, 0.0, np.pi / 4.0)
    baseline_eval = evaluate_layout_performance(
        base_inj,
        base_prod,
        pressure_params=params,
        thermal_props=props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
        pressure_drop_max_pa=8e6,
        spacing_min_ip_m=0.6 * 250.0,
        spacing_min_pp_m=0.7 * 250.0,
        producer_radius_bounds_m=(0.0, 1200.0),
    )

    free_default = optimize_3inj5prod_layout(
        pressure_params=params,
        thermal_props=props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
        cfg=cfg,
    )
    free_extended = optimize_3inj5prod_layout(
        pressure_params=params,
        thermal_props=props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
        cfg=cfg,
        r_prod_bounds_m=(260.0, 1200.0),
    )
    fixed_extended = optimize_3inj5prod_layout_fixed_center(
        pressure_params=params,
        thermal_props=props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
        cfg=cfg,
        r_prod_bounds_m=(260.0, 1200.0),
    )

    summaries = [
        summarize_case("baseline", baseline_eval, 250.0, 600.0, 600.0),
        summarize_case("free-center default bounds", free_default["evaluation"], free_default["x_best"][0], free_default["x_best"][1], 900.0),
        summarize_case("free-center extended bounds", free_extended["evaluation"], free_extended["x_best"][0], free_extended["x_best"][1], 1200.0),
        summarize_case("fixed-center extended bounds", fixed_extended["evaluation"], fixed_extended["x_best"][0], fixed_extended["x_best"][1], 1200.0),
    ]

    print("\n=== FREE vs FIXED CENTER COMPARISON ===")
    print_case_row(summaries[0])
    print_case_row(summaries[1])
    print_case_row(summaries[2])
    print_case_row(summaries[3])

    print("\nBoundary-hit diagnostics:")
    print(f"Free-center default optimization hits R_prod upper bound: {summaries[1]['R_prod_hit_upper_bound']}")
    print(f"Free-center extended optimization hits R_prod upper bound: {summaries[2]['R_prod_hit_upper_bound']}")
    print(f"Fixed-center optimization hits R_prod upper bound: {summaries[3]['R_prod_hit_upper_bound']}")

    print_details(summaries[1], free_default["evaluation"])
    print_details(summaries[2], free_extended["evaluation"])
    print_details(summaries[3], fixed_extended["evaluation"])

    obj_gap = summaries[3]["objective"] - summaries[2]["objective"]
    print("\nInterpretation:")
    if abs(obj_gap) < 0.01:
        print("- Free-center improves objective only slightly vs fixed-center; fixed-center is a good homogeneous-reservoir baseline.")
    else:
        print("- Free-center gives a noticeable objective gain vs fixed-center; asymmetry may be mathematically useful and needs physical justification.")

    if summaries[1]["R_prod_hit_upper_bound"] and summaries[2]["R_prod_hit_upper_bound"] and summaries[3]["R_prod_hit_upper_bound"]:
        print("- R_prod hits the upper bound in all optimization modes; investigate radius bounds or add stronger spacing/thermal trade-off penalties.")
    elif summaries[2]["R_prod_hit_upper_bound"] or summaries[3]["R_prod_hit_upper_bound"]:
        print("- Some modes still hit R_prod upper bound; boundary artifacts remain possible.")
    else:
        print("- Extended bounds removed hard boundary-hitting for R_prod in these runs.")


if __name__ == "__main__":
    main()
