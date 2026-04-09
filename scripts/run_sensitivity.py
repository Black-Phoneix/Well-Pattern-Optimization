#!/usr/bin/env python
"""One-factor-at-a-time sensitivity analysis around base and optimized cases."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.performance_metrics import evaluate_layout_performance
from models.thermal_decline import ThermalMaterialProperties
from optimizers.pattern_optimization import optimize_3inj5prod_layout, build_3inj5prod_layout


OUT_CSV = Path(__file__).resolve().parent / "sensitivity_results.csv"


def evaluate_case(inj_xy, prod_xy, pressure_params, thermal_props, t_inj_k, q_total):
    return evaluate_layout_performance(
        inj_xy=inj_xy,
        prod_xy=prod_xy,
        pressure_params=pressure_params,
        thermal_props=thermal_props,
        p_inj_pa=30e6,
        q_total_kg_s=q_total,
        t_inj_k=t_inj_k,
        t0_k=393.15,
        spacing_min_ip_m=180.0,
        spacing_min_pp_m=200.0,
        pressure_drop_max_pa=10e6,
    )


def main() -> None:
    base_pressure = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    thermal_props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)
    base_inj, base_prod = build_3inj5prod_layout(250.0, 600.0, 0.0, np.pi / 4)

    opt = optimize_3inj5prod_layout(
        pressure_params=base_pressure,
        thermal_props=thermal_props,
        p_inj_pa=30e6,
        q_total_kg_s=126.8,
        t_inj_k=313.15,
        t0_k=393.15,
    )
    opt_inj = opt["evaluation"]["geometry"]["inj_xy"]
    opt_prod = opt["evaluation"]["geometry"]["prod_xy"]

    perturbations = {
        "k": [0.7, 1.0, 1.3],
        "b": [0.8, 1.0, 1.2],
        "depth": [2500.0, 3000.0, 3500.0],
        "T_inj": [303.15, 313.15, 323.15],
        "q_total": [100.0, 126.8, 150.0],
        "r_scale": [0.85, 1.0, 1.15],
    }

    rows = []
    for label, inj_xy, prod_xy in (("base", base_inj, base_prod), ("optimized", opt_inj, opt_prod)):
        for p, vals in perturbations.items():
            for val in vals:
                params = dict(base_pressure)
                t_inj = 313.15
                q_total = 126.8
                inj = inj_xy.copy()
                prod = prod_xy.copy()

                if p == "k":
                    params["k"] = base_pressure["k"] * val
                elif p == "b":
                    params["b"] = base_pressure["b"] * val
                elif p == "depth":
                    # optional/secondary; represented as proxy in summary only
                    pass
                elif p == "T_inj":
                    t_inj = val
                elif p == "q_total":
                    q_total = val
                elif p == "r_scale":
                    inj *= val
                    prod[1:] *= val

                res = evaluate_case(inj, prod, params, thermal_props, t_inj, q_total)
                m = res["metrics"]
                rows.append({
                    "layout": label,
                    "parameter": p,
                    "value": val,
                    "objective": res["objective"],
                    "P_avg_w": m["P_avg_w"],
                    "cv_prod_rates": m["cv_prod_rates"],
                    "cv_inj_rates": m["cv_inj_rates"],
                    "max_pressure_drop_pa": m["max_pressure_drop_pa"],
                    "breakthrough_mean_years": float(np.mean(res["thermal"]["breakthrough_time_proxy_years"])),
                    "min_ip_spacing_m": m["min_ip_spacing_m"],
                    "min_pp_spacing_m": m["min_pp_spacing_m"],
                })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # simple ranked sensitivity by objective range
    print(f"Saved: {OUT_CSV}")
    for layout in ["base", "optimized"]:
        subset = [r for r in rows if r["layout"] == layout]
        print(f"\nMost sensitive parameters ({layout}):")
        for p in perturbations:
            vals = [r["objective"] for r in subset if r["parameter"] == p]
            print(f"  {p:<8} objective range = {max(vals)-min(vals):.4f}")


if __name__ == "__main__":
    main()
