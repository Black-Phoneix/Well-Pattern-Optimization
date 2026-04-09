#!/usr/bin/env python
"""Demo coupled hydraulic + thermal workflow and optimization."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pressure_only import solve_pressure_allocation
from models.thermal_decline import ThermalMaterialProperties
from models.performance_metrics import evaluate_layout_performance
from optimizers.pattern_optimization import (
    build_3inj5prod_layout,
    optimize_3inj5prod_layout,
    run_inverted_five_spot_spacing_study,
)


def main() -> None:
    params = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    thermal_props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)

    p_inj = 30e6
    q_total = 126.8
    t_inj = 313.15
    t_res = 393.15

    inj_xy, prod_xy = build_3inj5prod_layout(250.0, 600.0, 0.0, np.pi / 4.0)

    # pressure-only baseline
    from patterns.geometry import Well

    injectors = [Well(float(x), float(y), "injector") for x, y in inj_xy]
    producers = [Well(float(x), float(y), "producer") for x, y in prod_xy]
    po = solve_pressure_allocation(injectors, producers, p_inj, q_total / len(producers), params)

    coupled = evaluate_layout_performance(
        inj_xy, prod_xy, params, thermal_props, p_inj, q_total, t_inj, t_res
    )

    spacing_sweep = run_inverted_five_spot_spacing_study(
        r_inj=250.0,
        r_prod_values=np.linspace(400.0, 850.0, 20),
        pressure_params=params,
        thermal_props=thermal_props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
    )

    opt = optimize_3inj5prod_layout(
        pressure_params=params,
        thermal_props=thermal_props,
        p_inj_pa=p_inj,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=t_res,
    )

    print("=== Coupled Optimization Demo ===")
    print(f"Baseline pressure-only mean producer BHP [MPa]: {np.mean(po['P_prod'])/1e6:.3f}")
    print(f"Coupled baseline objective: {coupled['objective']:.5f}")
    print(f"Coupled baseline average thermal power [MW]: {coupled['metrics']['P_avg_w']/1e6:.3f}")
    print(f"Spacing sweep optimum R_prod [m]: {spacing_sweep['best_r_prod_m']:.1f}")
    print(f"Optimized objective: {opt['objective_best']:.5f}")
    print("Optimized design vars [R_inj, R_prod, phi_inj0, phi_prod0, cx, cy]:")
    print(opt["x_best"])

    eval_best = opt["evaluation"]
    print("Optimized injector rates [kg/s]:", np.round(eval_best["hydraulics"]["q_inj_kg_s"], 3))
    print("Optimized producer rates [kg/s]:", np.round(eval_best["hydraulics"]["q_prod_kg_s"], 3))
    print("Swept volumes [m3]:", np.round(eval_best["geometry"]["v_eff_m3"], 2))
    print("Breakthrough proxies [years]:", np.round(eval_best["thermal"]["breakthrough_time_proxy_years"], 2))

    # Save layout figure
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        g = eval_best["geometry"]
        ax.scatter(g["inj_xy"][:, 0], g["inj_xy"][:, 1], marker="^", s=90, label="Injectors")
        ax.scatter(g["prod_xy"][:, 0], g["prod_xy"][:, 1], marker="o", s=70, label="Producers")
        for i, (x, y) in enumerate(g["inj_xy"]):
            ax.text(x + 8, y + 8, f"I{i}")
        for i, (x, y) in enumerate(g["prod_xy"]):
            ax.text(x + 8, y + 8, f"P{i}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Optimized 3-Inj / 5-Prod Layout")
        out = Path(__file__).resolve().parent / "coupled_optimized_layout.png"
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)
        print(f"Saved figure: {out}")
    except Exception as exc:
        print(f"Could not save figure (matplotlib unavailable?): {exc}")


if __name__ == "__main__":
    main()
