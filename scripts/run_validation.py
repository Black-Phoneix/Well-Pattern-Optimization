#!/usr/bin/env python
"""Run analytical validation checks for hydraulic and thermal submodels."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pressure_only import compute_pairwise_impedance, impedance_doublet
from models.thermal_decline import (
    ThermalMaterialProperties,
    evaluate_thermal_performance,
    g_decline_no_conduction,
)


def run_impedance_validation() -> dict:
    params = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    l = 700.0
    z_scalar = impedance_doublet(params["mu"], params["rho"], params["k"], params["b"], l, params["rw"])
    inj_xy = np.array([[0.0, 0.0]])
    prod_xy = np.array([[l, 0.0]])
    z_matrix = compute_pairwise_impedance(inj_xy, prod_xy, params)[0, 0]

    symmetric_inj = np.array([[300.0, 0.0], [-150.0, 260.0], [-150.0, -260.0]])
    symmetric_prod = np.array([[0.0, 0.0]])
    z_sym = compute_pairwise_impedance(symmetric_inj, symmetric_prod, params)

    return {
        "doublet_match_relerr": abs(z_scalar - z_matrix) / z_scalar,
        "symmetric_cv": float(np.std(z_sym[0]) / np.mean(z_sym[0])),
        "note": "Literature scalar benchmark hook: replace with published values when curated data is added.",
    }


def run_thermal_validation() -> dict:
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)
    out = evaluate_thermal_performance(
        m_dot_i=[25.0, 25.0],
        v_eff_i=[4.0e8, 4.0e8],
        t_inj_k=313.15,
        t0_i_k=[393.15, 393.15],
        props=props,
        horizon_years=30.0,
        n_time_steps=100,
    )

    g0 = out["G_ti"][0, 0]
    monotonic = np.all(np.diff(out["G_ti"][:, 0]) <= 1e-12)
    g_fast = g_decline_no_conduction(np.array([10.0e6]), np.array([1.0e7]))[0]
    g_slow = g_decline_no_conduction(np.array([10.0e6]), np.array([2.0e7]))[0]

    return {
        "g0": float(g0),
        "monotonic": bool(monotonic),
        "faster_decline_with_smaller_tau": bool(g_fast < g_slow),
        "note": "Dimensionless curve shape benchmark implemented; insert literature target curves in post-processing.",
    }


def main() -> None:
    imp = run_impedance_validation()
    th = run_thermal_validation()
    print("=== Validation Summary ===")
    print("Impedance:", imp)
    print("Thermal:", th)


if __name__ == "__main__":
    main()
