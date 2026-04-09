"""Validation-oriented synthetic benchmarks for hydraulic and thermal models."""

import numpy as np

from models.pressure_only import compute_pairwise_impedance, impedance_doublet
from models.thermal_decline import ThermalMaterialProperties, evaluate_thermal_performance


def test_impedance_doublet_matrix_consistency():
    params = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    l = 500.0
    z1 = impedance_doublet(params["mu"], params["rho"], params["k"], params["b"], l, params["rw"])
    z2 = compute_pairwise_impedance(np.array([[0.0, 0.0]]), np.array([[l, 0.0]]), params)[0, 0]
    assert np.isclose(z1, z2)


def test_symmetric_inverted_pattern_equal_impedance_to_center():
    params = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    inj = np.array([[250.0, 0.0], [-125.0, 216.506], [-125.0, -216.506]])
    prod = np.array([[0.0, 0.0]])
    z = compute_pairwise_impedance(inj, prod, params)
    assert np.std(z[0]) / np.mean(z[0]) < 1e-3


def test_dimensionless_temperature_decline_shape():
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)
    out = evaluate_thermal_performance(
        m_dot_i=[25.0],
        v_eff_i=[4.0e8],
        t_inj_k=313.15,
        t0_i_k=[393.15],
        props=props,
        horizon_years=30.0,
    )
    g = out["G_ti"][:, 0]
    assert g[0] == 1.0
    assert g[-1] < 1.0
    assert np.all(np.diff(g) <= 1e-12)
