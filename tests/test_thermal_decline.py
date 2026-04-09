"""Unit tests for reduced-order analytical thermal decline model."""

import numpy as np

from models.thermal_decline import (
    ThermalMaterialProperties,
    evaluate_thermal_performance,
)


def test_g_initial_is_one_and_monotonic():
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=1000.0, c_co2=1200.0)
    out = evaluate_thermal_performance(
        m_dot_i=[20.0],
        v_eff_i=[2.0e8],
        t_inj_k=313.15,
        t0_i_k=[393.15],
        props=props,
        horizon_years=20.0,
        n_time_steps=120,
    )
    g = out["G_ti"][:, 0]
    assert np.isclose(g[0], 1.0)
    assert np.all(np.diff(g) <= 1e-12)


def test_higher_flow_faster_decline_and_larger_volume_slower():
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=1000.0, c_co2=1200.0)
    fast = evaluate_thermal_performance(
        m_dot_i=[30.0], v_eff_i=[2.0e8], t_inj_k=313.15, t0_i_k=[393.15], props=props
    )
    slow = evaluate_thermal_performance(
        m_dot_i=[15.0], v_eff_i=[2.0e8], t_inj_k=313.15, t0_i_k=[393.15], props=props
    )
    large = evaluate_thermal_performance(
        m_dot_i=[30.0], v_eff_i=[4.0e8], t_inj_k=313.15, t0_i_k=[393.15], props=props
    )

    idx = -1
    assert fast["G_ti"][idx, 0] < slow["G_ti"][idx, 0]
    assert large["G_ti"][idx, 0] > fast["G_ti"][idx, 0]


def test_equal_v_over_m_gives_equal_breakthrough_proxy():
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=1000.0, c_co2=1200.0)
    out = evaluate_thermal_performance(
        m_dot_i=[10.0, 20.0],
        v_eff_i=[1.0e8, 2.0e8],
        t_inj_k=313.15,
        t0_i_k=[393.15, 393.15],
        props=props,
    )
    bt = out["breakthrough_time_proxy_s"]
    assert np.isclose(bt[0], bt[1], rtol=1e-10)
