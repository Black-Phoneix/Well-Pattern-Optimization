import numpy as np

from models.performance_metrics import evaluate_layout_performance
from models.thermal_decline import ThermalMaterialProperties
from optimizers.pattern_optimization import build_3inj5prod_layout


def _base_eval(k=5e-14, b=300.0, t_inj=313.15, q_total=126.8, scale=1.0):
    pressure = {"mu": 5e-5, "rho": 800.0, "k": k, "b": b, "rw": 0.1}
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)
    inj_xy, prod_xy = build_3inj5prod_layout(250.0, 600.0, 0.0, np.pi / 4.0)
    inj_xy = inj_xy * scale
    prod_xy = prod_xy.copy()
    prod_xy[1:] *= scale
    return evaluate_layout_performance(
        inj_xy=inj_xy,
        prod_xy=prod_xy,
        pressure_params=pressure,
        thermal_props=props,
        p_inj_pa=30e6,
        q_total_kg_s=q_total,
        t_inj_k=t_inj,
        t0_k=393.15,
        pressure_drop_max_pa=10e6,
        spacing_min_ip_m=180.0,
        spacing_min_pp_m=200.0,
    )


def test_k_changes_pressure_and_objective():
    low_k = _base_eval(k=3.5e-14)
    high_k = _base_eval(k=6.5e-14)
    assert low_k["metrics"]["max_pressure_drop_pa"] > high_k["metrics"]["max_pressure_drop_pa"]
    assert abs(low_k["objective"] - high_k["objective"]) > 1e-6


def test_tinj_changes_thermal_power_and_objective():
    cool = _base_eval(t_inj=303.15)
    warm = _base_eval(t_inj=323.15)
    assert cool["metrics"]["P_avg_w"] > warm["metrics"]["P_avg_w"]
    assert abs(cool["objective"] - warm["objective"]) > 1e-4


def test_depth_explicitly_flagged_inactive():
    out = _base_eval()
    assert out["metrics"]["depth_is_active_in_current_model"] is False


def test_qtotal_and_spacing_affect_coupled_outputs():
    low_q = _base_eval(q_total=100.0)
    high_q = _base_eval(q_total=150.0)
    assert high_q["metrics"]["P_avg_w"] > low_q["metrics"]["P_avg_w"]
    assert high_q["metrics"]["max_pressure_drop_pa"] > low_q["metrics"]["max_pressure_drop_pa"]

    tight = _base_eval(scale=0.85)
    wide = _base_eval(scale=1.15)
    assert abs(tight["metrics"]["max_pressure_drop_pa"] - wide["metrics"]["max_pressure_drop_pa"]) > 1e-6
    assert abs(tight["metrics"]["P_avg_w"] - wide["metrics"]["P_avg_w"]) > 1e-6
