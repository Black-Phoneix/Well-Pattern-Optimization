import numpy as np

from models.thermal_decline import ThermalMaterialProperties
from optimizers.pattern_optimization import (
    OptimizationConfig,
    optimize_3inj5prod_layout_fixed_center,
)


def test_fixed_center_optimizer_result_structure_and_geometry():
    params = {"mu": 5e-5, "rho": 800.0, "k": 5e-14, "b": 300.0, "rw": 0.1}
    props = ThermalMaterialProperties(rho_eff=2500.0, c_eff=950.0, c_co2=1200.0)
    cfg = OptimizationConfig(popsize=4, maxiter=3, seed=7, polish=False)

    res = optimize_3inj5prod_layout_fixed_center(
        pressure_params=params,
        thermal_props=props,
        p_inj_pa=30e6,
        q_total_kg_s=126.8,
        t_inj_k=313.15,
        t0_k=393.15,
        cfg=cfg,
        r_prod_bounds_m=(260.0, 500.0),
    )

    for key in ["success", "message", "nfev", "nit", "x_best", "objective_best", "evaluation"]:
        assert key in res

    ev = res["evaluation"]
    for key in ["hydraulics", "thermal", "metrics", "geometry"]:
        assert key in ev

    inj_xy = np.asarray(ev["geometry"]["inj_xy"])
    prod_xy = np.asarray(ev["geometry"]["prod_xy"])

    assert inj_xy.shape == (3, 2)
    assert prod_xy.shape == (5, 2)
    np.testing.assert_allclose(prod_xy[0], np.array([0.0, 0.0]), atol=1e-10)
