"""
Thermal model using a conical frustum reservoir volume.

V_res = (1/3) π H (R_out^2 + R_out R_in + R_in^2)
τ_years = Q_res / (Q_dot * 3600 * 24 * 365)
"""

from __future__ import annotations

import importlib.util
import numpy as np

from config import (
    CP_ROCK,
    CP_WAT,
    H_THICK,
    M_DOT_TOTAL,
    P_INJ,
    P_PROD,
    POROSITY,
    RHO_ROCK,
    RHO_WAT,
    S_WIRR,
    T_INJ_C,
    T_PROD_C,
    T_RES_C,
    T_WORKING_C,
)


if importlib.util.find_spec("CoolProp") is None:
    raise ImportError("CoolProp is required. Install via: pip install CoolProp")

from CoolProp.CoolProp import PropsSI


def frustum_volume(r_in: float, r_out: float, thickness: float = H_THICK) -> float:
    """Return frustum volume V_res = (1/3)πH(R_out^2 + R_out R_in + R_in^2)."""

    return (np.pi * thickness / 3.0) * (r_out ** 2 + r_out * r_in + r_in ** 2)


def reservoir_heat_content(
    r_in: float,
    r_out: float,
    thickness: float = H_THICK,
    porosity: float = POROSITY,
    rho_rock: float = RHO_ROCK,
    cp_rock: float = CP_ROCK,
    rho_wat: float = RHO_WAT,
    cp_wat: float = CP_WAT,
    s_wirr: float = S_WIRR,
    t_res_c: float = T_RES_C,
    t_working_c: float = T_WORKING_C,
) -> float:
    """
    Compute total available sensible heat in the reservoir.

    Q_res = (m_rock c_p,rock + m_wat c_p,wat) (T_res - T_working)
    """

    v_res = frustum_volume(r_in, r_out, thickness=thickness)
    m_rock = v_res * (1.0 - porosity) * rho_rock
    m_wat = v_res * porosity * rho_wat * s_wirr
    delta_t = (t_res_c - t_working_c)
    return (m_rock * cp_rock + m_wat * cp_wat) * delta_t


def co2_heat_extraction_rate(
    m_dot_total: float = M_DOT_TOTAL,
    t_inj_c: float = T_INJ_C,
    t_prod_c: float = T_PROD_C,
    p_inj: float = P_INJ,
    p_prod: float = P_PROD,
) -> float:
    """
    Compute heat extraction rate for CO2.

    Q_dot = m_dot_total * cp_CO2 * (T_prod - T_inj)
    """

    if t_prod_c <= t_inj_c:
        raise ValueError("T_PROD_C must be greater than T_INJ_C.")
    t_mean = 0.5 * (t_inj_c + t_prod_c) + 273.15
    p_mean = 0.5 * (p_inj + p_prod)
    cp_co2 = float(PropsSI("C", "T", t_mean, "P", p_mean, "CO2"))
    return m_dot_total * cp_co2 * (t_prod_c - t_inj_c)


def lifetime_years(
    r_in: float,
    r_out: float,
    thickness: float = H_THICK,
) -> float:
    """Compute reservoir lifetime in years."""

    q_res = reservoir_heat_content(r_in, r_out, thickness=thickness)
    q_dot = co2_heat_extraction_rate()
    return q_res / (q_dot * 3600.0 * 24.0 * 365.0)
