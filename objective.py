"""
Objective function for well-field optimization.

J = w1*CV_inj + w2*CV_prod + w4*CV_tof - w3*(τ_years/τ_ref) + Penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from config import (
    DELTA_R_MIN,
    DELTA_THETA_MIN,
    S_MIN,
    TAU_REF_YEARS,
    W1,
    W2,
    W3,
    W4,
)
from breakthrough import compute_tof
from geometry import Well, generate_wells, geometry_violations
from hydraulics import allocate_flows, pressure_at_wells, well_rates
from thermal import lifetime_years


@dataclass
class ObjectiveResult:
    """Container for objective and diagnostic metrics."""

    value: float
    cv_inj: float
    cv_prod: float
    cv_tof: float
    tau_years: float
    penalties: Dict[str, float]


def utilization_cvs(wells: Iterable[Well], delta_p: np.ndarray) -> Tuple[float, float]:
    """Compute CV for injectors and producers based on Δp values."""

    wells = list(wells)
    inj_dp = [dp for dp, w in zip(delta_p, wells) if w.kind == "injector"]
    prod_dp = [dp for dp, w in zip(delta_p, wells) if w.kind == "producer"]
    cv_inj = float(np.std(inj_dp) / np.mean(inj_dp)) if len(inj_dp) >= 2 else 0.0
    cv_prod = float(np.std(prod_dp) / np.mean(prod_dp)) if len(prod_dp) >= 2 else 0.0
    return cv_inj, cv_prod


def penalty_from_violations(violations: Dict[str, float], weight: float = 1e6) -> float:
    """Quadratic penalty for violations."""

    return weight * float(sum(v * v for v in violations.values()))


def evaluate_objective(
    r_in: float,
    r_out: float,
    theta0: float,
    eps: Iterable[float],
) -> ObjectiveResult:
    """
    Evaluate objective with full physics and penalties.
    """

    eps = list(eps)
    injectors, producers = generate_wells(r_in, r_out, theta0, eps)
    wells = injectors + producers

    allocation = allocate_flows()
    q_rates = well_rates(wells, allocation)
    delta_p = pressure_at_wells(wells, q_rates, allocation.viscosity)
    cv_inj, cv_prod = utilization_cvs(wells, delta_p)

    tof = compute_tof(injectors, producers, q_rates, allocation.viscosity)
    tau_years = lifetime_years(r_in, r_out)

    violations = geometry_violations(
        r_in,
        r_out,
        eps,
        s_min=S_MIN,
        delta_r_min=DELTA_R_MIN,
        delta_theta_min=DELTA_THETA_MIN,
        wells=wells,
    )
    penalty = penalty_from_violations(violations)

    objective = (
        W1 * cv_inj
        + W2 * cv_prod
        + W4 * (tof.cv_tof if np.isfinite(tof.cv_tof) else 1.0)
        - W3 * (tau_years / TAU_REF_YEARS)
        + penalty
    )

    return ObjectiveResult(
        value=float(objective),
        cv_inj=cv_inj,
        cv_prod=cv_prod,
        cv_tof=float(tof.cv_tof),
        tau_years=float(tau_years),
        penalties=violations,
    )
