"""
Differential evolution wrapper for the 6-variable well pattern.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from config import (
    DE_MAXITER,
    DE_MUTATION,
    DE_POPSIZE,
    DE_RECOMBINATION,
    DE_SEED,
    DE_TOL,
    EPS_MAX,
    R_IN_MAX,
    R_IN_MIN,
    R_OUT_MAX,
    THETA0_MAX,
    THETA0_MIN,
)
from geometry import generate_wells
from objective import ObjectiveResult, evaluate_objective


def bounds_from_config() -> Tuple[Tuple[float, float], ...]:
    """Return optimization bounds for [R_in, R_out, θ0, ε1, ε2, ε3]."""

    return (
        (R_IN_MIN, R_IN_MAX),
        (R_IN_MIN + 1.0, R_OUT_MAX),
        (THETA0_MIN, THETA0_MAX),
        (-EPS_MAX, EPS_MAX),
        (-EPS_MAX, EPS_MAX),
        (-EPS_MAX, EPS_MAX),
    )


def optimize_layout() -> Tuple[np.ndarray, ObjectiveResult, Dict[str, float]]:
    """Run differential evolution and return best variables and metrics."""

    bounds = bounds_from_config()

    def objective_fn(x: np.ndarray) -> float:
        r_in, r_out, theta0, e1, e2, e3 = x
        result = evaluate_objective(r_in, r_out, theta0, (e1, e2, e3))
        return result.value

    result = differential_evolution(
        objective_fn,
        bounds,
        popsize=DE_POPSIZE,
        maxiter=DE_MAXITER,
        mutation=DE_MUTATION,
        recombination=DE_RECOMBINATION,
        tol=DE_TOL,
        seed=DE_SEED,
        polish=True,
    )

    r_in, r_out, theta0, e1, e2, e3 = result.x
    metrics = evaluate_objective(r_in, r_out, theta0, (e1, e2, e3))
    wells = generate_wells(r_in, r_out, theta0, (e1, e2, e3))
    return result.x, metrics, {"success": result.success, "message": str(result.message), "wells": wells}
