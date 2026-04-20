"""
Breakthrough proxy using streamlines and time-of-flight (TOF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from config import N_SEED, POROSITY, R_CAPTURE, R_SEED, TOF_T_MAX
from geometry import Well
from hydraulics import velocity_field


@dataclass
class TOFResult:
    """Time-of-flight statistics."""

    per_producer: Dict[str, float]
    cv_tof: float
    trajectories: List[np.ndarray]


def _velocity_fun(
    t: float,
    state: np.ndarray,
    wells: List[Well],
    q_rates: np.ndarray,
    viscosity: float,
    porosity: float,
) -> np.ndarray:
    x, y = state
    qx, qy = velocity_field(
        np.array([x]),
        np.array([y]),
        wells,
        q_rates,
        viscosity,
    )
    vpx = qx[0] / porosity
    vpy = qy[0] / porosity
    return np.array([vpx, vpy], dtype=float)


def compute_tof(
    injectors: List[Well],
    producers: List[Well],
    q_rates: np.ndarray,
    viscosity: float,
    porosity: float = POROSITY,
    n_seed: int = N_SEED,
    r_seed: float = R_SEED,
    r_capture: float = R_CAPTURE,
    t_max: float = TOF_T_MAX,
) -> TOFResult:
    """
    Track particles from injectors to producers using pore velocity.

    dX/dt = v_p(X) = q(X) / φ
    """

    trajectories = []
    travel_times: Dict[str, List[float]] = {p.label: [] for p in producers}

    for inj in injectors:
        for k in range(n_seed):
            ang = 2.0 * np.pi * k / n_seed
            x0 = inj.x + r_seed * np.cos(ang)
            y0 = inj.y + r_seed * np.sin(ang)

            def hit_producer(_, state, *_args):
                dx = np.array([state[0] - p.x for p in producers])
                dy = np.array([state[1] - p.y for p in producers])
                return np.min(np.hypot(dx, dy)) - r_capture

            hit_producer.terminal = True
            hit_producer.direction = -1.0

            sol = solve_ivp(
                _velocity_fun,
                (0.0, t_max),
                np.array([x0, y0], dtype=float),
                args=(injectors + producers, q_rates, viscosity, porosity),
                events=hit_producer,
                max_step=t_max / 200.0,
            )

            trajectories.append(sol.y.T)
            if sol.status == 1 and sol.t_events[0].size > 0:
                end = sol.y[:, -1]
                distances = [np.hypot(end[0] - p.x, end[1] - p.y) for p in producers]
                idx = int(np.argmin(distances))
                travel_times[producers[idx].label].append(float(sol.t[-1]))

    per_producer = {}
    for label, times in travel_times.items():
        if len(times) == 0:
            per_producer[label] = np.nan
        else:
            per_producer[label] = float(np.median(times))

    valid = [v for v in per_producer.values() if np.isfinite(v)]
    cv_tof = float(np.std(valid) / np.mean(valid)) if len(valid) >= 2 else np.nan
    return TOFResult(per_producer=per_producer, cv_tof=cv_tof, trajectories=trajectories)
