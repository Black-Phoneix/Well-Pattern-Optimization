#!/usr/bin/env python
"""Thesis-ready demo for the steady-state hydraulic model with thermal proxy.

The filename is kept for backward compatibility, but the demo now reports:
- hydraulic layout / spacing constraints,
- the equal-volume thermal design assumption,
- breakthrough-time proxies,
- lightweight thermal-depletion / lifetime outputs, and
- compact comparison studies for thesis-style tables.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from patterns.geometry import Well
from models.pressure_only import (
    optimize_layout_equal_injector_rate,
    run_layout_comparison_study,
    solve_pressure_allocation,
    validate_solution_variable_rate,
    validate_total_mass_balance,
)


def create_injector_ring(R_inj: float, phi_inj0: float = 0.0):
    """Create the fixed 3-injector ring used in the thesis layout."""
    inj_angles = phi_inj0 + np.arange(3) * (2.0 * np.pi / 3.0)
    return [
        Well(R_inj * np.cos(theta), R_inj * np.sin(theta), "injector")
        for theta in inj_angles
    ]


def baseline_layout(inj_xy: np.ndarray, radius_ratio: float):
    """Create a symmetric baseline with 4 equally spaced outer producers."""
    center_xy = np.mean(inj_xy, axis=0)
    R_inj = float(np.mean(np.linalg.norm(inj_xy - center_xy, axis=1)))
    R_out = radius_ratio * R_inj
    angles = np.arange(4) * (2.0 * np.pi / 4.0)
    outer_xy = center_xy + np.column_stack((R_out * np.cos(angles), R_out * np.sin(angles)))
    return np.vstack([center_xy, outer_xy])


def print_case_summary(title: str, result: dict):
    """Print compact thesis-friendly hydraulic and thermal outputs."""
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    print("Hydraulic placement / thermal-domain radii:")
    print(f"  R_inj = {float(result['R_inj']):8.2f} m")
    print(f"  R_in  = {float(result['R_in']):8.2f} m  (fixed equal to R_inj)")
    print(f"  R_out = {float(result['R_out']):8.2f} m  (hydraulic outer producer placement radius)")
    print(f"  R_top = {float(result['R_top']):8.2f} m  (thermal-domain radius)")
    print(f"  R_out / R_inj = {float(result['radius_ratio']):.4f}")

    print("\nHydraulic constraints and spacing metrics:")
    print(f"  injector rate relative spread      = {100.0 * float(result['injector_rate_rel_spread']):7.3f} %")
    print(f"  injector rate tolerance            = {100.0 * float(result['injector_rate_rtol']):7.3f} %")
    print(f"  min injector-producer distance     = {float(result['min_ip_distance']):8.2f} m")
    print(f"  required dmin_ip                   = {float(result['dmin_ip']):8.2f} m")
    print(f"  min producer-producer distance     = {float(result['min_pp_distance']):8.2f} m")
    print(f"  required dmin_pp                   = {float(result['dmin_pp']):8.2f} m")
    print(f"  achieved minimum outer gap         = {float(result['min_outer_gap_deg_achieved']):8.2f} deg")
    print(f"  wellhead-pressure variance         = {float(result['wellhead_pressure_variance']):.4e} Pa^2")

    print("\nThermal design assumption:")
    print("  - Thermal swept volume does NOT depend on R_out in this thesis version.")
    print("  - Center producer thermal control volume = cylinder(R_in = R_inj, b).")
    print("  - Four outer producers share the remaining thermal domain between R_in and R_top equally.")
    print(f"  - Central swept volume             = {float(result['central_swept_volume']):.4e} m^3")
    print(f"  - Outer swept volume per producer  = {float(result['outer_swept_volume_each']):.4e} m^3")

    print("\nBreakthrough proxy τ_i = V_i / q_i:")
    for i, tau_years in enumerate(result['tau_years']):
        print(f"  producer {i}: tau = {float(tau_years):9.4f} years")
    print(f"  breakthrough-time CV              = {float(result['breakthrough_time_cv']):.4e}")

    print("\nLightweight lifetime / depletion outputs:")
    for i, lifetime in enumerate(result['lifetime_years_per_producer']):
        print(f"  producer {i}: lifetime proxy = {float(lifetime):9.4f} years")
    print(f"  mean lifetime                    = {float(result['mean_lifetime_years']):9.4f} years")
    print(f"  minimum lifetime                 = {float(result['min_lifetime_years']):9.4f} years")
    print(f"  time-averaged thermal availability = {float(result['time_averaged_thermal_availability']):.4f}")
    print(f"  simulated horizon                = {float(result['time_years'][-1]):9.4f} years")


def main():
    """Run the comparison demo from the command line."""
    print("=" * 88)
    print("STEADY-STATE HYDRAULIC MODEL WITH THERMAL DESIGN PROXY")
    print("3 injectors + 5 producers | thesis comparison demo")
    print("=" * 88)

    params = {
        'mu': 5e-5,
        'rho': 800.0,
        'k': 5e-14,
        'b': 300.0,
        'rw': 0.1,
    }
    P_inj = 30.0e6
    q_total = 126.8
    R_inj = 300.0

    injectors = create_injector_ring(R_inj)
    inj_xy = np.array([[w.x, w.y] for w in injectors])

    print("\nBase inputs:")
    print(f"  permeability = {params['k']:.3e} m^2")
    print(f"  thickness    = {params['b']:.1f} m")
    print(f"  viscosity    = {params['mu']:.2e} Pa s")
    print(f"  density      = {params['rho']:.1f} kg/m^3")
    print(f"  injector BHP = {P_inj / 1e6:.2f} MPa")
    print(f"  total field rate = {q_total:.2f} kg/s")
    print(f"  injector ring radius R_inj = {R_inj:.2f} m")

    optimized = optimize_layout_equal_injector_rate(
        inj_xy=inj_xy,
        P_inj=P_inj,
        q_total=q_total,
        params=params,
        min_outer_gap_deg=20.0,
        min_ip_factor=0.5,
        min_pp_factor=0.85,
        injector_rate_rtol=0.02,
        n_trials=12000,
        random_seed=42,
    )
    if optimized is None:
        raise RuntimeError('No feasible optimized layout found.')

    # Baseline comparison: same physics, symmetric outer-producer angles, fixed radius ratio.
    baseline_prod_xy = baseline_layout(inj_xy, radius_ratio=2.30)
    baseline_producers = [Well(x, y, 'producer') for x, y in baseline_prod_xy]
    baseline_reference = solve_pressure_allocation(
        injectors=injectors,
        producers=baseline_producers,
        P_inj=P_inj,
        q_prod=q_total / 5.0,
        params=params,
        validate=True,
    )
    print_case_summary('Optimized layout', optimized)

    print(f"\n{'-' * 88}\nBaseline symmetric layout reference\n{'-' * 88}")
    print(f"  baseline outer radius ratio = 2.30")
    print(f"  baseline pressure variance  = {float(np.var(baseline_reference['P_prod'])):.4e} Pa^2 (BHP reference)")
    print(f"  optimized pressure variance = {float(optimized['wellhead_pressure_variance']):.4e} Pa^2 (wellhead proxy)")

    print("\nValidation checks for optimized layout:")
    validate_solution_variable_rate(optimized['q_ij'], optimized['q_prod_vec'], tol=1e-6)
    print("  ✓ producer mass balance")
    validate_total_mass_balance(optimized['q_inj'], float(optimized['q_prod']), 5, tol=1e-6)
    print("  ✓ field mass balance")
    print(f"  ✓ non-negative pairwise flows: {bool(np.all(optimized['q_ij'] >= 0.0))}")

    print(f"\n{'=' * 88}\nCompact comparison study\n{'=' * 88}")
    study = run_layout_comparison_study(
        base_case={
            'inj_xy': inj_xy,
            'P_inj': P_inj,
            'q_total': q_total,
            'params': params,
            'min_outer_gap_deg': 20.0,
            'min_ip_factor': 0.5,
            'min_pp_factor': 0.85,
            'injector_rate_rtol': 0.02,
            'n_trials': 6000,
            'random_seed': 7,
        },
        case_overrides=[
            {'label': 'geom_low_Rinj', 'inj_xy': np.array([[250.0, 0.0], [-125.0, 216.50635095], [-125.0, -216.50635095]])},
            {'label': 'reservoir_high_k', 'params': {'k': 8e-14}},
            {'label': 'ops_high_rate', 'q_total': 145.0},
        ],
    )
    for record in study:
        print(record)

    output_path = Path('scripts') / 'demo_pressure_only_summary.txt'
    output_path.write_text('\n'.join(str(r) for r in study) + '\n')
    print(f"\nSaved comparison summary to {output_path.resolve()}")


if __name__ == '__main__':
    main()
