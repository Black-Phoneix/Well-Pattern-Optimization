#!/usr/bin/env python
"""
Demo script for the pressure-only allocation model.

This script demonstrates the pressure-only model for a 3-injector / 5-producer
layout with fixed injector pressure and equal producer flow rates.

Run from the repository root:
    python scripts/demo_pressure_only.py

No thermal effects are considered. This is a pressure-only model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from patterns.geometry import (
    Well,
    generate_center_ring_pattern,
    distance_matrix as geometry_distance_matrix,
)
from models.pressure_only import (
    compute_pairwise_impedance,
    solve_producer_bhp_equal_rate,
    validate_solution,
    validate_total_mass_balance,
    solve_pressure_allocation,
)


def create_3inj_5prod_layout(
    R_inj: float = 600.0,
    R_prod: float = 300.0,
    phi_inj0: float = 0.0,
    phi_prod0: float = np.pi / 4,  # Offset for non-symmetric geometry
    asymmetric: bool = True,
):
    """
    Create a 3-injector / 5-producer well layout.

    Layout description:
    - 1 producer at center (0, 0)
    - 4 producers on a ring of radius R_prod
    - 3 injectors on a ring of radius R_inj at 120° spacing

    Parameters
    ----------
    R_inj : float
        Injector ring radius in meters. Default: 600.0 m.
    R_prod : float
        Producer ring radius in meters. Default: 300.0 m.
    phi_inj0 : float
        Starting angle for injectors in radians. Default: 0.0.
    phi_prod0 : float
        Starting angle for outer producers in radians. Default: π/4.
    asymmetric : bool
        If True, use non-symmetric starting angles. Default: True.

    Returns
    -------
    injectors : list[Well]
        List of 3 injector Well objects.
    producers : list[Well]
        List of 5 producer Well objects (1 center + 4 ring).
    """
    return generate_center_ring_pattern(
        n_inj=3,
        n_prod_outer=4,
        R_inj=R_inj,
        R_prod=R_prod,
        phi_inj0=phi_inj0,
        phi_prod0=phi_prod0 if asymmetric else 0.0,
        center_producer=True,
    )


def print_well_layout(injectors, producers):
    """Print well layout summary table."""
    print("\n" + "=" * 70)
    print("WELL LAYOUT")
    print("=" * 70)

    print("\nInjectors:")
    print(f"{'Index':<8}{'X [m]':<15}{'Y [m]':<15}")
    print("-" * 38)
    for j, w in enumerate(injectors):
        print(f"{j:<8}{w.x:<15.2f}{w.y:<15.2f}")

    print("\nProducers:")
    print(f"{'Index':<8}{'X [m]':<15}{'Y [m]':<15}")
    print("-" * 38)
    for i, w in enumerate(producers):
        print(f"{i:<8}{w.x:<15.2f}{w.y:<15.2f}")


def print_results(P_prod, q_ij, q_inj, producers, injectors, P_inj, q_prod):
    """Print results summary table."""
    print("\n" + "=" * 70)
    print("SOLUTION RESULTS")
    print("=" * 70)

    # Producer pressures
    print("\nProducer Bottom-Hole Pressures:")
    print(f"{'Index':<8}{'X [m]':<12}{'Y [m]':<12}{'P_prod [MPa]':<15}{'ΔP [MPa]':<12}")
    print("-" * 59)
    for i, (w, P) in enumerate(zip(producers, P_prod)):
        dP = P_inj - P
        print(f"{i:<8}{w.x:<12.2f}{w.y:<12.2f}{P/1e6:<15.4f}{dP/1e6:<12.4f}")

    # Statistics
    P_mean = np.mean(P_prod)
    P_std = np.std(P_prod)
    P_range = np.max(P_prod) - np.min(P_prod)
    print(f"\nStatistics:")
    print(f"  Mean P_prod:  {P_mean/1e6:.4f} MPa")
    print(f"  Std P_prod:   {P_std/1e6:.4f} MPa")
    print(f"  Range P_prod: {P_range/1e6:.4f} MPa")
    print(f"  CV (Coef. of Variation): {P_std/P_mean*100:.2f}%")

    # Injector flows
    print("\nInjector Mass Flow Rates:")
    print(f"{'Index':<8}{'X [m]':<12}{'Y [m]':<12}{'q_inj [kg/s]':<15}")
    print("-" * 47)
    for j, (w, q) in enumerate(zip(injectors, q_inj)):
        print(f"{j:<8}{w.x:<12.2f}{w.y:<12.2f}{q:<15.4f}")

    print(f"\nTotal injection rate: {np.sum(q_inj):.4f} kg/s")
    print(f"Total production rate: {q_prod * len(producers):.4f} kg/s ({len(producers)} producers × {q_prod:.2f} kg/s)")

    # Pairwise flow matrix
    print("\nPairwise Flow Matrix q_ij [kg/s] (rows=producers, cols=injectors):")
    header = "Prod\\Inj" + "".join([f"    Inj{j:<5}" for j in range(len(injectors))]) + "    SUM"
    print(header)
    print("-" * len(header))
    for i in range(len(producers)):
        row = f"  Prod{i:<3}"
        for j in range(len(injectors)):
            row += f"  {q_ij[i, j]:8.4f}"
        row += f"  {np.sum(q_ij[i, :]):8.4f}"
        print(row)

    # Column sums
    col_sums = "  SUM    "
    for j in range(len(injectors)):
        col_sums += f"  {np.sum(q_ij[:, j]):8.4f}"
    col_sums += f"  {np.sum(q_ij):8.4f}"
    print(col_sums)


def main():
    """Run the pressure-only allocation demo."""
    print("=" * 70)
    print("PRESSURE-ONLY ALLOCATION MODEL DEMO")
    print("3-Injector / 5-Producer Layout")
    print("=" * 70)

    # =========================================================================
    # 1. Define parameters
    # =========================================================================
    print("\n1. PARAMETERS")
    print("-" * 70)

    # Reservoir parameters
    k = 5e-14       # Permeability [m²] (≈50 mD)
    b = 300.0       # Reservoir thickness [m]
    rw = 0.1        # Well radius [m]

    # Fluid parameters (supercritical CO2 at reservoir conditions)
    mu = 5e-5       # Dynamic viscosity [Pa·s]
    rho = 800.0     # Density [kg/m³]

    # Operating conditions
    P_inj = 30.0e6  # Injector BHP [Pa] (30 MPa)
    q_prod = 10.0   # Production rate per producer [kg/s]

    # Geometry parameters
    R_inj = 600.0   # Injector ring radius [m]
    R_prod = 300.0  # Producer ring radius [m]

    params = {
        'mu': mu,
        'rho': rho,
        'k': k,
        'b': b,
        'rw': rw,
    }

    print(f"Reservoir:")
    print(f"  Permeability:   {k*1e15:.1f} mD (= {k:.2e} m²)")
    print(f"  Thickness:      {b:.1f} m")
    print(f"  Well radius:    {rw:.2f} m")
    print(f"\nFluid (supercritical CO₂):")
    print(f"  Viscosity:      {mu:.2e} Pa·s")
    print(f"  Density:        {rho:.1f} kg/m³")
    print(f"\nOperating Conditions:")
    print(f"  Injector BHP:   {P_inj/1e6:.1f} MPa")
    print(f"  Producer rate:  {q_prod:.1f} kg/s per producer")
    print(f"\nGeometry:")
    print(f"  Injector radius: {R_inj:.1f} m")
    print(f"  Producer radius: {R_prod:.1f} m")

    # =========================================================================
    # 2. Create well layout (non-symmetric)
    # =========================================================================
    print("\n2. WELL LAYOUT (Non-symmetric geometry)")
    print("-" * 70)

    injectors, producers = create_3inj_5prod_layout(
        R_inj=R_inj,
        R_prod=R_prod,
        asymmetric=True,  # Non-symmetric layout
    )

    print(f"Created {len(injectors)} injectors and {len(producers)} producers")
    print_well_layout(injectors, producers)

    # =========================================================================
    # 3. Compute impedance matrix
    # =========================================================================
    print("\n3. IMPEDANCE COMPUTATION")
    print("-" * 70)

    inj_xy = np.array([[w.x, w.y] for w in injectors])
    prod_xy = np.array([[w.x, w.y] for w in producers])

    Z = compute_pairwise_impedance(inj_xy, prod_xy, params)

    print(f"Impedance matrix shape: {Z.shape} (producers × injectors)")
    print("\nImpedance matrix Z [Pa/(kg/s)] (rows=producers, cols=injectors):")
    print(f"{'Prod\\Inj':<10}" + "".join([f"{'Inj' + str(j):<15}" for j in range(len(injectors))]))
    print("-" * (10 + 15 * len(injectors)))
    for i in range(len(producers)):
        row = f"{'Prod' + str(i):<10}"
        for j in range(len(injectors)):
            row += f"{Z[i, j]:.4e}     "
        print(row)

    # =========================================================================
    # 4. Solve for producer pressures and flows
    # =========================================================================
    print("\n4. SOLVE PRESSURE ALLOCATION")
    print("-" * 70)

    P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(P_inj, q_prod, Z)

    print_results(P_prod, q_ij, q_inj, producers, injectors, P_inj, q_prod)

    # =========================================================================
    # 5. Validate solution
    # =========================================================================
    print("\n5. VALIDATION")
    print("-" * 70)

    try:
        validate_solution(q_ij, q_prod, tol=1e-6)
        print("✓ Producer mass balance: PASSED")
    except ValueError as e:
        print(f"✗ Producer mass balance: FAILED - {e}")

    try:
        validate_total_mass_balance(q_inj, q_prod, len(producers), tol=1e-6)
        print("✓ Total mass balance: PASSED")
    except ValueError as e:
        print(f"✗ Total mass balance: FAILED - {e}")

    # Check all flows are positive
    if np.all(q_ij >= 0):
        print("✓ All pairwise flows non-negative: PASSED")
    else:
        print("✗ Some pairwise flows are negative: FAILED")

    # =========================================================================
    # 6. High-level API demo
    # =========================================================================
    print("\n6. HIGH-LEVEL API DEMO")
    print("-" * 70)

    result = solve_pressure_allocation(
        injectors=injectors,
        producers=producers,
        P_inj=P_inj,
        q_prod=q_prod,
        params=params,
        validate=True,
    )

    print("Using solve_pressure_allocation() convenience function...")
    print(f"  Returned keys: {list(result.keys())}")
    print(f"  P_prod matches: {np.allclose(result['P_prod'], P_prod)}")
    print(f"  q_ij matches:   {np.allclose(result['q_ij'], q_ij)}")
    print(f"  q_inj matches:  {np.allclose(result['q_inj'], q_inj)}")

    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Configuration: {len(injectors)} injectors, {len(producers)} producers")
    print(f"Boundary conditions:")
    print(f"  - Injector BHP (Dirichlet): {P_inj/1e6:.1f} MPa (fixed)")
    print(f"  - Producer rate (Neumann):  {q_prod:.1f} kg/s per producer (fixed)")
    print(f"\nKey results:")
    print(f"  - Producer BHP range: {np.min(P_prod)/1e6:.4f} - {np.max(P_prod)/1e6:.4f} MPa")
    print(f"  - Pressure uniformity CV: {np.std(P_prod)/np.mean(P_prod)*100:.2f}%")
    print(f"  - Total injection: {np.sum(q_inj):.2f} kg/s")
    print(f"  - Total production: {q_prod * len(producers):.2f} kg/s")
    print("\n✓ Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
