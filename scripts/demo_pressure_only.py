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
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from patterns.geometry import Well
from models.pressure_only import (
    validate_total_mass_balance,
    validate_solution_variable_rate,
    solve_pressure_allocation,
    optimize_layout_equal_injector_rate,
)


def create_3inj_5prod_layout(
    R_inj: float = 300.0,
    R_prod: float = 600.0,
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

    The producer ring radius must be larger than the injector ring radius.

    Parameters
    ----------
    R_inj : float
        Injector ring radius in meters. Default: 300.0 m.
    R_prod : float
        Producer ring radius in meters. Default: 600.0 m.
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
    if R_prod <= R_inj:
        raise ValueError("Producer ring radius must be larger than injector ring radius.")

    inj_angles = phi_inj0 + np.arange(3) * (2.0 * np.pi / 3.0)
    injectors = [
        Well(R_inj * np.cos(theta), R_inj * np.sin(theta), "injector")
        for theta in inj_angles
    ]

    prod_angles = phi_prod0 + np.arange(4) * (2.0 * np.pi / 4.0)
    if not asymmetric:
        prod_angles = np.arange(4) * (2.0 * np.pi / 4.0)

    producers = [Well(0.0, 0.0, "producer")]
    producers.extend(
        Well(R_prod * np.cos(theta), R_prod * np.sin(theta), "producer")
        for theta in prod_angles
    )

    return injectors, producers


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


def plot_well_layout(injectors, producers, R_inj: float, R_prod: float, output_path: str = "scripts/pressure_only_layout.png"):
    """Plot well coordinates with injector/producer rings and save to file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not available, skipping coordinate plot.")
        return None

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw rings centered at origin
    inj_circle = plt.Circle((0.0, 0.0), R_inj, color='tab:blue', fill=False, linestyle='--', linewidth=1.5, label='Injector ring')
    prod_circle = plt.Circle((0.0, 0.0), R_prod, color='tab:red', fill=False, linestyle='--', linewidth=1.5, label='Producer ring')
    ax.add_patch(inj_circle)
    ax.add_patch(prod_circle)

    inj_xy = np.array([[w.x, w.y] for w in injectors])
    prod_xy = np.array([[w.x, w.y] for w in producers])

    # Injectors: blue triangles
    ax.scatter(inj_xy[:, 0], inj_xy[:, 1], marker='^', c='tab:blue', s=90, label='Injection wells')
    # Producers: red circles
    ax.scatter(prod_xy[:, 0], prod_xy[:, 1], marker='o', c='tab:red', s=70, label='Production wells')

    # Annotate well indices
    for j, (x, y) in enumerate(inj_xy):
        ax.text(x + 10, y + 10, f"I{j}", color='tab:blue', fontsize=9)
    for i, (x, y) in enumerate(prod_xy):
        ax.text(x + 10, y + 10, f"P{i}", color='tab:red', fontsize=9)

    lim = max(R_prod, R_inj) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Pressure-only Well Layout (3 Injectors / 5 Producers)')
    ax.legend(loc='upper right')

    output = Path(output_path)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent.parent / output
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved coordinate plot: {output}")
    return str(output)


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
    q_total = 126.8 # Total production rate [kg/s]
    q_prod = q_total / 5.0  # Equal rate per producer [kg/s]

    # Geometry parameters
    R_inj = 300.0   # Injector ring radius [m]
    radius_ratio = 1.302
    R_prod = radius_ratio * R_inj  # Producer ring radius [m], constrained by ratio

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
    print(f"  Total production rate target: {q_total:.1f} kg/s")
    print(f"  Producer rate (equal split):  {q_prod:.3f} kg/s per producer")
    print(f"\nGeometry:")
    print(f"  Injector radius: {R_inj:.1f} m")
    print(f"  Producer radius: {R_prod:.1f} m")
    print(f"  Radius ratio R_prod/R_inj: {radius_ratio:.3f}")

    # =========================================================================
    # 2. Fix injectors, then optimize producer coordinates using superposition
    # =========================================================================
    print("\n2. PRESSURE OPTIMIZER (Constrained layout + objective)")
    print("-" * 70)

    injectors, _ = create_3inj_5prod_layout(
        R_inj=R_inj,
        R_prod=R_prod,
        asymmetric=True,
    )
    inj_xy = np.array([[w.x, w.y] for w in injectors])

    opt = optimize_layout_equal_injector_rate(
        inj_xy=inj_xy,
        P_inj=P_inj,
        q_total=q_total,
        params=params,
        outer_to_inner_radius_ratio=radius_ratio,
        min_outer_gap_deg=20.0,
        min_ip_factor=0.3,
        injector_rate_rtol=0.02,
        n_trials=30000,
        random_seed=42,
    )

    prod_xy = opt['prod_xy']
    producers = [Well(x, y, 'producer') for x, y in prod_xy]

    print(f"Created {len(injectors)} fixed injectors and optimized {len(producers)} producers")
    print("Applied constraints:")
    print("  1) All injector mass flow rates should be equal")
    print("  2) Center producer is fixed at injector-ring center")
    print("  3) R_prod / R_inj = 1.302")
    center_xy = np.mean(inj_xy, axis=0)
    center_dist = np.linalg.norm(opt['center_xy'] - center_xy)
    sorted_deg = np.sort(opt['outer_angles_deg'])
    gaps_deg = np.diff(np.concatenate([sorted_deg, [sorted_deg[0] + 360.0]]))
    print("Objective:")
    print("  Minimize producer wellhead-pressure variance")
    print("Constraint quality:")
    print(f"  Injector flow relative spread: {float(opt['injector_rate_rel_spread'])*100:.3f}%")
    print(f"  Injector flow tolerance:       {float(opt['injector_rate_rtol'])*100:.3f}%")
    print(f"  Constraint violation:          {float(opt['injector_rate_constraint_violation']):.4e}")
    print(f"  Wellhead pressure variance:    {float(opt['wellhead_pressure_variance']):.4e} Pa²")
    print(f"  center producer distance to injector-ring center: {center_dist:.3f} m")
    print(f"  minimum outer angular gap: {np.min(gaps_deg):.3f}°")
    print("Outer producer polar coordinates (deg, m):")
    for idx, ang in enumerate(opt['outer_angles_deg'], start=1):
        print(f"  Outer producer {idx}: angle={float(ang):.3f}°, radius={float(opt['R_prod']):.3f} m")

    q_prod_vec = opt['q_prod_vec']
    q_prod = float(np.mean(q_prod_vec))

    print("Optimized producer flow split [kg/s] (estimated):")
    for i, qi in enumerate(q_prod_vec):
        print(f"  Producer {i}: {float(qi):.4f}")

    print_well_layout(injectors, producers)
    plot_well_layout(injectors, producers, R_inj=R_inj, R_prod=float(opt['R_prod']))

    # =========================================================================
    # 3. Compute impedance matrix
    # =========================================================================
    print("\n3. IMPEDANCE COMPUTATION")
    print("-" * 70)

    inj_xy = np.array([[w.x, w.y] for w in injectors])
    prod_xy = np.array([[w.x, w.y] for w in producers])

    Z = opt['Z']

    print(f"Impedance matrix shape: {Z.shape} (producers × injectors)")
    print("\nImpedance matrix Z [Pa/(kg/s)] (rows=producers, cols=injectors):")
    header_left = 'Prod\\Inj'
    header_right = ''.join([f"{'Inj' + str(j):<15}" for j in range(len(injectors))])
    print(f"{header_left:<10}" + header_right)
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

    P_prod, q_ij, q_inj = opt['P_prod'], opt['q_ij'], opt['q_inj']

    print_results(P_prod, q_ij, q_inj, producers, injectors, P_inj, q_prod)

    # =========================================================================
    # 5. Validate solution
    # =========================================================================
    print("\n5. VALIDATION")
    print("-" * 70)

    try:
        validate_solution_variable_rate(q_ij, q_prod_vec, tol=1e-6)
        print("✓ Producer mass balance (variable rates): PASSED")
    except ValueError as e:
        print(f"✗ Producer mass balance (variable rates): FAILED - {e}")

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

    print("Using solve_pressure_allocation() convenience function (equal-rate reference)...")
    print(f"  Returned keys: {list(result.keys())}")
    print("  Note: reference API enforces equal producer rates; optimizer uses variable producer rates.")

    # =========================================================================
    # 7. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Configuration: {len(injectors)} injectors, {len(producers)} producers")
    print(f"Boundary conditions:")
    print(f"  - Injector BHP (Dirichlet): {P_inj/1e6:.1f} MPa (fixed)")
    print(f"  - Producer rates (Neumann): variable, mean {np.mean(q_prod_vec):.2f} kg/s")
    print("  - Injector mass flow rates: constrained to be equal (within tolerance).")
    print(f"\nKey results:")
    print(f"  - Producer BHP range: {np.min(P_prod)/1e6:.4f} - {np.max(P_prod)/1e6:.4f} MPa")
    print(f"  - Pressure uniformity CV: {np.std(P_prod)/np.mean(P_prod)*100:.2f}%")
    print(f"  - Total injection: {np.sum(q_inj):.2f} kg/s")
    print(f"  - Total production: {q_total:.2f} kg/s")
    print("  - Injector mass flow rates [kg/s]: " + ", ".join([f"I{j}={q:.2f}" for j, q in enumerate(q_inj)]))
    print("\n✓ Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
