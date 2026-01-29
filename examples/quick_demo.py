#!/usr/bin/env python
"""
Quick demonstration of well layout optimization system.

This script runs a short optimization to verify the system is working correctly.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from patterns.geometry import generate_ring_pattern, validate_well_layout
from src.optimization.solver import optimize_layout
from src.optimization.objective_func import evaluate_layout_quality
from src.config import ReservoirParams, FluidParams

def main():
    print("=" * 70)
    print("WELL LAYOUT OPTIMIZATION SYSTEM - QUICK DEMO")
    print("=" * 70)
    
    # 1. Define parameters
    print("\n1. Setting up parameters...")
    reservoir = ReservoirParams()
    fluid = FluidParams()
    print(f"   Reservoir: k={reservoir.permeability*1e15:.0f} mD, φ={reservoir.porosity}, h={reservoir.thickness} m")
    print(f"   Fluid: T_inj={fluid.injection_temperature-273.15:.0f}°C, P={fluid.operating_pressure/1e6:.0f} MPa")
    
    # 2. Generate initial pattern
    print("\n2. Generating initial well pattern...")
    injectors, producers = generate_ring_pattern(
        n_inj=3, n_prod=5,
        R_inj=600.0, R_prod=1200.0
    )
    initial_wells = injectors + producers
    print(f"   Created {len(injectors)} injectors and {len(producers)} producers")
    
    # Validate
    is_valid, msg = validate_well_layout(initial_wells, min_spacing=500.0)
    print(f"   Validation: {msg}")
    
    # 3. Evaluate initial quality
    print("\n3. Evaluating initial layout quality...")
    quality_initial = evaluate_layout_quality(
        initial_wells, 50.0, reservoir.__dict__, fluid.__dict__
    )
    print(f"   Breakthrough CV: {quality_initial['breakthrough_cv']:.4f}")
    print(f"   Pressure CV: {quality_initial['pressure_cv']:.4f}")
    print(f"   Min spacing: {quality_initial['min_spacing']:.1f} m")
    print(f"   Total cost: {quality_initial['total_cost']:.6f}")
    
    # 4. Run optimization (short for demo)
    print("\n4. Running optimization (reduced iterations for demo)...")
    print("   This will take 1-2 minutes...")
    
    optimized_wells, info = optimize_layout(
        initial_wells,
        reservoir.__dict__,
        fluid.__dict__,
        de_params={'maxiter': 10, 'popsize': 10},  # Reduced for demo
    )
    
    # 5. Evaluate optimized quality
    print("\n5. Evaluating optimized layout...")
    quality_final = evaluate_layout_quality(
        optimized_wells, 50.0, reservoir.__dict__, fluid.__dict__
    )
    print(f"   Breakthrough CV: {quality_final['breakthrough_cv']:.4f}")
    print(f"   Pressure CV: {quality_final['pressure_cv']:.4f}")
    print(f"   Min spacing: {quality_final['min_spacing']:.1f} m")
    print(f"   Total cost: {quality_final['total_cost']:.6f}")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    print(f"Initial cost:    {info['initial_cost']:.6f}")
    print(f"Final cost:      {info['final_cost']:.6f}")
    print(f"Improvement:     {info['improvement']:.2f}%")
    print(f"Iterations:      {info['n_iterations']}")
    print(f"Function evals:  {info['n_evaluations']}")
    
    # Calculate position changes
    avg_movement = np.mean([
        np.sqrt((w1.x - w2.x)**2 + (w1.y - w2.y)**2)
        for w1, w2 in zip(initial_wells, optimized_wells)
    ])
    print(f"Avg well movement: {avg_movement:.1f} m")
    
    if info['improvement'] > 0:
        print("\n✓ Optimization successful! Layout improved.")
    else:
        print("\n⚠ Optimization completed but no improvement (may need more iterations)")
    
    print("\n" + "=" * 70)
    print("Demo complete. For full optimization, increase maxiter to 50-100.")
    print("=" * 70)

if __name__ == "__main__":
    main()
