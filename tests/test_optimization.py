"""
Unit tests for optimization modules.

Tests constraints, objective function, and solver.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from patterns.geometry import Well, generate_ring_pattern
from src.optimization.constraints import (
    check_minimum_spacing,
    check_field_boundary,
    spacing_penalty_smooth,
    constraint_violation_count,
)
from src.optimization.objective_func import (
    breakthrough_variance_term,
    pressure_variance_term,
    spacing_penalty_term,
)


class TestMinimumSpacing:
    """Test minimum spacing constraint."""
    
    def test_valid_spacing(self):
        """Test that valid spacing passes."""
        positions = [(0, 0), (600, 0), (0, 600)]
        assert check_minimum_spacing(positions, 500.0)
    
    def test_invalid_spacing(self):
        """Test that invalid spacing fails."""
        positions = [(0, 0), (200, 0), (0, 600)]
        assert not check_minimum_spacing(positions, 500.0)
    
    def test_edge_case_exact_spacing(self):
        """Test edge case where spacing exactly equals minimum."""
        positions = [(0, 0), (500, 0)]
        # Should pass (equal to minimum)
        assert check_minimum_spacing(positions, 500.0)


class TestFieldBoundary:
    """Test field boundary constraint."""
    
    def test_circular_boundary_inside(self):
        """Test that wells inside boundary pass."""
        positions = [(0, 0), (500, 0), (0, 500)]
        boundary = {'type': 'circular', 'radius': 1000.0}
        assert check_field_boundary(positions, boundary)
    
    def test_circular_boundary_outside(self):
        """Test that wells outside boundary fail."""
        positions = [(0, 0), (1500, 0)]
        boundary = {'type': 'circular', 'radius': 1000.0}
        assert not check_field_boundary(positions, boundary)
    
    def test_rectangular_boundary(self):
        """Test rectangular boundary."""
        positions = [(50, 50), (100, 100)]
        boundary = {
            'type': 'rectangular',
            'x_min': 0, 'x_max': 200,
            'y_min': 0, 'y_max': 200,
        }
        assert check_field_boundary(positions, boundary)
        
        # Outside
        positions_out = [(250, 50)]
        assert not check_field_boundary(positions_out, boundary)


class TestSmoothPenalty:
    """Test smooth penalty function."""
    
    def test_no_penalty_for_valid_spacing(self):
        """Test that valid spacing has zero penalty."""
        positions = [(0, 0), (600, 0)]
        penalty = spacing_penalty_smooth(positions, 500.0)
        assert penalty == 0.0
    
    def test_penalty_for_violation(self):
        """Test that violation creates positive penalty."""
        positions = [(0, 0), (200, 0)]
        penalty = spacing_penalty_smooth(positions, 500.0)
        assert penalty > 0
    
    def test_penalty_increases_with_violation(self):
        """Test that penalty increases for worse violations."""
        d_min = 500.0
        pos1 = [(0, 0), (400, 0)]  # Smaller violation
        pos2 = [(0, 0), (200, 0)]  # Larger violation
        
        penalty1 = spacing_penalty_smooth(pos1, d_min)
        penalty2 = spacing_penalty_smooth(pos2, d_min)
        
        assert penalty2 > penalty1


class TestConstraintViolationCount:
    """Test constraint violation counting."""
    
    def test_no_violations(self):
        """Test that valid configuration has zero violations."""
        positions = [(0, 0), (600, 0), (0, 600)]
        count = constraint_violation_count(positions, 500.0, 1000.0)
        assert count == 0
    
    def test_spacing_violations(self):
        """Test counting of spacing violations."""
        positions = [(0, 0), (200, 0), (400, 0)]  # Two close pairs
        count = constraint_violation_count(positions, 500.0, 2000.0)
        assert count >= 1  # At least one violation
    
    def test_boundary_violations(self):
        """Test counting of boundary violations."""
        positions = [(0, 0), (1500, 0)]  # One outside
        count = constraint_violation_count(positions, 100.0, 1000.0)
        assert count >= 1


class TestBreakthroughVarianceTerm:
    """Test breakthrough variance objective term."""
    
    def test_variance_term_positive(self):
        """Test that variance term is non-negative."""
        injectors, producers = generate_ring_pattern(
            n_inj=3, n_prod=5, R_inj=500.0, R_prod=1000.0
        )
        wells = injectors + producers
        
        reservoir_params = {
            'porosity': 0.10,
            'thickness': 300.0,
            'rock_density': 2650.0,
            'rock_heat_capacity': 1000.0,
        }
        
        fluid_params = {
            'density': 600.0,
            'heat_capacity': 1200.0,
        }
        
        cv = breakthrough_variance_term(wells, 50.0, reservoir_params, fluid_params)
        
        assert cv >= 0, "Coefficient of variation should be non-negative"
        assert np.isfinite(cv), "CV should be finite"


class TestPressureVarianceTerm:
    """Test pressure variance objective term."""
    
    def test_pressure_variance_symmetric(self):
        """Test that symmetric layout has low pressure variance."""
        # Create perfectly symmetric layout
        R = 1000.0
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        
        injectors = [Well(x=0, y=0, kind='injector')]
        producers = [
            Well(x=R*np.cos(a), y=R*np.sin(a), kind='producer')
            for a in angles
        ]
        wells = injectors + producers
        
        reservoir_params = {
            'porosity': 0.10,
            'thickness': 300.0,
            'rock_density': 2650.0,
            'rock_heat_capacity': 1000.0,
        }
        
        fluid_params = {
            'density': 600.0,
            'heat_capacity': 1200.0,
        }
        
        cv = pressure_variance_term(wells, 50.0, reservoir_params, fluid_params)
        
        # Symmetric layout should have very low variance
        assert cv < 0.1, f"Expected low variance for symmetric layout, got {cv}"


class TestSpacingPenaltyTerm:
    """Test spacing penalty objective term."""
    
    def test_no_penalty_valid_spacing(self):
        """Test zero penalty for valid spacing."""
        wells = [
            Well(x=0, y=0, kind='injector'),
            Well(x=600, y=0, kind='producer'),
            Well(x=0, y=600, kind='producer'),
        ]
        
        penalty = spacing_penalty_term(wells, 500.0)
        assert penalty == 0.0
    
    def test_penalty_invalid_spacing(self):
        """Test positive penalty for invalid spacing."""
        wells = [
            Well(x=0, y=0, kind='injector'),
            Well(x=200, y=0, kind='producer'),
        ]
        
        penalty = spacing_penalty_term(wells, 500.0)
        assert penalty > 0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_optimization_improves_objective(self):
        """Test that a simple optimization iteration improves objective."""
        # This is a simplified test - full optimization tested in solver
        from src.optimization.objective_func import compute_total_cost
        
        reservoir_params = {
            'porosity': 0.10,
            'thickness': 300.0,
            'rock_density': 2650.0,
            'rock_heat_capacity': 1000.0,
        }
        
        fluid_params = {
            'density': 600.0,
            'heat_capacity': 1200.0,
        }
        
        # Initial layout with poor spacing
        x_bad = np.array([0, 0, 200, 0, 400, 0, 600, 0])  # All on a line
        
        # Improved layout with better spacing
        x_good = np.array([0, 0, 600, 0, 0, 600, 600, 600])  # Square pattern
        
        cost_bad = compute_total_cost(
            x_bad, n_injectors=2, n_producers=2,
            reservoir_params=reservoir_params,
            fluid_params=fluid_params,
            min_spacing=500.0,
        )
        
        cost_good = compute_total_cost(
            x_good, n_injectors=2, n_producers=2,
            reservoir_params=reservoir_params,
            fluid_params=fluid_params,
            min_spacing=500.0,
        )
        
        # Good layout should have lower cost
        # (or bad layout should have high penalty)
        assert cost_bad >= cost_good, "Better layout should have lower or equal cost"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
