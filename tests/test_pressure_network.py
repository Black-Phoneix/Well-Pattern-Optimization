"""
Unit tests for pressure network module.

Tests the Ganjdanesh superposition method implementation against
analytical solutions.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.reservoir.pressure_network import (
    compute_pressure_drop,
    get_well_interference,
    compute_pressure_matrix,
    validate_against_analytical,
)


class TestPressureDrop:
    """Test basic pressure drop calculation."""
    
    def test_pressure_drop_positive(self):
        """Test that injection creates positive pressure."""
        # Parameters
        r = 500.0  # m
        t = 365 * 24 * 3600  # 1 year in seconds
        Q = 0.1  # m³/s (injection)
        mu = 5e-5
        k = 5e-14
        h = 300.0
        phi = 0.10
        ct = 1e-9
        
        dP = compute_pressure_drop(r, t, Q, mu, k, h, phi, ct)
        
        assert dP > 0, "Injection should create positive pressure"
        assert np.isfinite(dP), "Pressure should be finite"
    
    def test_pressure_drop_scaling(self):
        """Test that pressure scales with flow rate."""
        r, t = 500.0, 1e8
        mu, k, h, phi, ct = 5e-5, 5e-14, 300.0, 0.10, 1e-9
        
        Q1 = 0.1
        Q2 = 0.2
        
        dP1 = compute_pressure_drop(r, t, Q1, mu, k, h, phi, ct)
        dP2 = compute_pressure_drop(r, t, Q2, mu, k, h, phi, ct)
        
        # Should approximately double
        ratio = dP2 / dP1
        assert 1.8 < ratio < 2.2, f"Pressure should scale linearly with flow rate, got ratio {ratio}"
    
    def test_zero_time(self):
        """Test that pressure is zero at t=0."""
        r, t = 500.0, 0.0
        Q, mu, k, h, phi, ct = 0.1, 5e-5, 5e-14, 300.0, 0.10, 1e-9
        
        dP = compute_pressure_drop(r, t, Q, mu, k, h, phi, ct)
        
        assert dP == 0.0, "Pressure should be zero at t=0"


class TestWellInterference:
    """Test well-to-well interference calculation."""
    
    def test_interference_symmetry(self):
        """Test that interference is symmetric for equal wells."""
        well1 = (0.0, 0.0)
        well2 = (1000.0, 0.0)
        Q = 0.1
        
        reservoir_params = {
            'permeability': 5e-14,
            'thickness': 300.0,
            'porosity': 0.10,
            'compressibility': 1e-9,
            'viscosity': 5e-5,
        }
        
        t = 1e8  # Large time
        
        dP12 = get_well_interference(well1, well2, Q, reservoir_params, t)
        dP21 = get_well_interference(well2, well1, Q, reservoir_params, t)
        
        # Should be equal due to symmetry
        assert np.isclose(dP12, dP21, rtol=0.01), "Interference should be symmetric"
    
    def test_distance_decay(self):
        """Test that pressure decays with distance."""
        well1 = (0.0, 0.0)
        well2_near = (500.0, 0.0)
        well2_far = (1500.0, 0.0)
        
        reservoir_params = {
            'permeability': 5e-14,
            'thickness': 300.0,
            'porosity': 0.10,
            'compressibility': 1e-9,
            'viscosity': 5e-5,
        }
        
        Q = 0.1
        t = 1e8
        
        dP_near = get_well_interference(well1, well2_near, Q, reservoir_params, t)
        dP_far = get_well_interference(well1, well2_far, Q, reservoir_params, t)
        
        assert dP_near > dP_far, "Pressure should decay with distance"


class TestPressureMatrix:
    """Test pressure matrix computation."""
    
    def test_matrix_shape(self):
        """Test that matrix has correct shape."""
        wells = [
            ((0, 0), 0.1, 'injector'),
            ((1000, 0), -0.1, 'producer'),
            ((0, 1000), -0.1, 'producer'),
        ]
        
        reservoir_params = {
            'permeability': 5e-14,
            'thickness': 300.0,
            'porosity': 0.10,
            'compressibility': 1e-9,
            'viscosity': 5e-5,
        }
        
        time_points = np.array([1e7, 1e8, 1e9])
        
        P = compute_pressure_matrix(wells, reservoir_params, time_points)
        
        assert P.shape == (3, 3, 3), f"Expected shape (3,3,3), got {P.shape}"
    
    def test_diagonal_zeros(self):
        """Test that diagonal elements are zero (self-interference skipped)."""
        wells = [
            ((0, 0), 0.1, 'injector'),
            ((1000, 0), -0.1, 'producer'),
        ]
        
        reservoir_params = {
            'permeability': 5e-14,
            'thickness': 300.0,
            'porosity': 0.10,
            'compressibility': 1e-9,
            'viscosity': 5e-5,
        }
        
        time_points = np.array([1e8])
        P = compute_pressure_matrix(wells, reservoir_params, time_points)
        
        assert P[0, 0, 0] == 0.0, "Diagonal should be zero"
        assert P[1, 1, 0] == 0.0, "Diagonal should be zero"


class TestAnalyticalValidation:
    """Test against analytical solutions."""
    
    def test_validation_passes(self):
        """Test that validation against analytical solution passes."""
        # This test uses the built-in validation function
        success = validate_against_analytical()
        
        # Should pass with <5% error
        assert success, "Validation against analytical solution should pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
