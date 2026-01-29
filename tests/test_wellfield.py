"""
Unit tests for the wellfield optimization package.

Tests all modules: config, geometry, hydraulics, thermal, breakthrough, objective, optimize.
"""

import pytest
import numpy as np


class TestConfig:
    """Test configuration module."""
    
    def test_config_defaults(self):
        """Test that default config has expected values."""
        from wellfield.config import Config
        
        config = Config()
        
        # Hydraulics
        assert config.M_DOT_TOTAL == 100.0
        assert config.K_PERM == 5e-14
        assert config.H_THICK == 300.0
        assert config.D_WELL == 0.41
        assert config.P_INJ == 200e5
        assert config.P_PROD == 150e5
        assert config.T_INJ_C == 40.0
        assert config.T_PROD_C == 100.0
        
        # Thermal
        assert config.POROSITY == 0.10
        assert config.RHO_ROCK == 2300.0
        assert config.CP_ROCK == 0.92e3
        
        # Constraints
        assert config.S_MIN == 500.0
        assert config.DELTA_R_MIN == 500.0
    
    def test_config_derived_properties(self):
        """Test derived properties are computed correctly."""
        from wellfield.config import Config
        
        config = Config()
        
        assert config.R_WELL == config.D_WELL / 2.0
        assert config.T_INJ_K == config.T_INJ_C + 273.15
        assert config.P_MEAN == 0.5 * (config.P_INJ + config.P_PROD)
        assert config.m_dot_inj_each == config.M_DOT_TOTAL / 3.0
        assert config.m_dot_prod_each == config.M_DOT_TOTAL / 5.0
    
    def test_config_validation(self):
        """Test config validation catches invalid values."""
        from wellfield.config import Config
        
        # T_prod <= T_inj should raise
        with pytest.raises(ValueError):
            Config(T_PROD_C=30.0, T_INJ_C=40.0)
        
        # P_inj <= P_prod should raise
        with pytest.raises(ValueError):
            Config(P_INJ=100e5, P_PROD=150e5)


class TestGeometry:
    """Test geometry module."""
    
    def test_well_coordinates(self):
        """Test well coordinate computation."""
        from wellfield.geometry import compute_well_coordinates
        
        coords = compute_well_coordinates(1000, 2000, 0, 0, 0, 0)
        
        # Check center producer
        assert coords['P0'] == (0.0, 0.0)
        
        # Check injector I1 at angle 0
        x, y = coords['I1']
        assert np.isclose(x, 1000.0, rtol=1e-6)
        assert np.isclose(y, 0.0, atol=1e-6)
        
        # Check injector I2 at angle 2π/3
        x, y = coords['I2']
        assert np.isclose(x, -500.0, rtol=1e-3)
        assert np.isclose(y, 1000 * np.sqrt(3)/2, rtol=1e-6)
    
    def test_distance_matrix(self):
        """Test distance matrix computation."""
        from wellfield.geometry import compute_distance_matrix
        
        positions = np.array([
            [0, 0],
            [1000, 0],
            [0, 1000],
        ])
        
        D = compute_distance_matrix(positions)
        
        assert D.shape == (3, 3)
        assert D[0, 0] == 0.0  # Self-distance
        assert np.isclose(D[0, 1], 1000.0)
        assert np.isclose(D[0, 2], 1000.0)
        assert np.isclose(D[1, 2], np.sqrt(2) * 1000)
    
    def test_minimum_spacing(self):
        """Test minimum spacing computation."""
        from wellfield.geometry import compute_minimum_spacing
        
        positions = np.array([
            [0, 0],
            [500, 0],
            [0, 600],
        ])
        
        min_spacing = compute_minimum_spacing(positions)
        assert np.isclose(min_spacing, 500.0)
    
    def test_constraint_check(self):
        """Test constraint checking."""
        from wellfield.geometry import check_geometry_constraints
        from wellfield.config import Config
        
        config = Config()
        
        # Valid configuration
        is_valid, violations, metrics = check_geometry_constraints(
            R_in=1000, R_out=2000, eps1=0, eps2=0, eps3=0, config=config
        )
        assert is_valid
        assert len(violations) == 0
        
        # Radial gap violation
        is_valid, violations, metrics = check_geometry_constraints(
            R_in=1000, R_out=1200, eps1=0, eps2=0, eps3=0, config=config
        )
        assert not is_valid
        assert any('Radial gap' in v for v in violations)


class TestHydraulics:
    """Test hydraulics module."""
    
    def test_fluid_properties(self):
        """Test CoolProp fluid property retrieval."""
        from wellfield.hydraulics import get_mean_fluid_properties
        from wellfield.config import Config
        
        config = Config()
        props = get_mean_fluid_properties(config)
        
        # Check reasonable values for CO2
        assert 1e-5 < props['mu'] < 1e-4  # Viscosity ~5e-5 Pa·s
        assert 300 < props['rho'] < 900    # Density ~600 kg/m³
        assert 1000 < props['cp'] < 5000   # Heat capacity ~2000 J/(kg·K)
    
    def test_pressure_cv(self):
        """Test pressure CV computation."""
        from wellfield.hydraulics import compute_cv_pressure
        from wellfield.geometry import compute_all_well_positions
        from wellfield.config import Config
        
        config = Config()
        _, inj_pos, prod_pos = compute_all_well_positions(1000, 2000, 0, 0, 0, 0)
        
        cv_inj, cv_prod = compute_cv_pressure(inj_pos, prod_pos, config)
        
        # CV should be non-negative
        assert cv_inj >= 0
        assert cv_prod >= 0
        
        # For symmetric layout, CV_inj should be low
        assert cv_inj < 0.1


class TestThermal:
    """Test thermal module."""
    
    def test_reservoir_volume(self):
        """Test frustum reservoir volume calculation."""
        from wellfield.thermal import compute_reservoir_volume
        from wellfield.config import Config
        
        config = Config()
        V = compute_reservoir_volume(1000, 2000, config)
        
        # V = (1/3) π H (R_out² + R_out*R_in + R_in²)
        expected = (1/3) * np.pi * config.H_THICK * (2000**2 + 2000*1000 + 1000**2)
        assert np.isclose(V, expected, rtol=1e-6)
    
    def test_thermal_lifetime(self):
        """Test thermal lifetime calculation."""
        from wellfield.thermal import compute_thermal_lifetime
        from wellfield.config import Config
        
        config = Config()
        tau = compute_thermal_lifetime(1000, 2000, config)
        
        # Lifetime should be positive
        assert tau > 0
        
        # Larger reservoir should have longer lifetime
        tau_large = compute_thermal_lifetime(1500, 3000, config)
        assert tau_large > tau


class TestBreakthrough:
    """Test breakthrough/TOF module."""
    
    def test_tof_simple(self):
        """Test simplified TOF calculation."""
        from wellfield.breakthrough import compute_tof_simple
        from wellfield.geometry import compute_all_well_positions
        from wellfield.config import Config
        
        config = Config()
        _, inj_pos, prod_pos = compute_all_well_positions(1000, 2000, 0, 0, 0, 0)
        
        t_bt, cv_tof = compute_tof_simple(inj_pos, prod_pos, config)
        
        # Should return 5 breakthrough times (one per producer)
        assert len(t_bt) == 5
        
        # CV should be non-negative
        assert cv_tof >= 0
        
        # Center producer (P0) should have shortest breakthrough time
        # since it's closest to injectors
        assert t_bt[0] == np.min(t_bt)


class TestObjective:
    """Test objective function module."""
    
    def test_objective_computation(self):
        """Test objective function returns finite value."""
        from wellfield.objective import compute_objective
        from wellfield.geometry import get_default_initial_guess
        from wellfield.config import Config
        
        config = Config()
        x0 = get_default_initial_guess(config)
        
        J = compute_objective(x0, config)
        
        # Should be finite
        assert np.isfinite(J)
    
    def test_objective_components(self):
        """Test objective function returns components."""
        from wellfield.objective import compute_objective
        from wellfield.geometry import get_default_initial_guess
        from wellfield.config import Config
        
        config = Config()
        x0 = get_default_initial_guess(config)
        
        result = compute_objective(x0, config, return_components=True)
        
        # Should have all expected keys
        assert 'J' in result
        assert 'CV_inj' in result
        assert 'CV_prod' in result
        assert 'CV_tof' in result
        assert 'tau_years' in result
        assert 'penalty' in result
        
        # Components should be finite
        assert np.isfinite(result['CV_inj'])
        assert np.isfinite(result['CV_prod'])
        assert np.isfinite(result['tau_years'])
    
    def test_objective_penalty(self):
        """Test that constraint violations result in penalty."""
        from wellfield.objective import compute_objective
        from wellfield.config import Config
        import numpy as np
        
        config = Config()
        
        # Valid configuration
        x_valid = np.array([1000, 2000, 0.5, 0, 0, 0])
        J_valid = compute_objective(x_valid, config)
        
        # Invalid: R_out too close to R_in
        x_invalid = np.array([1000, 1200, 0.5, 0, 0, 0])
        J_invalid = compute_objective(x_invalid, config)
        
        # Invalid should have much larger (more positive) objective
        assert J_invalid > J_valid


class TestOptimize:
    """Test optimization module."""
    
    def test_optimization_runs(self):
        """Test that optimization completes without error."""
        from wellfield.optimize import run_optimization
        from wellfield.config import Config
        
        # Quick optimization for testing
        config = Config()
        config.DE_MAXITER = 5
        config.DE_POPSIZE = 8
        
        x_best, info = run_optimization(config, verbose=False)
        
        # Should return 6 variables
        assert len(x_best) == 6
        
        # Should have info dict
        assert 'success' in info
        assert 'J_final' in info
        assert 'n_iterations' in info
    
    def test_optimization_improves(self):
        """Test that optimization improves objective."""
        from wellfield.optimize import run_optimization
        from wellfield.config import Config
        
        config = Config()
        config.DE_MAXITER = 15
        config.DE_POPSIZE = 10
        
        x_best, info = run_optimization(config, verbose=False)
        
        # Optimization should improve (reduce J)
        # Since J is negative for good solutions, improvement means more negative
        assert info['J_final'] <= info['J_initial']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
