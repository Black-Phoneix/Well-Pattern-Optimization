"""
Unit tests for wellbore hydraulics modules.

Tests friction factor calculations and wellbore integration.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wellbore.friction import (
    friction_factor_laminar,
    friction_factor_turbulent,
    colebrook_white,
    reynolds_number,
    pressure_drop_darcy_weisbach,
)
from src.wellbore.integrator import (
    integrate_wellbore,
    compute_thermosiphon_effect,
    calculate_pumping_power,
)


class TestFrictionFactor:
    """Test friction factor calculations."""
    
    def test_laminar_friction(self):
        """Test laminar friction factor f = 64/Re."""
        Re = 1000
        f = friction_factor_laminar(Re)
        expected = 64.0 / Re
        
        assert np.isclose(f, expected), f"Expected {expected}, got {f}"
    
    def test_turbulent_friction_smooth(self):
        """Test turbulent friction for smooth pipe."""
        Re = 1e5
        epsilon = 0.0  # Smooth pipe
        D = 0.2
        
        f = friction_factor_turbulent(Re, epsilon, D)
        
        # For smooth pipes at Re=1e5, f ≈ 0.018
        assert 0.015 < f < 0.025, f"Friction factor {f} out of expected range"
    
    def test_turbulent_friction_rough(self):
        """Test turbulent friction for rough pipe."""
        Re = 1e5
        epsilon = 0.045e-3  # Commercial steel
        D = 0.2
        
        f = friction_factor_turbulent(Re, epsilon, D)
        
        # Should be slightly higher than smooth pipe
        f_smooth = friction_factor_turbulent(Re, 0.0, D)
        assert f > f_smooth, "Rough pipe should have higher friction"
    
    def test_colebrook_regime_selection(self):
        """Test that colebrook_white selects correct regime."""
        epsilon = 0.045e-3
        D = 0.2
        
        # Laminar
        Re_lam = 1000
        f_lam = colebrook_white(Re_lam, epsilon, D)
        assert np.isclose(f_lam, 64.0/Re_lam)
        
        # Turbulent
        Re_turb = 1e5
        f_turb = colebrook_white(Re_turb, epsilon, D)
        assert f_turb < 0.1  # Reasonable turbulent value
    
    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        rho = 600.0  # kg/m³
        v = 2.0  # m/s
        D = 0.2  # m
        mu = 5e-5  # Pa·s
        
        Re = reynolds_number(rho, v, D, mu)
        expected = (rho * v * D) / mu
        
        assert np.isclose(Re, expected)


class TestDarcyWeisbach:
    """Test Darcy-Weisbach pressure drop."""
    
    def test_pressure_drop_positive(self):
        """Test that pressure drop is positive."""
        f = 0.02
        L = 1000.0  # m
        D = 0.2  # m
        rho = 600.0  # kg/m³
        v = 2.0  # m/s
        
        dP = pressure_drop_darcy_weisbach(f, L, D, rho, v)
        
        assert dP > 0, "Pressure drop should be positive"
    
    def test_pressure_drop_scaling(self):
        """Test that pressure drop scales with velocity squared."""
        f, L, D, rho = 0.02, 1000.0, 0.2, 600.0
        
        v1 = 1.0
        v2 = 2.0
        
        dP1 = pressure_drop_darcy_weisbach(f, L, D, rho, v1)
        dP2 = pressure_drop_darcy_weisbach(f, L, D, rho, v2)
        
        ratio = dP2 / dP1
        expected_ratio = (v2 / v1) ** 2
        
        assert np.isclose(ratio, expected_ratio), f"Expected ratio {expected_ratio}, got {ratio}"


class TestWellboreIntegration:
    """Test wellbore integration."""
    
    def test_pressure_increases_with_depth(self):
        """Test that pressure increases with depth (injection)."""
        wellbore_params = {
            'diameter': 0.2,
            'roughness': 0.045e-3,
            'geothermal_gradient': 0.03,
        }
        
        P_surf = 15e6  # Pa
        T_surf = 323.0  # K
        
        P_bottom, T_bottom = integrate_wellbore(
            depth=3000.0,
            flow_rate=50.0,
            T_surface=T_surf,
            P_surface=P_surf,
            wellbore_params=wellbore_params,
            direction='down',
        )
        
        assert P_bottom > P_surf, "Pressure should increase with depth"
        assert T_bottom > T_surf, "Temperature should increase with depth"
    
    def test_temperature_gradient(self):
        """Test geothermal gradient effect."""
        wellbore_params = {
            'diameter': 0.2,
            'roughness': 0.045e-3,
            'geothermal_gradient': 0.03,
        }
        
        depth = 3000.0
        grad = wellbore_params['geothermal_gradient']
        
        P_bottom, T_bottom = integrate_wellbore(
            depth=depth,
            flow_rate=50.0,
            T_surface=323.0,
            P_surface=15e6,
            wellbore_params=wellbore_params,
            direction='down',
        )
        
        # Temperature increase should be approximately depth * gradient
        dT = T_bottom - 323.0
        expected_dT = depth * grad
        
        # Allow some deviation due to numerical integration
        assert 0.5 * expected_dT < dT < 1.5 * expected_dT, \
            f"Temperature increase {dT:.1f} K not close to expected {expected_dT:.1f} K"


class TestThermosiphon:
    """Test thermosiphon effect calculation."""
    
    def test_thermosiphon_positive(self):
        """Test that cold injection creates positive thermosiphon."""
        injector_params = {
            'density': 600.0,  # Cold, dense
            'temperature': 323.0,
        }
        
        producer_params = {
            'density': 500.0,  # Hot, less dense
            'temperature': 423.0,
        }
        
        depth = 3000.0
        dP = compute_thermosiphon_effect(injector_params, producer_params, depth)
        
        assert dP > 0, "Thermosiphon should be positive (aids circulation)"
    
    def test_thermosiphon_scaling(self):
        """Test that thermosiphon scales with depth."""
        injector_params = {'density': 600.0, 'temperature': 323.0}
        producer_params = {'density': 500.0, 'temperature': 423.0}
        
        d1 = 1000.0
        d2 = 3000.0
        
        dP1 = compute_thermosiphon_effect(injector_params, producer_params, d1)
        dP2 = compute_thermosiphon_effect(injector_params, producer_params, d2)
        
        ratio = dP2 / dP1
        expected_ratio = d2 / d1
        
        assert np.isclose(ratio, expected_ratio), "Thermosiphon should scale linearly with depth"


class TestPumpingPower:
    """Test pumping power calculation."""
    
    def test_pumping_power_positive(self):
        """Test that pumping power is positive."""
        dP = 5e6  # Pa
        flow = 50.0  # kg/s
        
        P_pump = calculate_pumping_power(dP, flow)
        
        assert P_pump > 0, "Pumping power should be positive"
    
    def test_pumping_power_scaling(self):
        """Test that power scales with pressure and flow."""
        dP = 5e6
        flow1 = 50.0
        flow2 = 100.0
        
        P1 = calculate_pumping_power(dP, flow1)
        P2 = calculate_pumping_power(dP, flow2)
        
        ratio = P2 / P1
        expected_ratio = flow2 / flow1
        
        assert np.isclose(ratio, expected_ratio, rtol=0.01), \
            "Power should scale linearly with flow rate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
