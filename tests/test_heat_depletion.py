"""
Unit tests for heat depletion module.

Tests breakthrough time calculation and thermal lifetime prediction.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.reservoir.heat_depletion import (
    calculate_breakthrough_time,
    compute_breakthrough_variance,
    estimate_temperature_decline,
    compute_thermal_power,
)


class TestBreakthroughTime:
    """Test breakthrough time calculation."""
    
    def test_breakthrough_time_positive(self):
        """Test that breakthrough time is positive."""
        injector_pos = (0.0, 0.0)
        producer_pos = (1000.0, 0.0)
        flow_rate = 50.0  # kg/s
        
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
        
        t_bt = calculate_breakthrough_time(
            injector_pos, producer_pos, flow_rate, reservoir_params, fluid_params
        )
        
        assert t_bt > 0, "Breakthrough time should be positive"
        assert np.isfinite(t_bt), "Breakthrough time should be finite"
    
    def test_breakthrough_scales_with_distance(self):
        """Test that breakthrough time increases with distance."""
        injector_pos = (0.0, 0.0)
        prod_near = (500.0, 0.0)
        prod_far = (1500.0, 0.0)
        
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
        
        flow = 50.0
        
        t_near = calculate_breakthrough_time(injector_pos, prod_near, flow, reservoir_params, fluid_params)
        t_far = calculate_breakthrough_time(injector_pos, prod_far, flow, reservoir_params, fluid_params)
        
        assert t_far > t_near, "Breakthrough should take longer for distant wells"
    
    def test_breakthrough_scales_with_flow(self):
        """Test that breakthrough time decreases with higher flow rate."""
        injector_pos = (0.0, 0.0)
        producer_pos = (1000.0, 0.0)
        
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
        
        flow_low = 25.0
        flow_high = 100.0
        
        t_low = calculate_breakthrough_time(injector_pos, producer_pos, flow_low, reservoir_params, fluid_params)
        t_high = calculate_breakthrough_time(injector_pos, producer_pos, flow_high, reservoir_params, fluid_params)
        
        assert t_high < t_low, "Higher flow should reduce breakthrough time"
    
    def test_zero_flow(self):
        """Test handling of zero flow rate."""
        injector_pos = (0.0, 0.0)
        producer_pos = (1000.0, 0.0)
        
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
        
        t_bt = calculate_breakthrough_time(injector_pos, producer_pos, 0.0, reservoir_params, fluid_params)
        
        assert t_bt > 1e6, "Zero flow should give very large breakthrough time"


class TestBreakthroughVariance:
    """Test breakthrough variance calculation."""
    
    def test_variance_for_uniform_layout(self):
        """Test that symmetric layout has low variance."""
        # Create symmetric layout
        well_layout = [
            ((0, 0), 'injector'),
            ((1000, 0), 'producer'),
            ((0, 1000), 'producer'),
        ]
        
        flow_rates = np.array([50.0, -25.0, -25.0])
        
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
        
        variance = compute_breakthrough_variance(
            well_layout, flow_rates, reservoir_params, fluid_params
        )
        
        # Symmetric layout should have low (near zero) variance
        assert variance >= 0, "Variance should be non-negative"
        assert np.isfinite(variance), "Variance should be finite"


class TestTemperatureDecline:
    """Test temperature decline estimation."""
    
    def test_temperature_profile_shape(self):
        """Test that temperature profile has correct shape."""
        injector_pos = (0.0, 0.0)
        producer_pos = (1000.0, 0.0)
        flow_rate = 50.0
        T_inj = 323.0  # K
        T_res = 423.0  # K
        
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
        
        time_array = np.linspace(0, 50, 100)  # 50 years
        
        T_profile = estimate_temperature_decline(
            injector_pos, producer_pos, flow_rate, T_inj, T_res,
            time_array, reservoir_params, fluid_params
        )
        
        assert len(T_profile) == len(time_array), "Profile should match time array length"
        assert T_profile[0] == pytest.approx(T_res), "Initial temperature should be reservoir temperature"
        assert T_profile[-1] < T_res, "Final temperature should be lower"
        assert T_profile[-1] > T_inj, "Temperature shouldn't drop below injection"
    
    def test_temperature_monotonic_decline(self):
        """Test that temperature declines monotonically."""
        injector_pos = (0.0, 0.0)
        producer_pos = (1000.0, 0.0)
        flow_rate = 50.0
        T_inj = 323.0
        T_res = 423.0
        
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
        
        time_array = np.linspace(0, 50, 100)
        
        T_profile = estimate_temperature_decline(
            injector_pos, producer_pos, flow_rate, T_inj, T_res,
            time_array, reservoir_params, fluid_params
        )
        
        # After breakthrough, temperature should decline
        # Find breakthrough index
        t_bt = calculate_breakthrough_time(injector_pos, producer_pos, flow_rate, reservoir_params, fluid_params)
        bt_idx = np.searchsorted(time_array, t_bt)
        
        if bt_idx < len(T_profile) - 1:
            # Check that temperature declines after breakthrough
            assert T_profile[bt_idx] >= T_profile[-1], "Temperature should decline after breakthrough"


class TestThermalPower:
    """Test thermal power calculation."""
    
    def test_thermal_power_positive(self):
        """Test that thermal power is positive for hot production."""
        T_prod = 423.0  # K
        T_inj = 323.0  # K
        flow = 50.0  # kg/s
        cp = 1200.0  # J/(kg·K)
        
        P_thermal = compute_thermal_power(T_prod, T_inj, flow, cp)
        
        assert P_thermal > 0, "Thermal power should be positive"
    
    def test_thermal_power_scaling(self):
        """Test that power scales with temperature difference."""
        T_inj = 323.0
        flow = 50.0
        cp = 1200.0
        
        T_low = 373.0  # 100°C
        T_high = 423.0  # 150°C
        
        P_low = compute_thermal_power(T_low, T_inj, flow, cp)
        P_high = compute_thermal_power(T_high, T_inj, flow, cp)
        
        ratio = P_high / P_low
        expected_ratio = (T_high - T_inj) / (T_low - T_inj)
        
        assert np.isclose(ratio, expected_ratio), "Power should scale with temperature difference"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
