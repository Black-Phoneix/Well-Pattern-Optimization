"""
Unit tests for the pressure-only allocation model.

Tests the models/pressure_only.py module for:
1. Impedance computation
2. Pressure solution under equal-rate constraint
3. Validation functions
4. Symmetry properties for symmetric geometries
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.pressure_only import (
    compute_pairwise_impedance,
    compute_distance_matrix_from_arrays,
    impedance_doublet,
    solve_producer_bhp_equal_rate,
    validate_solution,
    validate_total_mass_balance,
    solve_pressure_allocation,
    optimize_outer_producer_ring,
    pressure_drop_variance,
)
from patterns.geometry import (
    Well,
    generate_center_ring_pattern,
    generate_ring_pattern,
)


class TestDistanceMatrix:
    """Test distance matrix computation."""

    def test_single_pair(self):
        """Test distance between single injector and producer."""
        inj_xy = np.array([[0.0, 0.0]])
        prod_xy = np.array([[3.0, 4.0]])

        D = compute_distance_matrix_from_arrays(inj_xy, prod_xy)

        assert D.shape == (1, 1)
        assert np.isclose(D[0, 0], 5.0)  # 3-4-5 triangle

    def test_multiple_wells(self):
        """Test distance matrix with multiple wells."""
        inj_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
        prod_xy = np.array([[0.0, 100.0], [100.0, 100.0], [50.0, 50.0]])

        D = compute_distance_matrix_from_arrays(inj_xy, prod_xy)

        assert D.shape == (3, 2)
        # Producer 0 to Injector 0: (0,100) - (0,0) = 100
        assert np.isclose(D[0, 0], 100.0)
        # Producer 0 to Injector 1: (0,100) - (100,0) = sqrt(100^2 + 100^2)
        assert np.isclose(D[0, 1], np.sqrt(2) * 100.0)
        # Producer 2 to both injectors: (50,50) - (0,0) = (50,50) - (100,0) = sqrt(50^2+50^2)
        assert np.isclose(D[2, 0], np.sqrt(2) * 50.0)

    def test_symmetry_with_geometry_module(self):
        """Test that our distance matrix matches the geometry module's."""
        # Create wells using geometry module
        injectors = [Well(0.0, 0.0, 'injector'), Well(100.0, 0.0, 'injector')]
        producers = [Well(50.0, 86.6, 'producer')]

        # Compute using geometry module
        from patterns.geometry import distance_matrix as geom_dm
        D_geom = geom_dm(injectors, producers)

        # Compute using our function
        inj_xy = np.array([[w.x, w.y] for w in injectors])
        prod_xy = np.array([[w.x, w.y] for w in producers])
        D_ours = compute_distance_matrix_from_arrays(inj_xy, prod_xy)

        np.testing.assert_allclose(D_ours, D_geom, rtol=1e-10)


class TestImpedanceDoublet:
    """Test the impedance doublet formula."""

    def test_impedance_positive(self):
        """Test that impedance is positive for valid inputs."""
        mu, rho, k, b, rw = 1e-3, 1000.0, 1e-13, 50.0, 0.1
        L = 1000.0

        Z = impedance_doublet(mu, rho, k, b, L, rw)

        assert Z > 0
        assert np.isfinite(Z)

    def test_impedance_scaling_with_distance(self):
        """Test that impedance increases with distance (logarithmically)."""
        mu, rho, k, b, rw = 1e-3, 1000.0, 1e-13, 50.0, 0.1

        Z1 = impedance_doublet(mu, rho, k, b, 100.0, rw)
        Z2 = impedance_doublet(mu, rho, k, b, 1000.0, rw)

        assert Z2 > Z1  # Farther distance = higher impedance

    def test_impedance_scaling_with_permeability(self):
        """Test that impedance decreases with permeability."""
        mu, rho, b, rw = 1e-3, 1000.0, 50.0, 0.1
        L = 500.0

        Z1 = impedance_doublet(mu, rho, 1e-14, b, L, rw)
        Z2 = impedance_doublet(mu, rho, 1e-13, b, L, rw)

        assert Z1 > Z2  # Lower permeability = higher impedance
        assert np.isclose(Z1 / Z2, 10.0, rtol=1e-10)  # 10x permeability = 1/10 impedance

    def test_impedance_array_input(self):
        """Test that impedance works with array inputs."""
        mu, rho, k, b, rw = 1e-3, 1000.0, 1e-13, 50.0, 0.1
        L = np.array([100.0, 500.0, 1000.0])

        Z = impedance_doublet(mu, rho, k, b, L, rw)

        assert Z.shape == (3,)
        assert np.all(Z > 0)
        assert Z[0] < Z[1] < Z[2]  # Increasing with distance

    def test_impedance_clamping(self):
        """Test that L < rw is clamped to rw."""
        mu, rho, k, b, rw = 1e-3, 1000.0, 1e-13, 50.0, 0.1

        Z_small = impedance_doublet(mu, rho, k, b, 0.01, rw)  # L < rw
        Z_at_rw = impedance_doublet(mu, rho, k, b, rw, rw)

        # Both should give Z at rw (ln(1) = 0, so Z = 0)
        assert np.isclose(Z_small, Z_at_rw)


class TestComputePairwiseImpedance:
    """Test the pairwise impedance computation."""

    def test_basic_computation(self):
        """Test basic impedance matrix computation."""
        inj_xy = np.array([[0.0, 0.0]])
        prod_xy = np.array([[500.0, 0.0]])
        params = {'mu': 1e-3, 'rho': 1000.0, 'k': 1e-13, 'b': 50.0, 'rw': 0.1}

        Z = compute_pairwise_impedance(inj_xy, prod_xy, params)

        assert Z.shape == (1, 1)
        assert Z[0, 0] > 0
        assert np.isfinite(Z[0, 0])

    def test_shape_3inj_5prod(self):
        """Test correct shape for 3 injector / 5 producer layout."""
        inj_xy = np.array([[0, 0], [100, 0], [50, 86.6]])
        prod_xy = np.array([[50, 0], [25, 43.3], [75, 43.3], [0, 50], [100, 50]])
        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}

        Z = compute_pairwise_impedance(inj_xy, prod_xy, params)

        assert Z.shape == (5, 3)
        assert np.all(Z > 0)
        assert np.all(np.isfinite(Z))


class TestSolveProducerBHP:
    """Test the pressure solution under equal-rate constraint."""

    def test_single_pair(self):
        """Test solution for single injector-producer pair."""
        P_inj = 30e6  # 30 MPa
        q_prod = 10.0  # kg/s
        Z = np.array([[1e6]])  # Pa/(kg/s)

        P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(P_inj, q_prod, Z)

        # P_prod = P_inj - q_prod / (1/Z) = P_inj - q_prod * Z
        expected_P_prod = P_inj - q_prod * Z[0, 0]
        assert np.isclose(P_prod[0], expected_P_prod)
        assert np.isclose(q_ij[0, 0], q_prod)
        assert np.isclose(q_inj[0], q_prod)

    def test_two_injectors_one_producer(self):
        """Test with 2 injectors and 1 producer."""
        P_inj = 30e6
        q_prod = 10.0
        Z = np.array([[1e6, 2e6]])  # Two different impedances

        P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(P_inj, q_prod, Z)

        # Check mass balance at producer
        assert np.isclose(np.sum(q_ij[0, :]), q_prod)

        # Check injector flows
        assert np.isclose(np.sum(q_inj), q_prod)

        # Lower impedance path should have higher flow
        assert q_ij[0, 0] > q_ij[0, 1]

    def test_mass_balance_3inj_5prod(self):
        """Test mass balance for 3 injector / 5 producer layout."""
        P_inj = 30e6
        q_prod = 10.0

        # Random but valid impedance matrix
        np.random.seed(42)
        Z = np.random.uniform(0.5e6, 2e6, size=(5, 3))

        P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(P_inj, q_prod, Z)

        # Check mass balance at each producer
        for i in range(5):
            assert np.isclose(np.sum(q_ij[i, :]), q_prod, rtol=1e-10)

        # Check total mass balance
        assert np.isclose(np.sum(q_inj), 5 * q_prod, rtol=1e-10)

    def test_all_pressures_below_injection(self):
        """Test that all producer pressures are below injection pressure."""
        P_inj = 30e6
        q_prod = 10.0
        Z = np.array([[1e6, 1.5e6, 2e6],
                      [1.2e6, 1e6, 1.8e6],
                      [1.3e6, 1.4e6, 1e6]])

        P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(P_inj, q_prod, Z)

        assert np.all(P_prod < P_inj)

    def test_symmetric_geometry_equal_pressures(self):
        """Test that symmetric geometry produces equal pressures for symmetric producers.
        
        With 3 injectors at 120° spacing and 4 outer producers at 90° spacing,
        the layout has reflection symmetry about the x-axis. Therefore:
        - Producer 2 (0, 300) and Producer 4 (0, -300) are symmetric
        - Producer 1 (300, 0) and Producer 3 (-300, 0) are NOT symmetric
          because Producer 1 is closer to one injector
        """
        injectors, producers = generate_center_ring_pattern(
            n_inj=3,
            n_prod_outer=4,
            R_inj=600.0,
            R_prod=300.0,
            phi_inj0=0.0,
            phi_prod0=0.0,
        )

        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}

        inj_xy = np.array([[w.x, w.y] for w in injectors])
        prod_xy = np.array([[w.x, w.y] for w in producers])

        Z = compute_pairwise_impedance(inj_xy, prod_xy, params)
        P_prod, q_ij, q_inj = solve_producer_bhp_equal_rate(30e6, 10.0, Z)

        # Producers symmetric about x-axis should have equal pressures
        # Producer indices: 0=center, 1=(300,0), 2=(0,300), 3=(-300,0), 4=(0,-300)
        # Producer 2 and 4 are symmetric about x-axis
        assert np.isclose(P_prod[2], P_prod[4], rtol=1e-6)

    def test_invalid_impedance_raises(self):
        """Test that invalid impedance values raise errors."""
        P_inj = 30e6
        q_prod = 10.0

        # Zero impedance
        with pytest.raises(ValueError, match="non-positive"):
            solve_producer_bhp_equal_rate(P_inj, q_prod, np.array([[0.0]]))

        # Negative impedance
        with pytest.raises(ValueError, match="non-positive"):
            solve_producer_bhp_equal_rate(P_inj, q_prod, np.array([[-1e6]]))


class TestValidateSolution:
    """Test the validation functions."""

    def test_valid_solution_passes(self):
        """Test that a valid solution passes validation."""
        q_ij = np.array([[5.0, 5.0],
                         [3.0, 7.0],
                         [6.0, 4.0]])
        q_prod = 10.0

        # Should not raise
        validate_solution(q_ij, q_prod, tol=1e-6)

    def test_invalid_mass_balance_fails(self):
        """Test that invalid mass balance raises error."""
        q_ij = np.array([[5.0, 5.0],
                         [3.0, 6.0]])  # Sum = 9, not 10
        q_prod = 10.0

        with pytest.raises(ValueError, match="Mass balance violated"):
            validate_solution(q_ij, q_prod, tol=1e-6)

    def test_negative_flow_fails(self):
        """Test that negative flow raises error."""
        q_ij = np.array([[5.0, 5.0],
                         [-1.0, 11.0]])  # Negative flow
        q_prod = 10.0

        with pytest.raises(ValueError, match="Negative"):
            validate_solution(q_ij, q_prod, tol=1e-6)

    def test_total_mass_balance_passes(self):
        """Test that valid total mass balance passes."""
        q_inj = np.array([20.0, 15.0, 15.0])  # Total = 50
        q_prod = 10.0
        n_prod = 5  # Total = 50

        # Should not raise
        validate_total_mass_balance(q_inj, q_prod, n_prod, tol=1e-6)

    def test_total_mass_balance_fails(self):
        """Test that invalid total mass balance fails."""
        q_inj = np.array([20.0, 15.0, 10.0])  # Total = 45
        q_prod = 10.0
        n_prod = 5  # Expected total = 50

        with pytest.raises(ValueError, match="Total mass balance violated"):
            validate_total_mass_balance(q_inj, q_prod, n_prod, tol=1e-6)


class TestSolvePressureAllocation:
    """Test the high-level API."""

    def test_solve_allocation_returns_correct_keys(self):
        """Test that solve_pressure_allocation returns correct dictionary keys."""
        injectors, producers = generate_center_ring_pattern(
            n_inj=3, n_prod_outer=4, R_inj=600.0, R_prod=300.0
        )
        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}

        result = solve_pressure_allocation(injectors, producers, 30e6, 10.0, params)

        expected_keys = {'P_prod', 'q_ij', 'q_inj', 'Z', 'injectors', 'producers'}
        assert set(result.keys()) == expected_keys

    def test_solve_allocation_shapes(self):
        """Test that solve_pressure_allocation returns correct shapes."""
        injectors, producers = generate_center_ring_pattern(
            n_inj=3, n_prod_outer=4, R_inj=600.0, R_prod=300.0
        )
        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}

        result = solve_pressure_allocation(injectors, producers, 30e6, 10.0, params)

        assert result['P_prod'].shape == (5,)
        assert result['q_ij'].shape == (5, 3)
        assert result['q_inj'].shape == (3,)
        assert result['Z'].shape == (5, 3)

    def test_solve_allocation_validation_enabled(self):
        """Test that validation runs when enabled."""
        injectors, producers = generate_center_ring_pattern(
            n_inj=3, n_prod_outer=4, R_inj=600.0, R_prod=300.0
        )
        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}

        # Should not raise with validation enabled
        result = solve_pressure_allocation(
            injectors, producers, 30e6, 10.0, params, validate=True
        )

        # Check mass balance manually
        for i in range(5):
            assert np.isclose(np.sum(result['q_ij'][i, :]), 10.0, rtol=1e-10)


class TestNonSymmetricGeometry:
    """Test that the model works for non-symmetric geometries."""

    def test_arbitrary_coordinates(self):
        """Test with completely arbitrary well coordinates."""
        # Arbitrary non-symmetric layout
        injectors = [
            Well(100.0, 200.0, 'injector'),
            Well(-150.0, 50.0, 'injector'),
            Well(300.0, -100.0, 'injector'),
        ]
        producers = [
            Well(0.0, 0.0, 'producer'),
            Well(50.0, 150.0, 'producer'),
            Well(-75.0, -50.0, 'producer'),
            Well(200.0, 75.0, 'producer'),
            Well(-100.0, 175.0, 'producer'),
        ]

        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}

        result = solve_pressure_allocation(injectors, producers, 30e6, 10.0, params)

        # Check that solution is valid
        assert np.all(result['P_prod'] < 30e6)  # All pressures below injection
        assert np.all(result['q_ij'] >= 0)  # All flows non-negative

        # Check mass balance
        for i in range(5):
            assert np.isclose(np.sum(result['q_ij'][i, :]), 10.0, rtol=1e-10)
        assert np.isclose(np.sum(result['q_inj']), 50.0, rtol=1e-10)

    def test_different_pressures_non_symmetric(self):
        """Test that non-symmetric geometry produces different producer pressures."""
        # Offset outer producers to break symmetry
        injectors, producers = generate_center_ring_pattern(
            n_inj=3,
            n_prod_outer=4,
            R_inj=600.0,
            R_prod=300.0,
            phi_inj0=0.0,
            phi_prod0=np.pi / 7,  # Arbitrary offset breaks symmetry
        )

        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}
        result = solve_pressure_allocation(injectors, producers, 30e6, 10.0, params)

        # Outer producers should NOT all have the same pressure
        outer_pressures = result['P_prod'][1:]
        # With non-symmetric geometry, pressures should vary
        assert np.std(outer_pressures) > 1e-3  # Some variation expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestProducerCoordinateOptimization:
    """Test producer-coordinate optimization with fixed injectors."""

    def test_optimize_outer_ring_reduces_pressure_drop_variance(self):
        """Optimizer should not be worse than a baseline ring layout."""
        injectors, producers = generate_center_ring_pattern(
            n_inj=3,
            n_prod_outer=4,
            R_inj=300.0,
            R_prod=600.0,
            phi_inj0=0.0,
            phi_prod0=np.pi / 4,
        )

        params = {'mu': 5e-5, 'rho': 800.0, 'k': 5e-14, 'b': 300.0, 'rw': 0.1}
        P_inj = 30e6
        q_total = 126.8
        q_prod = q_total / 5.0

        inj_xy = np.array([[w.x, w.y] for w in injectors])
        prod_xy_baseline = np.array([[w.x, w.y] for w in producers])

        Z_base = compute_pairwise_impedance(inj_xy, prod_xy_baseline, params)
        P_base, _, _ = solve_producer_bhp_equal_rate(P_inj, q_prod, Z_base)
        var_base = pressure_drop_variance(P_inj, P_base)

        opt = optimize_outer_producer_ring(
            inj_xy=inj_xy,
            P_inj=P_inj,
            q_prod=q_prod,
            params=params,
            R_inj=300.0,
            R_prod_bounds=(350.0, 1200.0),
            n_outer=4,
            n_radius_samples=20,
            n_angle_trials=800,
            min_angle_deg=10.0,
            random_seed=7,
        )

        assert opt['prod_xy'].shape == (5, 2)
        assert float(opt['R_prod']) > 300.0
        assert float(opt['variance_dP']) <= var_base + 1e-12
        assert float(opt['min_angle_deg_achieved']) >= 10.0

        # Check mass-balance constraints are still met
        for i in range(5):
            assert np.isclose(np.sum(opt['q_ij'][i, :]), q_prod, rtol=1e-10)
        assert np.isclose(np.sum(opt['q_inj']), q_total, rtol=1e-10)
