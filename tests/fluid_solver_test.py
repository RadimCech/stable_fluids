import numpy as np
import pytest
from src.fluid_solver import FluidSolver


class TestFluidSolverInitialization:
    """Test the initialization of FluidSolver."""

    def test_init_basic(self):
        """Test basic initialization with valid parameters."""
        solver = FluidSolver(
            domain_size=1.0,
            n_points=32,
            dt=0.1,
            viscosity=0.001,
            diffusion_rate=0.0001
        )

        assert solver.domain_size == 1.0
        assert solver.n_points == 32
        assert solver.dt == 0.1
        assert solver.viscosity == 0.001
        assert solver.diffusion_rate == 0.0001

    def test_init_field_shapes(self):
        """Test that initialized fields have correct shapes."""
        n_points = 32
        solver = FluidSolver(
            domain_size=1.0,
            n_points=n_points,
            dt=0.1,
            viscosity=0.001,
            diffusion_rate=0.0001
        )

        assert solver.velocity_field.shape == (n_points, n_points, 2)
        assert solver.velocity_field_prev.shape == (n_points, n_points, 2)
        assert solver.forcing_field.shape == (n_points, n_points, 2)
        assert solver.density.shape == (n_points, n_points)
        assert solver.density_prev.shape == (n_points, n_points)

    def test_init_field_zeros(self):
        """Test that fields are initialized to zero."""
        solver = FluidSolver(
            domain_size=1.0,
            n_points=32,
            dt=0.1,
            viscosity=0.001,
            diffusion_rate=0.0001
        )

        assert np.all(solver.velocity_field == 0)
        assert np.all(solver.velocity_field_prev == 0)
        assert np.all(solver.forcing_field == 0)
        assert np.all(solver.density == 0)
        assert np.all(solver.density_prev == 0)

    def test_init_computed_values(self):
        """Test computed initialization values."""
        domain_size = 1.0
        n_points = 32
        solver = FluidSolver(
            domain_size=domain_size,
            n_points=n_points,
            dt=0.1,
            viscosity=0.001,
            diffusion_rate=0.0001
        )

        expected_element_length = domain_size / (n_points - 1)
        assert solver.element_length == pytest.approx(expected_element_length)
        assert solver.vector_field_dof == n_points**2 * 2


class TestAddForces:
    """Test the add_forces method."""

    def test_add_forces_zero_forcing(self):
        """Test that zero forcing field doesn't change velocity."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        initial_velocity = solver.velocity_field.copy()
        solver.add_forces()

        np.testing.assert_array_equal(solver.velocity_field, initial_velocity)

    def test_add_forces_uniform_forcing(self):
        """Test uniform forcing field application."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.forcing_field[:, :, 0] = 1.0  # Uniform force in x-direction
        solver.forcing_field[:, :, 1] = 2.0  # Uniform force in y-direction

        solver.add_forces()

        expected_vx = solver.dt * 1.0
        expected_vy = solver.dt * 2.0

        assert np.all(solver.velocity_field[:, :, 0] == pytest.approx(expected_vx))
        assert np.all(solver.velocity_field[:, :, 1] == pytest.approx(expected_vy))

    def test_add_forces_cumulative(self):
        """Test that forces accumulate properly over multiple calls."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.forcing_field[:, :, 0] = 1.0

        solver.add_forces()
        first_velocity = solver.velocity_field.copy()

        solver.add_forces()

        np.testing.assert_array_almost_equal(
            solver.velocity_field,
            first_velocity * 2
        )


class TestAdvection:
    """Test the advection method."""

    def test_advection_zero_velocity(self):
        """Test that advection with zero velocity doesn't move the field."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)

        # Create a simple density field
        solver.density[16, 16] = 1.0
        initial_density = solver.density.copy()

        # Zero velocity field
        velocity = np.zeros_like(solver.velocity_field)

        advected = solver.advection(solver.density, velocity)

        np.testing.assert_array_almost_equal(advected, initial_density, decimal=5)

    def test_advection_uniform_field(self):
        """Test advection of uniform field remains uniform."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)

        # Uniform density
        uniform_density = np.ones((32, 32))

        # Some velocity field
        velocity = np.random.rand(32, 32, 2) * 0.1

        advected = solver.advection(uniform_density, velocity)

        # Uniform field should remain uniform (within numerical precision)
        np.testing.assert_array_almost_equal(
            advected,
            np.ones((32, 32)),
            decimal=5
        )

    def test_advection_stays_in_bounds(self):
        """Test that advection keeps values within domain bounds."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)

        # Create field with some structure
        x = y = np.linspace(0, 1.0, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        density = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)

        # Large velocity
        velocity = np.ones((32, 32, 2)) * 10.0

        advected = solver.advection(density, velocity)

        # Should not have NaN or Inf values
        assert np.all(np.isfinite(advected))

    def test_advection_vector_field(self):
        """Test advection works with vector fields (velocity)."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)

        # Create a velocity field
        velocity_to_advect = np.random.rand(32, 32, 2) * 0.1
        advecting_velocity = np.random.rand(32, 32, 2) * 0.1

        # Should handle both components
        advected = solver.advection(velocity_to_advect, advecting_velocity)

        assert advected.shape == velocity_to_advect.shape
        assert np.all(np.isfinite(advected))


class TestDiffusion:
    """Test the diffusion method."""

    def test_diffusion_zero_coefficient(self):
        """Test that zero diffusion coefficient leaves field unchanged."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        # Create some field
        field = np.random.rand(32, 32)

        diffused = solver.diffusion(field, diffusion_coefficient=0.0)

        np.testing.assert_array_almost_equal(diffused, field, decimal=5)

    def test_diffusion_smooths_field(self):
        """Test that diffusion smooths out sharp features."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        # Create field with sharp peak
        field = np.zeros((32, 32))
        field[16, 16] = 100.0

        initial_variance = np.var(field)

        diffused = solver.diffusion(field, diffusion_coefficient=0.1)

        final_variance = np.var(diffused)

        # Diffusion should reduce variance (smoothing)
        assert final_variance < initial_variance

        # Mass should be approximately conserved
        assert np.sum(diffused) == pytest.approx(np.sum(field), rel=0.1)

    def test_diffusion_vector_field(self):
        """Test diffusion on vector fields."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        # Create vector field with sharp features
        field = np.zeros((32, 32, 2))
        field[16, 16, :] = [10.0, 10.0]

        diffused = solver.diffusion(field, diffusion_coefficient=0.1)

        assert diffused.shape == field.shape
        assert np.all(np.isfinite(diffused))

        # Peak should be reduced
        assert diffused[16, 16, 0] < field[16, 16, 0]


class TestProjection:
    """Test the projection method (divergence-free enforcement)."""

    def test_projection_divergence_free_unchanged(self):
        """Test that divergence-free field is unchanged by projection."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        # Create a simple divergence-free field (rotation)
        x = y = np.linspace(0, 1.0, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Circular flow around center
        center_x, center_y = 0.5, 0.5
        dx = X - center_x
        dy = Y - center_y

        velocity = np.zeros((32, 32, 2))
        velocity[:, :, 0] = -dy  # vx
        velocity[:, :, 1] = dx   # vy

        projected = solver.projection(velocity)

        # Should be very similar (divergence was already zero)
        np.testing.assert_array_almost_equal(projected, velocity, decimal=2)

    def test_projection_reduces_divergence(self):
        """Test that projection reduces divergence."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 200

        # Import divergence for testing
        from src.math_helper import divergence

        # Create field with divergence
        velocity = np.random.rand(32, 32, 2) * 0.1

        initial_div = divergence(velocity, solver.element_length)
        initial_div_norm = np.linalg.norm(initial_div)

        projected = solver.projection(velocity)

        final_div = divergence(projected, solver.element_length)
        final_div_norm = np.linalg.norm(final_div)

        # Divergence should be significantly reduced
        assert final_div_norm < initial_div_norm * 0.1

    def test_projection_output_shape(self):
        """Test that projection maintains field shape."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        velocity = np.random.rand(32, 32, 2)
        projected = solver.projection(velocity)

        assert projected.shape == velocity.shape


class TestStepMethod:
    """Test the complete step method."""

    def test_step_executes(self):
        """Test that step method executes without errors."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 50

        # Add some initial conditions
        solver.density[16, 16] = 1.0
        solver.forcing_field[16, 16, :] = [0.1, 0.1]

        # Should execute without error
        solver.step()

        assert np.all(np.isfinite(solver.velocity_field))
        assert np.all(np.isfinite(solver.density))

    def test_step_updates_prev_fields(self):
        """Test that step updates previous fields correctly."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 50

        solver.density[16, 16] = 1.0
        solver.velocity_field[16, 16, :] = [0.1, 0.1]

        solver.step()

        # Previous fields should match current fields after step
        np.testing.assert_array_equal(
            solver.velocity_field_prev,
            solver.velocity_field
        )
        np.testing.assert_array_equal(
            solver.density_prev,
            solver.density
        )

    def test_step_sequence(self):
        """Test that multiple steps execute correctly."""
        solver = FluidSolver(1.0, 32, 0.05, 0.001, 0.0001)
        solver.max_iter = 50

        # Add continuous forcing
        solver.forcing_field[16, 16, :] = [1.0, 0.0]
        solver.density[16, 16] = 1.0

        # Run multiple steps
        for _ in range(5):
            solver.step()

            # Fields should remain valid
            assert np.all(np.isfinite(solver.velocity_field))
            assert np.all(np.isfinite(solver.density))

    def test_step_mass_conservation(self):
        """Test that total density is approximately conserved."""
        solver = FluidSolver(1.0, 32, 0.01, 0.0, 0.0)  # No diffusion
        solver.max_iter = 50

        # Add some density
        solver.density[16, 16] = 100.0
        initial_mass = np.sum(solver.density)

        # Run several steps
        for _ in range(3):
            solver.step()

        final_mass = np.sum(solver.density)

        # Mass should be approximately conserved (within numerical error)
        assert final_mass == pytest.approx(initial_mass, rel=0.15)

    def test_step_with_forcing(self):
        """Test that forcing field affects velocity."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 50

        # Add forcing in x-direction
        solver.forcing_field[:, :, 0] = 1.0

        solver.step()

        # Velocity should be non-zero after forcing
        assert np.any(solver.velocity_field[:, :, 0] > 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_timestep(self):
        """Test with very small timestep."""
        solver = FluidSolver(1.0, 16, dt=0.001, viscosity=0.001, diffusion_rate=0.0001)
        solver.max_iter = 50
        solver.density[8, 8] = 1.0

        solver.step()

        assert np.all(np.isfinite(solver.density))
        assert np.all(np.isfinite(solver.velocity_field))

    def test_large_viscosity(self):
        """Test with large viscosity."""
        solver = FluidSolver(1.0, 16, dt=0.01, viscosity=1.0, diffusion_rate=0.0001)
        solver.max_iter = 100

        solver.velocity_field[8, 8, :] = [10.0, 10.0]
        solver.step()

        # High viscosity should smooth velocity
        assert np.all(np.isfinite(solver.velocity_field))

    def test_small_grid(self):
        """Test with small grid size."""
        solver = FluidSolver(1.0, 8, 0.1, 0.001, 0.0001)
        solver.max_iter = 50

        solver.density[4, 4] = 1.0
        solver.forcing_field[4, 4, :] = [0.1, 0.1]

        solver.step()

        assert np.all(np.isfinite(solver.density))
        assert np.all(np.isfinite(solver.velocity_field))


class TestNumericalStability:
    """Test numerical stability of the solver."""

    def test_no_nan_or_inf(self):
        """Test that solver doesn't produce NaN or Inf values."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 50

        # Add extreme values
        solver.density[16, 16] = 1000.0
        solver.forcing_field[15:18, 15:18, :] = 10.0

        for _ in range(10):
            solver.step()

            assert np.all(np.isfinite(solver.velocity_field))
            assert np.all(np.isfinite(solver.density))

    def test_energy_dissipation(self):
        """Test that energy dissipates over time with viscosity."""
        solver = FluidSolver(1.0, 32, 0.01, 0.01, 0.0)
        solver.max_iter = 50

        # Add initial velocity
        solver.velocity_field[:, :, :] = np.random.rand(32, 32, 2) * 0.5
        initial_energy = np.sum(solver.velocity_field**2)

        # Run without forcing
        for _ in range(10):
            solver.step()

        final_energy = np.sum(solver.velocity_field**2)

        # Energy should decrease due to viscosity
        assert final_energy < initial_energy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
