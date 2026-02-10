import numpy as np
import pytest
from src.fluid_solver import FluidSolver
from src.math_helper import divergence


class TestFluidSolverInitialization:
    """Test the initialization of FluidSolver."""

class TestAddForces:
    """Test the add_forces method."""

    def test_add_forces_uniform_forcing(self):
        """Test uniform forcing field application."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.forcing_field[:, :, 0] = 1.0
        solver.forcing_field[:, :, 1] = 2.0

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

        solver.density[16, 16] = 1.0
        initial_density = solver.density.copy()

        velocity = np.zeros_like(solver.velocity_field)

        advected = solver.advection(solver.density, velocity)

        np.testing.assert_array_almost_equal(advected, initial_density, decimal=5)

    def test_advection_uniform_field(self):
        """Test advection of uniform field remains uniform."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)

        uniform_density = np.ones((32, 32))

        velocity = np.random.rand(32, 32, 2) * 0.1

        advected = solver.advection(uniform_density, velocity)

        np.testing.assert_array_almost_equal(
            advected,
            np.ones((32, 32)),
            decimal=5
        )

    def test_advection_stays_in_bounds(self):
        """Test that advection keeps values within domain bounds."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)

        x = y = np.linspace(0, 1.0, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        density = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1)

        velocity = np.ones((32, 32, 2)) * 10.0

        advected = solver.advection(density, velocity)

        assert np.all(np.isfinite(advected))

    def test_advection_vector_field(self):
        """Test advection works with vector fields (velocity)."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)

        velocity_to_advect = np.random.rand(32, 32, 2) * 0.1
        advecting_velocity = np.random.rand(32, 32, 2) * 0.1

        advected = solver.advection(velocity_to_advect, advecting_velocity)

        assert advected.shape == velocity_to_advect.shape
        assert np.all(np.isfinite(advected))


class TestDiffusion:
    """Test the diffusion method."""

    def test_diffusion_zero_coefficient(self):
        """Test that zero diffusion coefficient leaves field unchanged."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        field = np.random.rand(32, 32)

        diffused = solver.diffusion(field, diffusion_coefficient=0.0)

        np.testing.assert_array_almost_equal(diffused, field, decimal=5)

    def test_diffusion_smooths_field(self):
        """Test that diffusion smooths out sharp features."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        field = np.zeros((32, 32))
        field[16, 16] = 100.0

        initial_variance = np.var(field)

        diffused = solver.diffusion(field, diffusion_coefficient=0.1)

        final_variance = np.var(diffused)

        assert final_variance < initial_variance

        assert np.sum(diffused) == pytest.approx(np.sum(field), rel=0.1)

    def test_diffusion_vector_field(self):
        """Test diffusion on vector fields."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 100

        field = np.zeros((32, 32, 2))
        field[16, 16, :] = [10.0, 10.0]

        diffused = solver.diffusion(field, diffusion_coefficient=0.1)

        assert diffused.shape == field.shape
        assert np.all(np.isfinite(diffused))

        assert diffused[16, 16, 0] < field[16, 16, 0]

class TestFluidSolverProjection:
    """Test class for FluidSolver projection method."""

    @pytest.fixture
    def solver(self):
        """Create a FluidSolver instance matching the original test parameters."""
        return FluidSolver(
            domain_size=1.0,
            n_points=41,
            dt=0.01,
            viscosity=0.1,
            diffusion_rate=0.05
        )

    def test_projection_uniform_divergent_field(self, solver):
        """Test 1: Uniform Divergent Field (u=x, v=y) - exact mapping from test.py"""
        print("\nTest 1: Uniform Divergent Field (u=x, v=y)")
        print("-" * 70)

        # Create coordinates (same as original test)
        x = np.linspace(0.0, solver.domain_size, solver.n_points)
        y = np.linspace(0.0, solver.domain_size, solver.n_points)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Create velocity field (same as original test)
        velocities = np.zeros((solver.n_points, solver.n_points, 2))
        velocities[..., 0] = 0.5 * X  # u = 0.5*x
        velocities[..., 1] = 0.5 * Y


        div_before = divergence(velocities, solver.element_length)
        div_max_before = np.abs(div_before[2:-2, 2:-2]).max()
        div_rms_before = np.sqrt(np.mean(div_before[2:-2, 2:-2]**2))

        print(f"  Initial divergence (interior):")
        print(f"    Max: {div_max_before:.6e}")
        print(f"    RMS: {div_rms_before:.6e}")


        velocities_projected = solver.projection(velocities)


        div_after = divergence(velocities_projected, solver.element_length)
        div_max_after = np.abs(div_after[2:-2, 2:-2]).max()
        div_rms_after = np.sqrt(np.mean(div_after[2:-2, 2:-2]**2))

        print(f"  Final divergence (interior):")
        print(f"    Max: {div_max_after:.6e}")
        print(f"    RMS: {div_rms_after:.6e}")

        reduction = div_max_before / (div_max_after + 1e-16)
        print(f"  Divergence reduction: {reduction:.2e}x")

        test_passed = div_max_after < 5e-2
        print(f"  Result: {'✓ PASS' if test_passed else '✗ FAIL'}")


        assert test_passed, f"Divergence reduction failed: {div_max_after:.6e} >= 5e-2"
        assert reduction > 10.0, f"Insufficient divergence reduction: {reduction:.2e}x"

    def test_projection_smooth_oscillating_field(self, solver):
        """Test 2: Smooth Oscillating Field - exact mapping from test.py"""
        print("\nTest 2: Smooth Oscillating Field")
        print("-" * 70)


        x = np.linspace(0.0, solver.domain_size, solver.n_points)
        y = np.linspace(0.0, solver.domain_size, solver.n_points)
        X, Y = np.meshgrid(x, y, indexing="ij")


        velocities = np.zeros((solver.n_points, solver.n_points, 2))
        velocities[..., 0] = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        velocities[..., 1] = np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)



        div_before = divergence(velocities, solver.element_length)
        div_max_before = np.abs(div_before[2:-2, 2:-2]).max()
        div_rms_before = np.sqrt(np.mean(div_before[2:-2, 2:-2]**2))

        print(f"  Initial divergence (interior):")
        print(f"    Max: {div_max_before:.6e}")
        print(f"    RMS: {div_rms_before:.6e}")


        velocities_projected = solver.projection(velocities)


        div_after = divergence(velocities_projected, solver.element_length)
        div_max_after = np.abs(div_after[2:-2, 2:-2]).max()
        div_rms_after = np.sqrt(np.mean(div_after[2:-2, 2:-2]**2))

        print(f"  Final divergence (interior):")
        print(f"    Max: {div_max_after:.6e}")
        print(f"    RMS: {div_rms_after:.6e}")

        reduction = div_max_before / (div_max_after + 1e-16)
        print(f"  Divergence reduction: {reduction:.2e}x")

        test_passed = div_max_after < 5e-1
        print(f"  Result: {'✓ PASS' if test_passed else '✗ FAIL'}")


        assert test_passed, f"Divergence reduction failed: {div_max_after:.6e} >= 5e-1"

    def test_projection_localized_source(self, solver):
        """Test 3: Localized Source (Gaussian Blob) - exact mapping from test.py"""
        print("\nTest 3: Localized Source (Gaussian Blob)")
        print("-" * 70)


        x = np.linspace(0.0, solver.domain_size, solver.n_points)
        y = np.linspace(0.0, solver.domain_size, solver.n_points)
        X, Y = np.meshgrid(x, y, indexing="ij")


        velocities = np.zeros((solver.n_points, solver.n_points, 2))

        cx, cy = 0.5, 0.25
        sigma = 0.08
        r_squared = (X - cx)**2 + (Y - cy)**2
        blob = np.exp(-r_squared / (2 * sigma**2))
        velocities[..., 1] = blob

        # Calculate initial divergence
        div_before = divergence(velocities, solver.element_length)
        div_max_before = np.abs(div_before[2:-2, 2:-2]).max()
        div_rms_before = np.sqrt(np.mean(div_before[2:-2, 2:-2]**2))

        print(f"  Initial divergence (interior):")
        print(f"    Max: {div_max_before:.6e}")
        print(f"    RMS: {div_rms_before:.6e}")


        velocities_projected = solver.projection(velocities)


        div_after = divergence(velocities_projected, solver.element_length)
        div_max_after = np.abs(div_after[2:-2, 2:-2]).max()
        div_rms_after = np.sqrt(np.mean(div_after[2:-2, 2:-2]**2))

        print(f"  Final divergence (interior):")
        print(f"    Max: {div_max_after:.6e}")
        print(f"    RMS: {div_rms_after:.6e}")

        reduction = div_max_before / (div_max_after + 1e-16)
        print(f"  Divergence reduction: {reduction:.2e}x")

        test_passed = div_max_after < 5e-1
        print(f"  Result: {'✓ PASS' if test_passed else '✗ FAIL'}")


        assert test_passed, f"Divergence reduction failed: {div_max_after:.6e} >= 5e-1"

    def test_projection_identity_on_divergence_free_field(self, solver):
        """Test that projection of a divergence-free field is the identity (field remains unchanged)."""
        print("\nTest: Projection Identity on Divergence-Free Field")
        print("-" * 70)

        x = np.linspace(0.0, solver.domain_size, solver.n_points)
        y = np.linspace(0.0, solver.domain_size, solver.n_points)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Create a divergence-free vector field using stream function approach
        # Using stream function ψ = sin(2πx)sin(2πy), then u = ∂ψ/∂y, v = -∂ψ/∂x
        velocities = np.zeros((solver.n_points, solver.n_points, 2))
        velocities[..., 0] = 2 * np.pi * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)  # u = ∂ψ/∂y
        velocities[..., 1] = -2 * np.pi * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)  # v = -∂ψ/∂x

        div_before = divergence(velocities, solver.element_length)
        div_max_before = np.abs(div_before[2:-2, 2:-2]).max()
        div_rms_before = np.sqrt(np.mean(div_before[2:-2, 2:-2]**2))

        print(f"  Initial divergence (should be ~0 for divergence-free field):")
        print(f"    Max: {div_max_before:.6e}")
        print(f"    RMS: {div_rms_before:.6e}")

        velocities_original = velocities.copy()

        velocities_projected = solver.projection(velocities)

        diff_field = velocities_projected - velocities_original
        max_diff = np.abs(diff_field[2:-2, 2:-2]).max()
        rms_diff = np.sqrt(np.mean(diff_field[2:-2, 2:-2]**2))

        print(f"  Difference after projection (should be ~0 for identity):")
        print(f"    Max: {max_diff:.6e}")
        print(f"    RMS: {rms_diff:.6e}")


        div_after = divergence(velocities_projected, solver.element_length)
        div_max_after = np.abs(div_after[2:-2, 2:-2]).max()

        print(f"  Final divergence:")
        print(f"    Max: {div_max_after:.6e}")

        initial_div_free = div_max_before < 1e-10
        identity_preserved = max_diff < 1e-10
        final_div_free = div_max_after < 1e-10

        test_passed = initial_div_free and identity_preserved and final_div_free

        print(f"  Initial field divergence-free: {'✓' if initial_div_free else '✗'}")
        print(f"  Identity property preserved: {'✓' if identity_preserved else '✗'}")
        print(f"  Final field divergence-free: {'✓' if final_div_free else '✗'}")
        print(f"  Result: {'✓ PASS' if test_passed else '✗ FAIL'}")

        assert initial_div_free, f"Initial field not divergence-free: {div_max_before:.6e}"
        assert identity_preserved, f"Identity property violated, max change: {max_diff:.6e}"
        assert final_div_free, f"Final field not divergence-free: {div_max_after:.6e}"


class TestStepMethod:
    """Test the complete step method."""

    def test_step_executes(self):
        """Test that step method executes without errors."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 50


        solver.density[16, 16] = 1.0
        solver.forcing_field[16, 16, :] = [0.1, 0.1]

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


        solver.forcing_field[16, 16, :] = [1.0, 0.0]
        solver.density[16, 16] = 1.0

        for _ in range(5):
            solver.step()


            assert np.all(np.isfinite(solver.velocity_field))
            assert np.all(np.isfinite(solver.density))

    def test_step_mass_conservation(self):
        """Test that total density is approximately conserved."""
        solver = FluidSolver(1.0, 32, 0.01, 0.0, 0.0)
        solver.max_iter = 50


        solver.density[16, 16] = 100.0
        initial_mass = np.sum(solver.density)


        for _ in range(3):
            solver.step()

        final_mass = np.sum(solver.density)


        assert final_mass == pytest.approx(initial_mass, rel=0.15)

    def test_step_with_forcing(self):
        """Test that forcing field affects velocity."""
        solver = FluidSolver(1.0, 32, 0.1, 0.001, 0.0001)
        solver.max_iter = 50


        solver.forcing_field[:, :, 0] = 1.0

        solver.step()


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


        solver.velocity_field[:, :, :] = np.random.rand(32, 32, 2) * 0.5
        initial_energy = np.sum(solver.velocity_field**2)


        for _ in range(10):
            solver.step()

        final_energy = np.sum(solver.velocity_field**2)


        assert final_energy < initial_energy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
