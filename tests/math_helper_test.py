
import numpy as np
import pytest
from src.math_helper import (
    partial_derivative_x,
    partial_derivative_y,
    gradient,
    divergence,
    laplace
)


class TestPartialDerivatives:
    """Test partial derivative functions."""

    def test_partial_derivative_x_linear_field(self):
        """Test ∂/∂x of a linear field f(x,y) = x."""

        field = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ], dtype=float)

        result = partial_derivative_x(field, element_length=1.0)


        assert np.allclose(result[1:-1, 1:-1], 1.0)

        assert np.all(result[0, :] == 0)
        assert np.all(result[-1, :] == 0)

    def test_partial_derivative_y_linear_field(self):
        """Test ∂/∂y of a linear field f(x,y) = y."""
        field = np.array([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ], dtype=float)

        result = partial_derivative_y(field, element_length=1.0)

        assert np.allclose(result[1:-1, 1:-1], 1.0)
        assert np.all(result[:, 0] == 0)
        assert np.all(result[:, -1] == 0)

    def test_partial_derivative_x_quadratic_field(self):
        """Test ∂/∂x of f(x,y) = x²."""
        x = np.arange(5)
        y = np.arange(5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        field = X**2

        result = partial_derivative_x(field, element_length=1.0)


        expected_middle = 2 * X[1:-1, 1:-1]
        assert np.allclose(result[1:-1, 1:-1], expected_middle)

    def test_partial_derivative_element_length_scaling(self):
        """Test that element_length correctly scales the derivative."""
        field = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ], dtype=float)

        result_h1 = partial_derivative_x(field, element_length=1.0)
        result_h2 = partial_derivative_x(field, element_length=2.0)


        assert np.allclose(result_h2[1:-1, 1:-1],
                          result_h1[1:-1, 1:-1] / 2)


class TestGradient:
    """Test gradient function."""

    def test_gradient_linear_field(self):
        """Test gradient of f(x,y) = 2x + 3y."""
        x = np.arange(5)
        y = np.arange(5)
        X, Y = np.meshgrid(x, y, indexing='ij')
        field = 2*X + 3*Y

        grad = gradient(field, element_length=1.0)


        assert grad.shape == (*field.shape, 2)
        assert np.allclose(grad[1:-1, 1:-1, 0], 2.0)
        assert np.allclose(grad[1:-1, 1:-1, 1], 3.0)

    def test_gradient_shape(self):
        """Test that gradient output has correct shape."""
        field = np.random.rand(10, 10)
        grad = gradient(field, element_length=1.0)

        assert grad.shape == (10, 10, 2)

    def test_gradient_constant_field(self):
        """Test gradient of constant field is zero."""
        field = np.ones((5, 5)) * 42
        grad = gradient(field, element_length=1.0)

        assert np.allclose(grad[1:-1, 1:-1], 0.0)


class TestDivergence:
    """Test divergence function."""

    def test_divergence_constant_field(self):
        """Test divergence of constant vector field is zero."""
        vector_field = np.ones((5, 5, 2)) * 3.0
        div = divergence(vector_field, element_length=1.0)

        assert np.allclose(div[1:-1, 1:-1], 0.0)

    def test_divergence_linear_expansion(self):
        """Test div of expanding field F = [x, y]."""
        x = np.arange(5)
        y = np.arange(5)
        X, Y = np.meshgrid(x, y, indexing='ij')

        vector_field = np.stack([X, Y], axis=-1).astype(float)
        div = divergence(vector_field, element_length=1.0)


        assert np.allclose(div[1:-1, 1:-1], 2.0)

    def test_divergence_shape(self):
        """Test divergence output shape."""
        vector_field = np.random.rand(8, 8, 2)
        div = divergence(vector_field, element_length=1.0)

        assert div.shape == (8, 8)


class TestLaplace:
    """Test Laplacian operator."""

    def test_laplace_linear_field(self):
        """Test Laplacian of linear field is zero."""
        x = np.arange(7)
        y = np.arange(7)
        X, Y = np.meshgrid(x, y, indexing='ij')
        field = 2*X + 3*Y

        lap = laplace(field, element_length=1.0)


        assert np.allclose(lap[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_laplace_quadratic_field(self):
        """Test Laplacian of f(x,y) = x² + y²."""
        x = np.arange(7)
        y = np.arange(7)
        X, Y = np.meshgrid(x, y, indexing='ij')
        field = X**2 + Y**2

        lap = laplace(field, element_length=1.0)


        assert np.allclose(lap[1:-1, 1:-1], 4.0)

    def test_laplace_element_length_scaling(self):
        """Test Laplacian scaling with element length."""
        field = np.random.rand(7, 7)

        lap_h1 = laplace(field, element_length=1.0)
        lap_h2 = laplace(field, element_length=2.0)


        assert np.allclose(lap_h2[1:-1, 1:-1],
                          lap_h1[1:-1, 1:-1] / 4)

    def test_laplace_sine_wave(self):
        """Test Laplacian of sin(x)sin(y)."""
        x = np.linspace(0, 2*np.pi, 20)
        y = np.linspace(0, 2*np.pi, 20)
        X, Y = np.meshgrid(x, y, indexing='ij')
        field = np.sin(X) * np.sin(Y)

        dx = x[1] - x[0]
        lap = laplace(field, element_length=dx)


        expected = -2 * field
        assert np.allclose(lap[1:-1, 1:-1],
                          expected[1:-1, 1:-1],
                          rtol=0.01)


class TestBoundaryConditions:
    """Test that boundaries are handled correctly."""

    def test_all_functions_zero_boundaries(self):
        """Verify all functions return zero at boundaries."""
        field = np.random.rand(6, 6)
        vector_field = np.random.rand(6, 6, 2)
        h = 1.0


        for func, arg in [
            (partial_derivative_x, field),
            (partial_derivative_y, field),
            (laplace, field),
            (divergence, vector_field),
        ]:
            result = func(arg, h)


            assert np.all(result[0, :] == 0)
            assert np.all(result[-1, :] == 0)
            assert np.all(result[:, 0] == 0)
            assert np.all(result[:, -1] == 0)



@pytest.mark.parametrize("size", [3, 5, 10, 20])
def test_laplace_on_various_sizes(size):
    """Test Laplacian works correctly on different grid sizes."""
    field = np.ones((size, size))
    lap = laplace(field, element_length=1.0)

    assert lap.shape == (size, size)
    assert np.allclose(lap[1:-1, 1:-1], 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
