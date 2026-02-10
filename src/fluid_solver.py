import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
from .math_helper import laplace, divergence, gradient
import pygame

class FluidSolver:

    def __init__(self, domain_size: float, n_points: int, dt: float,
                 viscosity: float, diffusion_rate: float,
                 dissipation_rate: float = 0.0, window_size: int = 800):

        self.domain_size = domain_size
        self.n_points = n_points
        self.dt = dt
        self.viscosity = viscosity
        self.diffusion_rate = diffusion_rate
        self.dissipation_rate = dissipation_rate

        self.vector_field_shape = (n_points, n_points, 2)
        self.element_length = self.domain_size / (self.n_points - 1)
        self.vector_field_dof = self.n_points**2*2
        self.max_iter = None

        self.velocity_field = np.zeros(self.vector_field_shape, dtype=np.float32)
        self.velocity_field_prev = np.zeros(self.vector_field_shape, dtype=np.float32)
        self.forcing_field = np.zeros(self.vector_field_shape, dtype=np.float32)


        self.density = np.zeros((n_points, n_points), dtype=np.float32)
        self.density_prev = np.zeros((n_points, n_points), dtype=np.float32)

        # Pygame setup
        self.window_size = window_size
        self.screen = None
        self.init_pygame()

    def init_pygame(self):
        """Initialize pygame for visualization"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Stable Fluid Simulation")

    def apply_boundary_conditions(self, field):
        if field.ndim == 2:  # Scalar field
            field[0, :] = 0.0
            field[-1, :] = 0.0
            field[:, 0] = 0.0
            field[:, -1] = 0.0
        elif field.ndim == 3:  # Vector field
            field[0, :, :] = 0.0
            field[-1, :, :] = 0.0
            field[:, 0, :] = 0.0
            field[:, -1, :] = 0.0
        return field

    def add_forces(self):
        self.velocity_field += self.dt * self.forcing_field
        self.density += self.dt * np.linalg.norm(self.forcing_field, axis=-1)

    def advection(self, advected_field, velocity_vector_field):
        # we need coordinates!
        x = y = np.linspace(0., self.domain_size, self.n_points)
        X, Y = np.meshgrid(x, y, indexing='ij')
        coordinates = np.concatenate(
            (X[..., np.newaxis], Y[..., np.newaxis]),
            axis=-1
        )

        backtracked_positions = np.clip(
            coordinates - self.dt * velocity_vector_field,
            0.,
            self.domain_size
        )

        advected_field = interpolate.interpn(
            points=(x, y),
            values=advected_field,
            xi=backtracked_positions,
        )

        return advected_field



    def diffusion(self, field, diffusion_coefficient):
        original_shape = field.shape
        field_dof = np.prod(original_shape)
        def diffusion_operator(field_flattened):
            _field = field_flattened.reshape(original_shape)
            diffusion_applied = (
                _field - diffusion_coefficient * self.dt * laplace(_field, self.element_length)
            )

            return diffusion_applied.flatten()

        return splinalg.cg(
            A=splinalg.LinearOperator(
                shape=(field_dof, field_dof),
                matvec=diffusion_operator,
            ),
            b=field.flatten(),
            maxiter=self.max_iter,
        )[0].reshape(original_shape)

    def projection(self, vector_field):
            def poisson_operator(field_flattened):
                field = field_flattened.reshape((self.n_points, self.n_points))
                poisson_applied = laplace(field, self.element_length)
                return poisson_applied.flatten()

            pressure = splinalg.cg(
                A=splinalg.LinearOperator(
                    shape=(self.n_points**2, self.n_points**2),
                    matvec=poisson_operator,
                ),
                b=divergence(vector_field, self.element_length).flatten(),
                maxiter=self.max_iter,
            )[0].reshape((self.n_points, self.n_points))
            return vector_field - gradient(pressure, self.element_length)

    def draw(self):
        """Draw the density field using pygame"""
        if self.screen is None:
            return

        # Normalize density for visualization (0 to 1 range)
        max_density = np.max(self.density)
        if max_density > 0:
            normalized_density = np.clip(self.density / max_density, 0, 1)
        else:
            normalized_density = self.density

        # Create a surface array for better performance
        surface_array = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

        # Scale up the density field to window size
        scale_x = self.window_size / self.n_points
        scale_y = self.window_size / self.n_points

        # Map density values to colors more efficiently
        for i in range(self.n_points):
            for j in range(self.n_points):
                # Get density value
                density_val = normalized_density[i, j]

                # Convert to grayscale (white = high density)
                color_val = int(density_val * 255)

                # Calculate pixel ranges
                x_start = int(i * scale_x)
                x_end = int((i + 1) * scale_x)
                y_start = int(j * scale_y)
                y_end = int((j + 1) * scale_y)

                # Fill the corresponding pixels
                surface_array[x_start:x_end, y_start:y_end] = [color_val, color_val, color_val]

        # Create surface from array and blit to screen
        surf = pygame.surfarray.make_surface(surface_array)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def step(self):
        self.add_forces()

        self.velocity_field = self.apply_boundary_conditions(self.velocity_field)
        self.velocity_field = self.diffusion(self.velocity_field, self.viscosity)
        self.velocity_field = self.apply_boundary_conditions(self.velocity_field)
        self.velocity_field = self.advection(self.velocity_field, self.velocity_field)
        self.velocity_field = self.apply_boundary_conditions(self.velocity_field)
        self.velocity_field = self.projection(self.velocity_field)

        self.density = self.diffusion(self.density, self.diffusion_rate)
        self.density = self.advection(self.density, self.velocity_field)
        #self.density = self.density / (1.0 + self.dt * self.dissipation_rate)

        self.draw()
        self.velocity_field_prev = self.velocity_field.copy()
        self.density_prev = self.density.copy()
