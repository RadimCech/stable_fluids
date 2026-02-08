"""
Jos Stam's Stable Fluids Solver
A real-time 2D fluid simulation using the stable fluids method.

Controls:
- Left Mouse: Add density and force
- Right Mouse: Add negative density
- 'R': Reset simulation
- 'V': Toggle velocity field visualization
- Space: Pause/Resume
"""

from typing import Tuple

import numpy as np
import pygame

# Simulation parameters
GRID_SIZE = 128  # Grid resolution (NxN)
CELL_SIZE = 5  # Pixel size of each cell
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

# Fluid parameters
VISCOSITY = 0.0000001  # Kinematic viscosity
DIFFUSION = 0.0000001  # Density diffusion rate
DT = 0.1  # Time step
ITERATIONS = 4  # Gauss-Seidel iterations for linear solver

# Interaction parameters
FORCE_SCALE = 50.0
DENSITY_AMOUNT = 100.0


class FluidSolver:
    """Jos Stam's stable fluids solver implementation."""

    def __init__(self, size: int, dt: float, diffusion: float, viscosity: float):
        self.size = size
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity

        # Velocity field (u, v components)
        self.u = np.zeros((size, size), dtype=np.float32)
        self.v = np.zeros((size, size), dtype=np.float32)

        # Previous velocity field
        self.u_prev = np.zeros((size, size), dtype=np.float32)
        self.v_prev = np.zeros((size, size), dtype=np.float32)

        # Density field
        self.density = np.zeros((size, size), dtype=np.float32)
        self.density_prev = np.zeros((size, size), dtype=np.float32)

    def add_density(self, x: int, y: int, amount: float):
        """Add density at a specific location."""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.density[y, x] += amount

    def add_velocity(self, x: int, y: int, vx: float, vy: float):
        """Add velocity at a specific location."""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.u[y, x] += vx
            self.v[y, x] += vy

    def step(self):
        """Perform one simulation step."""
        # Velocity step
        self.diffuse_velocity()
        self.project()
        self.advect_velocity()
        self.project()

        # Density step
        self.diffuse_density()
        self.advect_density()

        # Fade density over time
        self.density *= 0.99

    def diffuse_velocity(self):
        """Diffuse velocity field (viscosity)."""
        a = self.dt * self.visc * (self.size - 2) ** 2
        self.u = self.lin_solve(self.u, self.u.copy(), a, 1 + 4 * a)
        self.v = self.lin_solve(self.v, self.v.copy(), a, 1 + 4 * a)

    def diffuse_density(self):
        """Diffuse density field."""
        a = self.dt * self.diff * (self.size - 2) ** 2
        self.density = self.lin_solve(self.density, self.density.copy(), a, 1 + 4 * a)

    def lin_solve(
        self, x: np.ndarray, x0: np.ndarray, a: float, c: float
    ) -> np.ndarray:
        """
        Solve linear system using Gauss-Seidel iteration.
        Solves: (1 - a*nabla^2) x = x0
        """
        c_recip = 1.0 / c

        for _ in range(ITERATIONS):
            x[1:-1, 1:-1] = (
                x0[1:-1, 1:-1]
                + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])
            ) * c_recip
            self.set_bounds(x)

        return x

    def project(self):
        """
        Project velocity field to be divergence-free (incompressible).
        This is the Helmholtz-Hodge decomposition.
        """
        # Compute divergence
        div = np.zeros_like(self.u)
        div[1:-1, 1:-1] = (
            -0.5
            * (
                self.u[1:-1, 2:]
                - self.u[1:-1, :-2]
                + self.v[2:, 1:-1]
                - self.v[:-2, 1:-1]
            )
            / self.size
        )

        # Compute pressure field
        p = np.zeros_like(self.u)
        p = self.lin_solve(p, div, 1, 4)

        # Subtract pressure gradient from velocity
        self.u[1:-1, 1:-1] -= 0.5 * self.size * (p[1:-1, 2:] - p[1:-1, :-2])
        self.v[1:-1, 1:-1] -= 0.5 * self.size * (p[2:, 1:-1] - p[:-2, 1:-1])

        self.set_bounds(self.u)
        self.set_bounds(self.v)

    def advect_velocity(self):
        """Advect velocity field using semi-Lagrangian advection."""
        self.u = self.advect(self.u, self.u, self.v)
        self.v = self.advect(self.v, self.u, self.v)

    def advect_density(self):
        """Advect density field using semi-Lagrangian advection."""
        self.density = self.advect(self.density, self.u, self.v)

    def advect(self, d: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Semi-Lagrangian advection (backward particle tracing).
        For each grid cell, trace back through velocity field and interpolate.
        """
        dt0 = self.dt * (self.size - 2)
        d_new = d.copy()

        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                # Trace particle backwards in time
                x = j - dt0 * u[i, j]
                y = i - dt0 * v[i, j]

                # Clamp to grid boundaries
                x = np.clip(x, 0.5, self.size - 2 + 0.5)
                y = np.clip(y, 0.5, self.size - 2 + 0.5)

                # Bilinear interpolation
                i0, j0 = int(y), int(x)
                i1, j1 = i0 + 1, j0 + 1

                s1 = x - j0
                s0 = 1 - s1
                t1 = y - i0
                t0 = 1 - t1

                d_new[i, j] = t0 * (s0 * d[i0, j0] + s1 * d[i0, j1]) + t1 * (
                    s0 * d[i1, j0] + s1 * d[i1, j1]
                )

        self.set_bounds(d_new)
        return d_new

    def set_bounds(self, x: np.ndarray):
        """Set boundary conditions (no-slip walls)."""
        # Left and right walls
        x[:, 0] = -x[:, 1]
        x[:, -1] = -x[:, -2]

        # Top and bottom walls
        x[0, :] = -x[1, :]
        x[-1, :] = -x[-2, :]

        # Corners
        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, -1] = 0.5 * (x[1, -1] + x[0, -2])
        x[-1, 0] = 0.5 * (x[-2, 0] + x[-1, 1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

    def reset(self):
        """Reset simulation to initial state."""
        self.u.fill(0)
        self.v.fill(0)
        self.u_prev.fill(0)
        self.v_prev.fill(0)
        self.density.fill(0)
        self.density_prev.fill(0)


def draw_density(screen: pygame.Surface, fluid: FluidSolver):
    """Render density field as colors."""
    # Normalize density for display
    max_density = max(np.max(fluid.density), 1.0)
    normalized = np.clip(fluid.density / max_density, 0, 1)

    # Create color array (grayscale)
    colors = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    colors[:, :, 0] = normalized * 255  # Red
    colors[:, :, 1] = normalized * 200  # Green
    colors[:, :, 2] = normalized * 255  # Blue

    # Scale up to screen size
    surface = pygame.surfarray.make_surface(colors.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surface, (WIDTH, HEIGHT))
    screen.blit(scaled, (0, 0))


def draw_velocity(screen: pygame.Surface, fluid: FluidSolver):
    """Draw velocity field as arrows."""
    step = 8  # Draw every Nth arrow
    scale = 2.0

    for i in range(0, GRID_SIZE, step):
        for j in range(0, GRID_SIZE, step):
            x = j * CELL_SIZE
            y = i * CELL_SIZE

            vx = fluid.u[i, j] * scale
            vy = fluid.v[i, j] * scale

            magnitude = np.sqrt(vx**2 + vy**2)
            if magnitude > 0.5:
                end_x = x + vx
                end_y = y + vy
                pygame.draw.line(screen, (0, 255, 0), (x, y), (end_x, end_y), 1)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Jos Stam's Stable Fluids")
    clock = pygame.time.Clock()

    # Create fluid solver
    fluid = FluidSolver(GRID_SIZE, DT, DIFFUSION, VISCOSITY)

    # State variables
    running = True
    paused = False
    show_velocity = False
    prev_mouse_pos = None

    print("Controls:")
    print("- Left Mouse: Add density and force")
    print("- Right Mouse: Add negative density")
    print("- 'R': Reset simulation")
    print("- 'V': Toggle velocity field visualization")
    print("- Space: Pause/Resume")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    fluid.reset()
                elif event.key == pygame.K_v:
                    show_velocity = not show_velocity
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        # Handle mouse input
        mouse_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()

        if mouse_buttons[0] or mouse_buttons[2]:  # Left or right click
            grid_x = mouse_pos[0] // CELL_SIZE
            grid_y = mouse_pos[1] // CELL_SIZE

            # Add density
            density_sign = 1.0 if mouse_buttons[0] else -1.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    fluid.add_density(
                        grid_x + dx, grid_y + dy, DENSITY_AMOUNT * density_sign
                    )

            # Add velocity based on mouse movement
            if prev_mouse_pos is not None:
                dx = (mouse_pos[0] - prev_mouse_pos[0]) * FORCE_SCALE
                dy = (mouse_pos[1] - prev_mouse_pos[1]) * FORCE_SCALE

                for dy_offset in range(-1, 2):
                    for dx_offset in range(-1, 2):
                        fluid.add_velocity(
                            grid_x + dx_offset, grid_y + dy_offset, dx, dy
                        )

        prev_mouse_pos = mouse_pos if mouse_buttons[0] or mouse_buttons[2] else None

        # Update simulation
        if not paused:
            fluid.step()

        # Render
        screen.fill((0, 0, 0))
        draw_density(screen, fluid)

        if show_velocity:
            draw_velocity(screen, fluid)

        # Draw status
        font = pygame.font.Font(None, 24)
        status = f"FPS: {int(clock.get_fps())} | {'PAUSED' if paused else 'RUNNING'} | Velocity: {'ON' if show_velocity else 'OFF'}"
        text = font.render(status, True, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
