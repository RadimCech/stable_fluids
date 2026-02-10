import pygame
import numpy as np


class FluidVisualizer:
    """Handles pygame visualization for fluid simulation"""

    def __init__(self, window_size: int = 800):
        self.window_size = window_size
        self.screen = None
        self.clock = None
        self.init_pygame()

    def init_pygame(self):
        """Initialize pygame for visualization"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Stable Fluid Simulation")
        self.clock = pygame.time.Clock()

    def handle_events(self):
        """Handle pygame events and return whether to continue running"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def draw_density_field(self, density_field, n_points):
        """Draw the density field as red colors"""
        if self.screen is None:
            return

        max_density = np.max(density_field)
        if max_density > 0:
            normalized_density = np.clip(density_field / max_density, 0, 1)
        else:
            normalized_density = density_field

        surface_array = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

        scale_x = self.window_size / n_points
        scale_y = self.window_size / n_points

        for i in range(n_points):
            for j in range(n_points):
                density_val = normalized_density[i, j]

                color_val = int(density_val * 255)

                x_start = int(i * scale_x)
                x_end = int((i + 1) * scale_x)
                y_start = int(j * scale_y)
                y_end = int((j + 1) * scale_y)

                surface_array[x_start:x_end, y_start:y_end] = [color_val, 0, 0]

        surf = pygame.surfarray.make_surface(surface_array)
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

    def tick(self, fps=60):
        """Control frame rate"""
        if self.clock is not None:
            self.clock.tick(fps)

    def cleanup(self):
        """Clean up pygame resources"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
