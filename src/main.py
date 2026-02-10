import pygame
from .fluid_solver import FluidSolver


if __name__ == "__main__":

    solver = FluidSolver(
        domain_size=1.0,
        n_points=64,
        dt=0.01,
        viscosity=0.0001,
        diffusion_rate=0.0001
    )

    # Set up some initial forcing in the middle of the domain
    mid = solver.n_points // 2
    solver.forcing_field[mid-5:mid+5, mid-5:mid+5, 0] = 0.0
    solver.forcing_field[mid-5:mid+5, mid-5:mid+5, 1] = 10.0

    # Main simulation loop with pygame event handling
    clock = pygame.time.Clock()
    running = True
    step_count = 0
    max_steps = 450

    print("Starting fluid simulation...")
    print("Close the pygame window or press ESC to exit")

    while running and step_count < max_steps:
        if step_count > 150:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Update simulation
            solver.step()
        step_count += 1

        # Control frame rate
        clock.tick(60)  # 60 FPS

        # Print progress occasionally
        if step_count % 100 == 0:
            print(f"Step {step_count}/{max_steps}")

    print("Simulation completed!")
    pygame.quit()
