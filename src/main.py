from .fluid_solver import FluidSolver
from .visualizer import FluidVisualizer


def main():
    """Main entry point for the stable fluids simulation."""
    solver = FluidSolver(
        domain_size=1.0,
        n_points=64,
        dt=0.01,
        viscosity=0.0001,
        diffusion_rate=0.0001
    )

    visualizer = FluidVisualizer(window_size=800)

    mid = solver.n_points // 2
    solver.forcing_field[mid-5:mid+5, mid-5:mid+5, 0] = 10.0
    solver.forcing_field[mid-5:mid+5, mid-5:mid+5, 1] = 5.0

    running = True
    step_count = 0
    max_steps = 1000

    print("Starting fluid simulation...")
    print("Close the pygame window or press ESC to exit")

    while running and step_count < max_steps:
        running = visualizer.handle_events()

        solver.step()
        step_count += 1

        visualizer.draw_density_field(solver.density, solver.n_points)

        visualizer.tick(60)

        if step_count % 100 == 0:
            print(f"Step {step_count}/{max_steps}")

    print("Simulation completed!")
    visualizer.cleanup()

if __name__ == "__main__":
    main()
