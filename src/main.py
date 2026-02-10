import numpy as np
from tqdm import tqdm

from .fluidSolver import FluidSolver



if __name__ == "__main__":

    solver = FluidSolver(
        domain_size=1.0,
        n_points=64,
        dt=0.01,
        viscosity=0.0001,
        diffusion_rate=0.0001
    )


    solver.forcing_field = np.ones(solver.vector_field_shape, dtype=np.float32)
    solver.density = np.ones((solver.n_points, solver.n_points))
    solver.density_prev = np.ones((solver.n_points, solver.n_points))

    for i in tqdm(range(10)):
        solver.step()
