import numpy as np
from src.HDF5File import HDF5File
from src.FiniteDifferences import fourth_diff


class PDESolver:
    def __init__(self, path, x, t, init, bc, f):
        self.x = x
        self.t = t
        self.init = init
        self.bc = bc
        self.f = f
        self.file = HDF5File(f"{path}/data.h5")

    def params(self, params):
        courant_param, dx, sigma = params
        dt = courant_param * dx
        m_max = int(self.x / dx)
        n_max = int(self.t / dt)
        x_grid = np.linspace(0, m_max * dx, m_max + 1)
        t_grid = np.linspace(0, n_max * dt, n_max + 1)

        return [courant_param, dx, sigma, dt, m_max, n_max, x_grid, t_grid]

    def solve(self, params, depth, scale=None):
        dataset = f"sol_{params}"
        courant_param, dx, sigma, dt, m_max, n_max, x_grid = self.params(params)[:-1]
        scale = [lambda u: u, lambda u: u] if scale is None else scale

        if self.file.dataset(dataset): return dataset

        self.file.create_dataset(dataset, (n_max + 1, m_max + 1, 2))

        for j in range(2): self.file.save(dataset, (0, slice(None), j), self.init[j](x_grid))

        q = lambda u: sigma * dt**3 / 16 * fourth_diff(u, dx, self.bc)

        for n in range(0, n_max, depth):
            grid = np.empty([min(depth, n_max - n) + 1, m_max + 1, 2])
            for j in range(2): grid[0, :, j] = scale[0](self.file.load(dataset, (n, slice(None), j)))

            for i in range(min(depth, n_max - n)):
                k1 = [self.f[j]((n + i) * dt, grid[i, :, 0], grid[i, :, 1]) - q(grid[i, :, j]) for j in range(2)]
                k2 = [self.f[j]((n + i + 0.5) * dt, grid[i, :, 0] + dt / 2 * k1[0], grid[i, :, 1] + dt / 2 * k1[1]) - q(grid[i, :, j] + dt / 2 * k1[j]) for j in range(2)]
                k3 = [self.f[j]((n + i + 0.5) * dt, grid[i, :, 0] + dt / 2 * k2[0], grid[i, :, 1] + dt / 2 * k2[1]) - q(grid[i, :, j] + dt / 2 * k2[j]) for j in range(2)]
                k4 = [self.f[j]((n + i + 1) * dt, grid[i, :, 0] + dt * k3[0], grid[i, :, 1] + dt * k3[1]) - q(grid[i, :, j] + dt * k3[j]) for j in range(2)]

                for j in range(2): grid[i + 1, :, j] = grid[i, :, j] + dt / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])

            for j in range(2): self.file.save(dataset, (slice(n + 1, min(n + depth, n_max) + 1), slice(None), j), np.apply_along_axis(scale[1], axis=1, arr=grid[1:, :, j]))
        return dataset

    def solve_convergence(self, params, depth, **kwargs):
        dataset = f'{'n' if kwargs.get('norm') is True else 'p'}_{'e' if kwargs.get('exact') is not None else 's'}_conv_{params}'
        courant_param, dx, sigma, dt, m_max, n_max, x_grid, t_grid = self.params(params)

        n_sol = 2 if kwargs.get("exact") is not None else 3
        params = [[courant_param, dx / 2**i, sigma] for i in range(n_sol)]
        sol_dataset = [self.solve(params[i], depth) for i in range(n_sol)]

        self.file.create_dataset(dataset, (n_max + 1,) if kwargs.get("norm") is True else (2, n_max + 1, m_max + 1))

        for n in range(0, n_max, depth):
            sol = [self.file.load(sol_dataset[i], (slice(2**i * n, 2**i * min(n + depth, n_max + 1), 2**i), slice(None, None, 2**i), 0)) for i in range(n_sol)]

            if kwargs.get("exact") is not None:
                error = [sol[i] - kwargs["exact"](x_grid[None, :], t_grid[n:(min(n + depth, n_max + 1))][:, None]) for i in range(n_sol)]
            else:
                error = [sol[i] - sol[i + 1] for i in range(n_sol - 1)]

            if kwargs.get("norm") is True:
                norm_error = [np.sqrt(dx * np.sum(error[i]**2, axis=-1)) for i in range(2)]
                self.file.save(dataset, slice(n, min(n + depth, n_max + 1)), np.log2(norm_error[0] / norm_error[1]))
            else:
                self.file.save(dataset, (0, slice(n, min(n + depth, n_max + 1)), slice(None)), error[0])
                self.file.save(dataset, (1, slice(n, min(n + depth, n_max + 1)), slice(None)), error[1])

    def get_x_grid(self, params):
        return self.params(params)[6]

    def get_t_grid(self, params):
        return self.params(params)[7]

    def get_solution(self, params, t):
        dataset = f"sol_{params}"
        courant_param, dx, sigma, dt = self.params(params)[:4]

        return self.file.load(dataset, (int(t / dt), slice(None), 0))

    def get_convergence(self, params, t=None, **kwargs):
        dataset = f'{'n' if kwargs.get('norm') is True else 'p'}_{'e' if kwargs.get('exact') is not None else 's'}_conv_{params}'
        courant_param, dx, sigma, dt = self.params(params)[:4]
        index = slice(None) if kwargs.get('norm') is True else (slice(None), int(t / dt), slice(None))

        return self.file.load(dataset, index)
