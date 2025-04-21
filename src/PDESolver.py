import numpy as np
from src.HDF5File import HDF5File


class PDESolver1D:
    def __init__(self, x, t, init, f):
        self.x = x
        self.t = t
        self.init = init
        self.f = f
        self.file = HDF5File(f"bin/data.h5")

    def solve(self, solver_params):
        dataset = f"sol_{solver_params}"

        courant_param, delta_x, depth = solver_params
        delta_t = courant_param * delta_x
        m_max = int(self.x / delta_x)
        n_max = int(self.t / delta_t)
        spatial_domain = np.linspace(0, m_max * delta_x, m_max + 1)
        time_domain = np.linspace(0, n_max * delta_t, n_max + 1)

        self.file.create_dataset(dataset, (n_max + 1, m_max + 1, 2))
        self.file.save_metadata(dataset, solver_params=solver_params, delta_t=delta_t, m_max=m_max, n_max=n_max, spatial_domain=spatial_domain, time_domain=time_domain)

        for j in range(2): self.file.save(dataset, (0, slice(None), j), self.init[j](spatial_domain))

        for n in range(0, n_max, depth):
            grid = np.empty([min(depth, n_max - n) + 1, m_max + 1, 2])
            grid[0, :, :] = self.file.load(dataset, (n, slice(None), slice(None)))

            for i in range(min(depth, n_max - n)):
                k1 = [self.f[j](n + i, grid[i, :, 0], grid[i, :, 1]) for j in range(2)]
                k2 = [self.f[j](n + i + delta_t / 2, grid[i, :, 0] + delta_t / 2 * k1[0], grid[i, :, 1] + delta_t / 2 * k1[1]) for j in range(2)]
                k3 = [self.f[j](n + i + delta_t / 2, grid[i, :, 0] + delta_t / 2 * k2[0], grid[i, :, 1] + delta_t / 2 * k2[1]) for j in range(2)]
                k4 = [self.f[j](n + i + delta_t, grid[i, :, 0] + delta_t * k3[0], grid[i, :, 1] + delta_t * k3[1]) for j in range(2)]

                for j in range(2): grid[i + 1, :, j] = grid[i, :, j] + delta_t / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j])

            self.file.save(dataset, (slice(n + 1, min(n + depth, n_max) + 1), slice(None), slice(None)), grid[1:, :, :])
        return dataset

    def solve_convergence(self, solver_params, **kwargs):
        dataset = (f"{"exact" if kwargs.get("exact") is not None else "self"}_"
                   f"{"norm" if kwargs.get("norm") is True else "point"}_"
                   f"conv_{solver_params}")

        courant_param, delta_x, depth = solver_params
        n_solvers = 2 if kwargs.get("exact") is not None else 3
        solver_params = [[courant_param, delta_x / pow(2, i), depth] for i in range(n_solvers)]
        sol_dataset = [self.solve(solver_params[i]) for i in range(n_solvers)]
        m_max, n_max, spatial_domain, time_domain = self.file.load_metadata(sol_dataset[0], ["m_max", "n_max", "spatial_domain", "time_domain"])

        self.file.create_dataset(dataset, (n_max + 1,) if kwargs.get("norm") is True else (2, n_max + 1, m_max + 1))
        self.file.copy_metadata(sol_dataset[0], dataset)

        for n in range(0, n_max, depth):
            sol = [self.file.load(sol_dataset[i], (slice(pow(2, i) * n, pow(2, i) * min(n + depth, n_max + 1), pow(2, i)), slice(None, None, pow(2, i)), 0)) for i in range(n_solvers)]

            if kwargs.get("exact") is not None:
                error = [sol[i] - kwargs["exact"](spatial_domain[None, :], time_domain[n:(min(n + depth, n_max + 1))][:, None]) for i in range(n_solvers)]
            else:
                error = [sol[i] - sol[i + 1] for i in range(n_solvers - 1)]

            if kwargs.get("norm") is True:
                norm_error = [np.sqrt(delta_x * np.sum(pow(error[i], 2), axis=-1)) for i in range(2)]
                self.file.save(dataset, slice(n, min(n + depth, n_max + 1)), np.log2(norm_error[0] / norm_error[1]))
            else:
                self.file.save(dataset, (0, slice(n, min(n + depth, n_max + 1)), slice(None)), error[0])
                self.file.save(dataset, (1, slice(n, min(n + depth, n_max + 1)), slice(None)), error[1])

    def get_spatial_domain(self, simul_params):
        dataset = f"sol_{simul_params}"
        return self.file.load_metadata(dataset, ["spatial_domain"])

    def get_time_domain(self, simul_params):
        dataset = f"sol_{simul_params}"
        return self.file.load_metadata(dataset, ["time_domain"])

    def get_solution(self, simul_params, t):
        dataset = f"sol_{simul_params}"
        delta_t = self.file.load_metadata(dataset, ["delta_t"])

        return self.file.load(dataset, (int(t / delta_t), slice(None), 0))

    def get_convergence(self, solver_params, t=None, **kwargs):
        dataset = (f"{"exact" if kwargs.get("exact") is not None else "self"}_"
                   f"{"norm" if kwargs.get("norm") is True else "point"}_"
                   f"conv_{solver_params}")
        delta_t = self.file.load_metadata(dataset, ["delta_t"])
        index = slice(None) if kwargs.get("norm") is True else (slice(None), int(t / delta_t), slice(None))

        return self.file.load(dataset, index)


def first_spatial_derivative(arr, h): # uses lopsided methods @ boundary
    return np.concatenate(([(-3 * arr[0] + 4 * arr[1] - arr[2]) / (2 * h)], (arr[2:] - arr[:-2]) / (2 * h), [(3 * arr[-1] - 4 * arr[-2] + arr[-3]) / (2 * h)]))


def second_spatial_derivative(arr, h, **kwargs): # uses ghost point methods @ boundary & only accepts homogeneous bc atm
    if kwargs.get('bc_left') == 'dirichlet':
        arr = np.concatenate(([2 * arr[0] - arr[1]], arr))
    elif kwargs.get('bc_left') == 'neumann':
        arr = np.concatenate(([arr[1]], arr))

    if kwargs.get('bc_right') == 'dirichlet':
        arr = np.concatenate((arr, [2 * arr[-1] - arr[-2]]))
    elif kwargs.get('bc_right') == 'neumann':
        arr = np.concatenate((arr, [arr[-2]]))

    return (arr[2:] - 2 * arr[1:-1] + arr[:-2]) / pow(h, 2)


class WESolver1D(PDESolver1D):
    def __init__(self, x, t, init, bc):
        super().__init__(x, t, init, None)
        self.bc = bc

    def solve(self, solver_params):
        courant_param, delta_x, depth = solver_params
        self.f = [lambda t, x1, x2: x2, lambda t, x1, x2: second_spatial_derivative(x1, delta_x, bc_left=self.bc[0], bc_right=self.bc[1])]
        return super().solve(solver_params)


class WESolver3D(PDESolver1D):
    def __init__(self, x, t, init):
        super().__init__(x, t, init, None)

    def solve(self, solver_params):
        courant_param, delta_x, depth = solver_params
        m_max = int(self.x / delta_x)
        spatial_domain = np.linspace(0, m_max * delta_x, m_max + 1)

        # linear WE
        self.f = [lambda t, x1, x2: x2,
                  lambda t, x1, x2: np.concatenate((
                      [2 * second_spatial_derivative(x1, delta_x, bc_left='neumann')[0]],
                      second_spatial_derivative(x1, delta_x) + first_spatial_derivative(x1, delta_x)[1:-1] / spatial_domain[1:-1],
                      [- first_spatial_derivative(x2, delta_x)[-1] - x2[-1] / spatial_domain[-1]]
                  ))]

#        # exploding non-linear WE
#        self.f = [lambda t, x1, x2: x2,
#                  lambda t, x1, x2: np.concatenate((
#                      [2 * second_spatial_derivative(x1, delta_x, bc_left='neumann')[0] + pow(x2[0], 2)],
#                      second_spatial_derivative(x1, delta_x) + first_spatial_derivative(x1, delta_x)[1:-1] /
#                      spatial_domain[1:-1] + pow(x2, 2)[1:-1],
#                      [- first_spatial_derivative(x2, delta_x)[-1] - x2[-1] / spatial_domain[-1]]
#                  ))]
#
#        # stable non-linear WE
#        self.f = [lambda t, x1, x2: x2,
#                  lambda t, x1, x2: np.concatenate((
#                      [2 * second_spatial_derivative(x1, delta_x, bc_left='neumann')[0] - (x2[0] - first_spatial_derivative(x1, delta_x)[0]) * (x2[0] + first_spatial_derivative(x1, delta_x)[0])],
#                      second_spatial_derivative(x1, delta_x) + first_spatial_derivative(x1, delta_x)[1:-1] /
#                      spatial_domain[1:-1] - (x2[1:-1] - first_spatial_derivative(x1, delta_x)[1:-1]) * (x2[1:-1] + first_spatial_derivative(x1, delta_x)[1:-1]),
#                      [- first_spatial_derivative(x2, delta_x)[-1] - x2[-1] / spatial_domain[-1]]
#                  ))]

        return super().solve(solver_params)
