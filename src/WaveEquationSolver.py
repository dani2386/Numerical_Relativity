import numpy as np


class WaveEquationSolver:
    def __init__(self, n, m, delta_x, delta_t):
        self._N = n
        self._M = m
        self._delta_x = delta_x
        self._delta_t = delta_t
        self._courant_param = delta_t / delta_x
        self._grid = np.zeros([self._N + 1, self._M + 1, 2])

    @staticmethod
    def _second_spatial_derivative(arr):
        return np.concatenate(([0], arr[:-2] - 2 * arr[1:-1] + arr[2:], [0]))

    def spatial_domain(self):
        return np.linspace(0, self._M * self._delta_x, self._M + 1)

    def solve_fd(self, f, g, theta):
        self._grid[0, :, 0] = f(self.spatial_domain())
        self._grid[1, :, 0] = (self._grid[0, :, 0] + self._delta_t * g(self.spatial_domain())
                               + pow(self._courant_param, 2) / 2 * self._second_spatial_derivative(self._grid[0, :, 0]))

        a = (np.diag(2 * (1 / pow(self._courant_param, 2) + theta) * np.ones(self._M + 1), k=0)
             + np.diag(-theta * np.ones(self._M), k=1) + np.diag(-theta * np.ones(self._M), k=-1))
        b = np.diag(-2 * np.ones(self._M + 1), k=0) + np.diag(np.ones(self._M), k=1) + np.diag(np.ones(self._M), k=-1)
        c = 2 * (np.diag(np.ones(self._M + 1), k=0) + np.linalg.inv(a) @ b)

        for n in range(1, self._N):
            self._grid[n + 1, :, 0] = c @ self._grid[n, :, 0] - self._grid[n - 1, : 0]

    def solve_rk4(self, f, g):
        self._grid[0] = np.column_stack((f(self.spatial_domain()), g(self.spatial_domain())))

        _f = [lambda x: x, lambda x: 1 / pow(self._delta_x, 2) * self._second_spatial_derivative(x)]

        for n in range(self._N):
            k1_1 = _f[0](self._grid[n, :, 1])
            k1_2 = _f[1](self._grid[n, :, 0])

            k2_1 = _f[0](self._grid[n, :, 1] + self._delta_t / 2 * k1_2)
            k2_2 = _f[1](self._grid[n, :, 0] + self._delta_t / 2 * k1_1)

            k3_1 = _f[0](self._grid[n, :, 1] + self._delta_t / 2 * k2_2)
            k3_2 = _f[1](self._grid[n, :, 0] + self._delta_t / 2 * k2_1)

            k4_1 = _f[0](self._grid[n, :, 1] + self._delta_t * k3_2)
            k4_2 = _f[1](self._grid[n, :, 0] + self._delta_t * k3_1)

            self._grid[n + 1, :, 0] = self._grid[n, :, 0] + self._delta_t / 6 * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)
            self._grid[n + 1, :, 1] = self._grid[n, :, 1] + self._delta_t / 6 * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

    def solution_n(self, n):
        return self._grid[n, :, 0]
