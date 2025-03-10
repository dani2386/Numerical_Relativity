import numpy as np


class WESolver:
    def __init__(self, x, delta_x, courant_param):
        self._delta_x = delta_x
        self._delta_t = courant_param * self._delta_x
        self._m = int(x / self._delta_x)
        self._f = [lambda u: u, lambda u: self.second_spatial_derivative(u)]

    def spatial_domain(self):
        return np.linspace(0, self._m * self._delta_x, self._m + 1)

    def second_spatial_derivative(self, arr):
        return 1 / pow(self._delta_x, 2) * np.concatenate(([0], arr[:-2] - 2 * arr[1:-1] + arr[2:], [0]))

    def solve(self, init, t):
        n = int(t / self._delta_t)
        grid = np.empty([n + 1, self._m + 1, 2])

        grid[0] = np.column_stack((init[0](self.spatial_domain()), init[1](self.spatial_domain())))

        for i in range(n):
            k1_1 = self._f[0](grid[i, :, 1])
            k1_2 = self._f[1](grid[i, :, 0])

            k2_1 = self._f[0](grid[i, :, 1] + self._delta_t / 2 * k1_2)
            k2_2 = self._f[1](grid[i, :, 0] + self._delta_t / 2 * k1_1)

            k3_1 = self._f[0](grid[i, :, 1] + self._delta_t / 2 * k2_2)
            k3_2 = self._f[1](grid[i, :, 0] + self._delta_t / 2 * k2_1)

            k4_1 = self._f[0](grid[i, :, 1] + self._delta_t * k3_2)
            k4_2 = self._f[1](grid[i, :, 0] + self._delta_t * k3_1)

            grid[i + 1, :, 0] = grid[i, :, 0] + self._delta_t / 6 * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1)
            grid[i + 1, :, 1] = grid[i, :, 1] + self._delta_t / 6 * (k1_2 + 2 * k2_2 + 2 * k3_2 + k4_2)

        return grid[-1, :, 0]
