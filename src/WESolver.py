import numpy as np
from src.PDESolver import PDESolver
from src.FiniteDifferences import first_diff_tem, second_diff_tem


class WESolver1D(PDESolver):
    def __init__(self, path, x, t, init, bc):
        super().__init__(path, x, t, init, bc,None)

    def solve(self, params, depth):
        courant_param, dx, sigma = params

        self.f = [
            lambda t, u, v: v,
            lambda t, u, v: second_diff_tem(u, dx, self.bc)
        ]

        return super().solve(params, depth)


class WESolver3D(PDESolver):
    def __init__(self, path, x, t, init):
        super().__init__(path, x, t, init, ['sym', None],None)

    def solve(self, params, depth):
        courant_param, dx, sigma, dt, m_max, n_max, x_grid = self.params(params)[:-1]

        # linear
        self.f = [
            lambda t, u, v: v,
            lambda t, u, v: np.concatenate((
                [3 * second_diff_tem(u, dx, self.bc)[0]],
                second_diff_tem(u, dx) + 2 * first_diff_tem(u, dx) / x_grid[1:-1],
                [-first_diff_tem(v, dx, self.bc)[-1] - v[-1] / x_grid[-1]]
            ))
        ]

#        # exploding non-linear
#        self.f = [
#            lambda t, u, v: v,
#            lambda t, u, v: np.concatenate((
#                [3 * second_diff_tem(u, dx, self.bc)[0] + v[0]**2],
#                second_diff_tem(u, dx) + 2 * first_diff_tem(u, dx) / x_grid[1:-1] + v[1:-1]**2,
#                [- first_diff_tem(v, dx, [None, None])[-1] - v[-1] / x_grid[-1]]
#            ))
#        ]

#        # stable non-linear
#        self.f = [
#            lambda t, u, v: v,
#            lambda t, u, v: np.concatenate((
#                [3 * second_diff_tem(u, dx, self.bc)[0] - (v[0] - first_diff_tem(u, dx, self.bc)[0]) * (v[0] + first_diff_tem(u, dx, self.bc)[0])],
#                second_diff_tem(u, dx) + 2 * first_diff_tem(u, dx) / x_grid[1:-1] - (v[1:-1] - first_diff_tem(u, dx)) * (v[1:-1] + first_diff_tem(u, dx)),
#               [- first_diff_tem(v, dx, [None, None])[-1] - v[-1] / x_grid[-1]]
#            ))
#        ]

        return super().solve(params, depth)


class WESolver1DHyper(PDESolver):
    def __init__(self, path, r, s, t, init, bc):
        self.r = r
        self.s = s
        super().__init__(path, s, t, init, bc, None)

    def solve(self, params, depth):
        courant_param, dx, sigma, dt, m_max, n_max, x_grid = self.params(params)[:-1]

        omega = 1 - np.heaviside(x_grid[:-1] - self.r, 1) * (x_grid[:-1] - self.r)**2 / (self.s - self.r)**2
        l = 1 + np.heaviside(x_grid[:-1] - self.r, 1) * (x_grid[:-1] - self.r)**2 / (self.s - self.r)**2
        h = 1 - omega**2 / l
        d_h = (-2 * (x_grid[:-1] - self.r) / (self.s - self.r)**2) * np.heaviside(x_grid[:-1] - self.r, 1) * (2 * omega / l + omega**2 / l**2)

        self.f = [
            lambda t, u, v: v,
            lambda t, u, v: np.concatenate((
                (omega**2 / ((1 - h**2) * l)) *
                ((omega**2 / l) * second_diff_tem(u, dx, self.bc)[:-1]
                 - d_h * first_diff_tem(u, dx, self.bc)[:-1] - d_h * v[:-1]
                 - 2 * h * first_diff_tem(v, dx, self.bc)[:-1]),
                [-first_diff_tem(v, dx, [None, None])[-1] - v[-1] / x_grid[-1]]
            ))
        ]

        return super().solve(params, depth)


class WESolver3DHyper(PDESolver):
    def __init__(self, path, r, s, t, init):
        self.r = r
        self.s = s
        super().__init__(path, s, t, init, ['sym', None], None)

    def solve(self, params, depth):
        courant_param, dx, sigma, dt, m_max, n_max, x_grid = self.params(params)[:-1]

        omega = 1 - np.heaviside(x_grid[1:-1] - self.r, 1) * ((x_grid[1:-1] - self.r) / (self.s - self.r))**4
        l = 1 + np.heaviside(x_grid[1:-1] - self.r, 1) * (x_grid[1:-1] - self.r)**3 * (3 * x_grid[1:-1] + self.r) / (self.s - self.r)**4
        d_omega = - np.heaviside(x_grid[1:-1] - self.r, 1) * 4 * (x_grid[1:-1] - self.r)**3 / (self.s - self.r)**4
        d_l = np.heaviside(x_grid[1:-1] - self.r, 1) * 12 * x_grid[1:-1] * (x_grid[1:-1] - self.r)**2 / (self.r - self.s)**4
        ricci = 6 * omega * (omega * d_l - 2 * l * d_omega) / (x_grid[1:-1]**2 * l**3)

        self.f = [
            lambda t, u, v: v,
            lambda t, u, v: np.concatenate((
                [2 * second_diff_tem(u, dx, self.bc)[0]],
                (l**2 / (omega**2 - 2 * l)) *
                (- 2 * ((omega**2 - l) / l**2) * first_diff_tem(v, dx)
                 + (1 / (l * x_grid[1:-1]) - (l * (omega**2 - 2 * x_grid[1:-1] * omega * d_omega) - x_grid[1:-1] * omega**2 * d_l) / (l**3 * x_grid[1:-1])) * v[1:-1]
                 - (omega**2 / l**2) * second_diff_tem(u, dx)
                 - ((l * (omega**2 - 2 * x_grid[1:-1] * omega * d_omega) - x_grid[1:-1] * omega**2 * d_l) / (l**3 * x_grid[1:-1])) * first_diff_tem(u, dx)
                 + ricci / 6 * v[1:-1]),
                [-first_diff_tem(v, dx, [None, None])[-1] - v[-1] / x_grid[-1]]
            ))
        ]

        return super().solve(params, depth)
