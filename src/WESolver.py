import numpy as np
from src.PDESolver import PDESolver
from src.FiniteDifferences import first_spatial_derivative, second_spatial_derivative


class WESolver1D(PDESolver):
    def __init__(self, x, t, init, bc):
        super().__init__(x, t, init, None)
        self.bc = bc

    def solve(self, solver_params):
        courant_param, delta_x, depth = solver_params
        self.f = [lambda t, x1, x2: x2, lambda t, x1, x2: second_spatial_derivative(x1, delta_x, bc_left=self.bc[0], bc_right=self.bc[1])]
        return super().solve(solver_params)


class WESolver3D(PDESolver):
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