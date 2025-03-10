import numpy as np
from src.WESolver import WESolver


class WEConvergenceTester:
    def __init__(self, x, init, t, f):
        self._x = x
        self._init = init
        self._t = t
        self._f = f

    def norm_convergence(self, courant_param, step_i, n):
        norm_error = np.empty(n)

        for i in range(n):
            solver = WESolver(self._x, step_i / pow(2, i), courant_param)
            error = self._f(solver.spatial_domain(), self._t) - solver.solve(self._init, self._t)
            norm_error[i] = np.sqrt(step_i / pow(2, i) * np.sum(pow(error, 2)))

        return np.log2(norm_error[:-1] / norm_error[1:])

    def point_wise_convergence(self):
        return

    def self_convergence(self):
        return
