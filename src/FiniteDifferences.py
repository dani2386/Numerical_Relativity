import numpy as np
import matplotlib.pyplot as plt
import os


def first_diff(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if (bc[0] == 'neumann' or bc[0] == 'sym') else (-u[2] + 4 * u[1] - 3 * u[0]) / (2 * dx)
        up_diff = 0 if (bc[1] == 'neumann' or bc[1] == 'sym') else (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * dx)

        return np.concatenate(([low_diff], (u[2:] - u[:-2]) / (2 * dx), [up_diff]))
    else:
        return (u[2:] - u[:-2]) / (2 * dx)


def first_diff_tem(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if (bc[0] == 'neumann' or bc[0] == 'sym') else (-5 * u[0] + 11 * u[1] - 10 * u[2] + 5 * u[3] - u[4]) / (2 * dx)
        up_diff = 0 if (bc[1] == 'neumann' or bc[1] == 'sym') else (5 * u[-1] - 11 * u[-2] + 10 * u[-3] - 5 * u[-4] + u[-5]) / (2 * dx)

        return np.concatenate(([low_diff], (u[2:] - u[:-2]) / (2 * dx), [up_diff]))
    else:
        return (u[2:] - u[:-2]) / (2 * dx)


def second_diff(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if bc[0] == 'dirichlet' else 2 * (u[1] - u[0]) / dx**2 if (bc[0] == 'neumann' or bc[0] == 'sym') \
            else (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx**2
        up_diff = 0 if bc[1] == 'dirichlet' else 2 * (u[-2] - u[-1]) / dx**2 if (bc[1] == 'neumann' or bc[1] == 'sym') \
            else (-u[-4] + 4 * u[-3] - 5 * u[-2] + 2 * u[-1]) / dx**2

        return np.concatenate(([low_diff], (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2, [up_diff]))
    else:
        return (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2


def second_diff_tem(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if bc[0] == 'dirichlet' else 2 * (u[1] - u[0]) / dx**2 if (bc[0] == 'neumann' or bc[0] == 'sym') \
            else (4 * u[0] - 14 * u[1] + 20 * u[2] - 15 * u[3] + 6 * u[4] - u[5]) / dx ** 2
        up_diff = 0 if bc[1] == 'dirichlet' else 2 * (u[-2] - u[-1]) / dx**2 if (bc[1] == 'neumann' or bc[1] == 'sym') \
            else (4 * u[-1] - 14 * u[-2] + 20 * u[-3] - 15 * u[-4] + 6 * u[-5] - u[-6]) / dx ** 2

        return np.concatenate(([low_diff], (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2, [up_diff]))
    else:
        return (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2


def tem_test():
    x = 100
    dx = 0.25
    m_max = int(x / dx)
    x_grid = np.linspace(0, m_max * dx, m_max + 1)

    func = lambda u: np.exp(-pow(u - 90, 2) / 32)
    func_first_deriv = lambda u: - (u - 90) / 16 * np.exp(-pow(u - 90, 2) / 32)
    func_second_deriv = lambda u: ((u - 90)**2 - 16) / 256 * np.exp(-pow(u - 90, 2) / 32)

    os.makedirs('../bin/tests/tem_test', exist_ok=True)

    fig1, ax1 = plt.subplots()
    ax1.plot(x_grid, func(x_grid))
    ax1.plot(x_grid, func_first_deriv(x_grid))
    ax1.plot(x_grid, func_second_deriv(x_grid))
    fig1.savefig('../bin/tests/tem_test/func_deriv.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(x_grid, first_diff(func(x_grid), dx, [None, None]) - func_first_deriv(x_grid))
    ax2.plot(x_grid, first_diff_tem(func(x_grid), dx, [None, None]) - func_first_deriv(x_grid))
    fig2.savefig('../bin/tests/tem_test/first_diff.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(x_grid, func_second_deriv(x_grid) - second_diff(func(x_grid), dx, [None, None]))
    ax3.plot(x_grid, func_second_deriv(x_grid) - second_diff_tem(func(x_grid), dx, [None, None]))
    fig3.savefig('../bin/tests/tem_test/second_diff.png')


def diff_test():
    x = 100
    dx = 0.25
    m_max = int(x / dx)
    x_grid = np.linspace(0, m_max * dx, m_max + 1)

    func = lambda u: np.exp(-pow(u - 10, 2) / 32)
    func_first_deriv = lambda u: - (u - 10) / 16 * np.exp(-pow(u - 10, 2) / 32)
    func_second_deriv = lambda u: ((u - 10)**2 - 16) / 256 * np.exp(-pow(u - 10, 2) / 32)

    os.makedirs('../bin/tests/diff_test', exist_ok=True)

    fig1, ax1 = plt.subplots()
    ax1.plot(x_grid, func(x_grid))
    ax1.plot(x_grid, func_first_deriv(x_grid))
    ax1.plot(x_grid, func_second_deriv(x_grid))
    fig1.savefig('../bin/tests/diff_test/func_deriv.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(x_grid, func_first_deriv(x_grid))
    ax2.plot(x_grid, first_diff_tem(func(x_grid), dx, ['sym', None]))
    ax2.plot(x_grid, first_diff_tem(func(x_grid), dx, [None, None]))
    fig2.savefig('../bin/tests/diff_test/func_first_diff.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(x_grid, func_second_deriv(x_grid))
    ax3.plot(x_grid, second_diff_tem(func(x_grid), dx, ['sym', None]))
    ax3.plot(x_grid, second_diff_tem(func(x_grid), dx, [None, None]))
    fig3.savefig('../bin/tests/diff_test/func_second_diff.png')


if __name__ == '__main__':
    diff_test()
