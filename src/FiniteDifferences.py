import numpy as np


def first_diff(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if (bc[0] == 'neumann' or bc[0] == 'sym') else 2 * u[1] if bc[0] == 'odd' \
            else -u[2] + 4 * u[1] - 3 * u[0]
        up_diff = 0 if (bc[1] == 'neumann' or bc[1] == 'sym') else -2 * u[-2] if bc[1] == 'odd' \
            else 3 * u[-1] - 4 * u[-2] + u[-3]

        return np.concatenate(([low_diff], u[2:] - u[:-2], [up_diff])) / (2 * dx)
    else:
        return (u[2:] - u[:-2]) / (2 * dx)


def first_diff_tem(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if (bc[0] == 'neumann' or bc[0] == 'sym') else 2 * u[1] if bc[0] == 'odd' \
            else -5 * u[0] + 11 * u[1] - 10 * u[2] + 5 * u[3] - u[4]
        up_diff = 0 if (bc[1] == 'neumann' or bc[1] == 'sym') else -2 * u[-2] if bc[1] == 'odd' \
            else 5 * u[-1] - 11 * u[-2] + 10 * u[-3] - 5 * u[-4] + u[-5]

        return np.concatenate(([low_diff], u[2:] - u[:-2], [up_diff])) / (2 * dx)
    else:
        return (u[2:] - u[:-2]) / (2 * dx)


def second_diff(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if bc[0] == 'dirichlet' else 2 * (u[1] - u[0]) if (bc[0] == 'neumann' or bc[0] == 'sym') \
            else -2 * u[0] if bc[0] == 'odd' else 2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]
        up_diff = 0 if bc[1] == 'dirichlet' else 2 * (u[-2] - u[-1]) if (bc[1] == 'neumann' or bc[1] == 'sym') \
            else -2 * u[-1] if bc[1] == 'odd' else -u[-4] + 4 * u[-3] - 5 * u[-2] + 2 * u[-1]

        return np.concatenate(([low_diff], u[2:] - 2 * u[1:-1] + u[:-2], [up_diff])) / dx**2
    else:
        return (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2


def second_diff_tem(u, dx, bc=None):
    if bc is not None:
        low_diff = 0 if bc[0] == 'dirichlet' else 2 * (u[1] - u[0]) if (bc[0] == 'neumann' or bc[0] == 'sym') \
            else -2 * u[0] if bc[0] == 'odd' else 4 * u[0] - 14 * u[1] + 20 * u[2] - 15 * u[3] + 6 * u[4] - u[5]
        up_diff = 0 if bc[1] == 'dirichlet' else 2 * (u[-2] - u[-1]) if (bc[1] == 'neumann' or bc[1] == 'sym') \
            else -2 * u[-1] if bc[1] == 'odd' else 4 * u[-1] - 14 * u[-2] + 20 * u[-3] - 15 * u[-4] + 6 * u[-5] - u[-6]

        return np.concatenate(([low_diff], u[2:] - 2 * u[1:-1] + u[:-2], [up_diff])) / dx**2
    else:
        return (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2


def fourth_diff(u, dx, bc=None):
    if bc is not None:
        low_diff = [2 * u[2] - 8 * u[1] + 6 * u[0], u[3] - 4 * u[2] + 7 * u[1] - 4 * u[0]]if bc[0] == 'sym' \
            else [6 * u[0], u[3] - 4 * u[2] + 5 * u[1] - 4 * u[0]] if bc[0] == 'odd' else [0, 0]
        up_diff = [u[-4] - 4 * u[-3] + 7 * u[-2] - 4 * u[-1], 2 * u[-3] - 8 * u[-2] + 6 * u[-1]] if bc[1] == 'sym' \
            else [u[-4] - 4 * u[-3] + 5 * u[-2] - 4 * u[-1], 6 * u[-1]] if bc[1] == 'odd' else [0, 0]

        return np.concatenate((low_diff, u[4:] - 4 * u[3:-1] + 6 * u[2:-2] - 4 * u[1:-3] + u[:-4], up_diff)) / dx**4
    else:
        return (u[4:] - 4 * u[3:-1] + 6 * u[2:-2] - 4 * u[1:-3] + u[:-4]) / dx**4


def fourth_diff_tem(u, dx, bc=None):
    pass
