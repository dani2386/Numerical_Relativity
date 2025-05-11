import numpy as np


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
