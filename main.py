import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.WESolver import *


################################################################################################

# Physical parameters
X = 100
R = 60
S = 100
T = 120
init_1d = [lambda x: np.exp(-pow(x - 20, 2) / 8), lambda x: (x - 20) / 4 * np.exp(-pow(x - 20, 2) / 8)]
init_3d = [lambda x: np.exp(-pow(x, 2) / 8), lambda x: x / 4 * np.exp(-pow(x, 2) / 8)]
bc = ['dirichlet', 'dirichlet']

# Solver params
courant_param = 0.5
dx = 0.25
sigma = 0.02

# Target directory
tar_dir_ = f'bin/3DHyperNEW_no_rescale'

depth = 50

runs = 1

################################################################################################

def main(params, tar_dir):
    os.makedirs(f'{tar_dir}/{params}', exist_ok=True)

    #solver = WESolver1D(tar_dir, X, T, init_1d, bc)
    #solver = WESolver3D(tar_dir, X, T, init_3d)
    #solver = WESolver1DHyper(tar_dir, R, S, T, init_1d, bc)
    solver = WESolver3DHyper(tar_dir, R, S, T, init_3d)

    x_grid = solver.get_x_grid(params)
    t_grid = solver.get_t_grid(params)

    ################################################

    print('Solving...')
    solver.solve(params, depth)
    print('Done\n')

    ################################################

    print('Plotting...')
    fig2, ax2 = plt.subplots()
    ax2.set_ylim(-1.2, 1.2)
    ax2.axvline(x=R, color='r', linestyle='--')

    line2, = ax2.plot(x_grid, solver.get_solution(params, 0))
    text2 = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

    def update2(frame):
        line2.set_ydata(solver.get_solution(params, t_grid[frame]))
        text2.set_text(f't = {t_grid[frame]} s')
        return line2, text2

    ani2 = animation.FuncAnimation(fig2, update2, frames=len(t_grid) - 1, blit=False)
    ani2.save(f"{tar_dir}/{params}/wave.mp4", writer="ffmpeg", dpi=200, fps=60)
    print('Done\n')

    ################################################

    print('Solving norm convergence...')
    solver.solve_convergence(params, depth, norm=True)
    print('Done\n')

    ################################################

    print('Plotting norm convergence...')
    fig3, ax3 = plt.subplots()
    ax3.set_ylim(0, 4)

    ax3.plot(t_grid, solver.get_convergence(params, norm=True))
    fig3.savefig(f"{tar_dir}/{params}/norm_self_conv.png")
    print('Done\n')

    ################################################

    print('Solving point-wise convergence...')
    solver.solve_convergence(params, depth)
    print('Done\n')

    ################################################

    print('Plotting point-wise convergence...')
    fig4, ax4 = plt.subplots()
    ax4.axvline(x=R, color='r', linestyle='--')

    line4, = ax4.plot(x_grid, solver.get_convergence(params, 0)[0])
    line5, = ax4.plot(x_grid, 4 * solver.get_convergence(params, 0)[1])
    text4 = ax4.text(0.05, 0.9, '', transform=ax4.transAxes)

    def update4(frame):
        line4.set_ydata(solver.get_convergence(params, t_grid[frame])[0])
        line5.set_ydata(4 * solver.get_convergence(params, t_grid[frame])[1])
        text4.set_text(f't = {t_grid[frame]} s')
        return line4, line5, text4

    ani4 = animation.FuncAnimation(fig4, update4, frames=len(t_grid) - 1, blit=False)
    ani4.save(f"{tar_dir}/{params}/point_self_conv.mp4", writer="ffmpeg", dpi=200, fps=60)
    print('Done\n')

    ################################################


if __name__ == '__main__':
    params_set = [[courant_param, dx / 2**i, sigma] for i in range(runs)]

    for params_ in params_set:
        main(params_, tar_dir_)
