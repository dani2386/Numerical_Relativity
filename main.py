import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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
dx = 0.1
sigma = 0.02

# Target directory
TAR_DIR = f'bin/3DHyper'

depth = 50

runs = 1

################################################################################################

def main(params, tar_dir):
    os.makedirs(f'{tar_dir}/{params}', exist_ok=True)

    solver = WESolver1D(tar_dir, X, T, init_1d, bc)
    #solver = WESolver3D(tar_dir, X, T, init_3d, bc[1])
    #solver = WESolver1DHyper(tar_dir, R, S, T, init_1d, bc)
    #solver = WESolver3DHyper(tar_dir, R, S, T, init_3d)

#    print('Solving...')
#    solver.solve(params, depth)
#    print('Done\n')
#
#    print('Solving norm convergence...')
#    solver.solve_convergence(params, depth, norm=True)
#    print('Done\n')
#
#    print('Solving point-wise convergence...')
#    solver.solve_convergence(params, depth)
#    print('Done\n')

    ################################################

    print('Plotting...')
    x_grid = solver.get_x_grid(params)
    t_grid = solver.get_t_grid(params)

    fig, ax = plt.subplots(figsize=(12, 4))
#
#    ax.plot(x_grid, solver.get_solution(params, 0), color='black', linewidth=0.8, linestyle='--')
#
#    t_frames = [1, 2, 5, 10, 25, 50, 75]
#    norm = plt.Normalize(min(t_frames), max(t_frames))
#    cmap = plt.get_cmap('bwr')
#    for t in t_frames:
#        ax.plot(x_grid, solver.get_solution(params, t), color=cmap(norm(t)), label=f't={t}s')
#
#    y_lim_up = 1.2
#    y_lim_down = -0.7
#
#    ax.plot([0, X], [0, 0], linewidth=0.8, color='black')
#    ax.plot([0, X], [y_lim_up, y_lim_up], color='black', clip_on=False)
#    ax.plot([0, X], [y_lim_down, y_lim_down], color='black', clip_on=False)
#    ax.plot([0, 0], [y_lim_down, y_lim_up], color='black', clip_on=False)
#    ax.plot([X, X], [y_lim_down, y_lim_up], color='black', clip_on=False)
#    ax.plot([R, R], [y_lim_down, y_lim_up], color='black', linestyle='--', clip_on=False)
#
#    ax.set_xlim(0, X)
#    ax.set_ylim(y_lim_down, y_lim_up)
#
#    ax.set_xticks(np.linspace(0, X, 6))
#    ax.set_yticks(np.linspace(-0.4, y_lim_up, 5))
#    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
#    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
#    ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
#    ax.tick_params(axis='both', which='major', length=7, labelsize=25)
#    ax.tick_params(axis='both', which='minor', length=5)
#
#    ax.set_xlabel(r'$\rho$', fontsize=30)
#    ax.set_ylabel(r'$u$', fontsize=30, rotation=0)
#    ax.xaxis.set_label_coords(1.05, 0.02)
#    ax.yaxis.set_label_coords(0, 1.05)
#
#    ax.legend(loc='lower right', borderpad=1, fontsize=20, ncol=2)
#
#    ax.set_frame_on(False)
#
#    plt.savefig(f'{tar_dir}/{params}/sol.pdf')
#    ax.clear()

#    fig.set_size_inches(12, 4)

    ax.plot(t_grid, solver.get_convergence(params, norm=True), color='red')

    ax.plot([0, T], [0, 0], color='black', clip_on=False)
    ax.plot([0, T], [4, 4], color='black', clip_on=False)
    ax.plot([0, 0], [0, 4], color='black', clip_on=False)
    ax.plot([T, T], [0, 4], color='black', clip_on=False)

    ax.set_xlim(0, T)
    ax.set_ylim(0, 4)

    ax.set_xticks(np.linspace(0, T, 7))
    ax.set_yticks(np.linspace(1, 4, 4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
    ax.tick_params(axis='both', which='major', length=8, labelsize=25)
    ax.tick_params(axis='both', which='minor', length=5)

    ax.set_xlabel(r'$\tau$', fontsize=30)
    ax.set_ylabel('Convergence order', fontsize=30)
    ax.xaxis.set_label_coords(1.05, 0.02)
    ax.yaxis.set_label_coords(-0.05, 0.5)

    ax.set_frame_on(False)

    plt.savefig(f'{tar_dir}/{params}/norm_conv.pdf')
    print('Done\n')

    ################################################

#    print('Animating...')
#    x_grid = solver.get_x_grid(params)
#    t_grid = solver.get_t_grid(params)
#
#    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
#
#    for j in range(2):
#        y_lim = [-0.4, 1.2] if j == 0 else [-0.001, 0.001]
#        axs[j].plot([0, X], [0, 0], color='black', linewidth=0.8, clip_on=False)
#        axs[j].plot([0, X], [y_lim[1], y_lim[1]], color='black', clip_on=False)
#        axs[j].plot([0, X], [y_lim[0], y_lim[0]], color='black', clip_on=False)
#        axs[j].plot([0, 0], [y_lim[0], y_lim[1]], color='black', clip_on=False)
#        axs[j].plot([X, X], [y_lim[0], y_lim[1]], color='black', clip_on=False)
#        axs[j].plot([R, R], [y_lim[0], y_lim[1]], color='black', linestyle='--', clip_on=False)
#
#        axs[j].set_xlim(0, X)
#        axs[j].set_ylim(y_lim[0], y_lim[1])
#
#        axs[j].set_xticks(np.linspace(0, X, 6))
#        axs[j].set_yticks(np.linspace(y_lim[0], y_lim[1], 5)) if j == 0 else axs[j].set_yticks([y_lim[0], 0, y_lim[1]])
#        axs[j].xaxis.set_minor_locator(AutoMinorLocator(4))
#        axs[j].yaxis.set_minor_locator(AutoMinorLocator(4)) if j == 0 else axs[j].yaxis.set_minor_locator(AutoMinorLocator(5))
#        axs[j].tick_params(axis='x', which='both', bottom=True, top=True, direction='in', labelbottom=False) if j == 0 \
#            else axs[j].tick_params(axis='x', which='both', bottom=True, top=True, direction='in')
#        axs[j].tick_params(axis='y', which='both', left=True, right=True, direction='in')
#        axs[j].tick_params(axis='both', which='major', length=7, labelsize=20)
#        axs[j].tick_params(axis='both', which='minor', length=5)
#
#        if j == 1: axs[j].set_xlabel(r'$\rho$', fontsize=25)
#        axs[j].set_ylabel('u', fontsize=25, rotation=0) if j == 0 else axs[j].set_ylabel('Error', fontsize=25)
#        axs[j].xaxis.set_label_coords(1.05, 0.02)
#        axs[j].yaxis.set_label_coords(-0.1, 0.5)
#
#        axs[j].set_frame_on(False)
#
#    sol_line, = axs[0].plot(x_grid, solver.get_solution(params, 0), color='red')
#    error_line_h, = axs[1].plot(x_grid, solver.get_convergence(params, 0)[0], color='red', label=r'∆x')
#    error_line_h2, = axs[1].plot(x_grid, solver.get_convergence(params, 0)[1] * 4, color='blue', label=r'∆x/2 [scaled by 4]')
#    text = axs[0].text(0.05, 0.9, '', transform=axs[0].transAxes, fontsize=20)
#
#    axs[1].legend(loc='upper left', fontsize=15, ncol=2)
#
#    def update(frame):
#        sol_line.set_ydata(solver.get_solution(params, t_grid[frame * 2]))
#        error_line_h.set_ydata(solver.get_convergence(params, t_grid[frame * 2])[0])
#        error_line_h2.set_ydata(solver.get_convergence(params, t_grid[frame * 2])[1] * 4)
#        text.set_text(f't = {t_grid[frame * 2]:.2f} s')
#        return sol_line, error_line_h, error_line_h2, text
#
#    ani = animation.FuncAnimation(fig, update, frames=int(len(t_grid) / 2) - 1, blit=True)
#    ani.save(f"{tar_dir}/{params}/wave.mp4", writer="ffmpeg", dpi=200, fps=60)
#    print('Done\n')


if __name__ == '__main__':
    os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "Times New Roman"

    params_set = [[courant_param, dx / 2**i, sigma] for i in range(runs)]

    for params_i in params_set:
        main(params_i, TAR_DIR)
