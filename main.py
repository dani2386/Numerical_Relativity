import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.WESolver import WESolver3D


# Physical parameters
X = 100
T = 120
#init = [lambda x: np.exp(-pow(x - 20, 2) / 8), lambda x: (x - 20) / 4 * np.exp(-pow(x - 20, 2) / 8)]
init = [lambda x: 0.75 * np.exp(-pow(x, 2) / 8), lambda x: 0.75 * x / 4 * np.exp(-pow(x, 2) / 8)]

# Solver parameters
courant_param = 0.5
delta_x = 0.25
depth = 50

solver_params = [courant_param, delta_x, depth]

# Analytic expression
#f = lambda x, t: np.exp(-pow(x - t - 20, 2) / 8)

# Initialize solver
solver = WESolver3D(X, T, init)

# Get solution
solver.solve(solver_params)

spatial_domain = solver.get_spatial_domain(solver_params)
time_domain = solver.get_time_domain(solver_params)

fig1, ax1 = plt.subplots()
# ax1.set_ylim(-1.2, 1.2)
ax1.set_ylim(-1.2, 2)

line1, = ax1.plot(spatial_domain, solver.get_solution(solver_params, 0))

def update1(frame):
    line1.set_ydata(solver.get_solution(solver_params, time_domain[frame]))
    return line1,

ani1 = animation.FuncAnimation(fig1, update1, frames=len(time_domain) - 1, blit=False)
ani1.save("bin/wave.mp4", writer="ffmpeg", dpi=200, fps=60)

## Get exact norm convergence
#solver.solve_convergence(solver_params, exact=f, norm=True)
#
#fig2, ax2 = plt.subplots()
#ax2.set_ylim(0, 4)
#
#ax2.plot(time_domain, solver.get_convergence(solver_params, exact=f, norm=True))
#fig2.savefig("bin/exact_norm_conv.pdf")
#
## Get exact point wise convergence
#solver.solve_convergence(solver_params, exact=f)
#
#fig3, ax3 = plt.subplots()
#ax3.set_ylim(-1.2, 1.2)
#
#line3, = ax3.plot(spatial_domain, solver.get_convergence(solver_params, 0, exact=f)[0])
#
#def update3(frame):
#    line3.set_ydata(solver.get_convergence(solver_params, time_domain[frame], exact=f)[0])
#    return line3,
#
#ani3 = animation.FuncAnimation(fig3, update3, frames=len(time_domain) - 1, blit=False)
#ani3.save("bin/exact_point_conv.mp4", writer="ffmpeg", dpi=200, fps=30)

# Get self norm convergence
#solver.solve_convergence(solver_params, norm=True)
#
#fig4, ax4 = plt.subplots()
#ax4.set_ylim(0, 4)
#
#ax4.plot(time_domain, solver.get_convergence(solver_params, norm=True))
#fig4.savefig("bin/self_norm_conv.pdf")
#
## Get self point wise convergence
#solver.solve_convergence(solver_params)
#
#fig5, ax5 = plt.subplots()
#ax5.set_ylim(-1.2, 1.2)
#
#line5, = ax5.plot(spatial_domain, solver.get_convergence(solver_params, 0)[0])
#
#def update5(frame):
#    line5.set_ydata(solver.get_convergence(solver_params, time_domain[frame])[0])
#    return line5,
#
#ani5 = animation.FuncAnimation(fig5, update5, frames=len(time_domain) - 1, blit=False)
#ani5.save("bin/self_point_conv.mp4", writer="ffmpeg", dpi=200, fps=60)

# rust
