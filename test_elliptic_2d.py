import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Forward model and solver
# import model_elliptic_2d as m
import model_simple_3d as m
import model_elliptic_1d as m

import solver_cbs
import solver_eks
import solver_md

# Set seed to zero
np.random.seed(0)

# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
# plotter = m.MainModesPlotter(m.ip, show_weights=True)
plotter = m.Plotter()

solver = solver_cbs.CbsSolver(
    dt=np.inf,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

# solver = solver_eks.EksSolver(
#     dt=1,
#     reg=True,
#     noise=True,
#     parallel=True,
#     adaptive=True,
#     dirname=m.__name__)

# solver = solver_md.MdSolver(
#     delta=.001, sigma=.001,
#     dt=.05, reg=False, noise=False,
#     parallel=True,
#     adaptive=True,
#     dt_min=1e-10,
#     dirname=m.__name__)


# Number of particles
J = 10

# Initial parameters
theta, xis = np.zeros(m.ip.d), np.random.randn(J, m.ip.d)
ensembles = theta, xis

# Initialization of ens
ensembles = 2*np.random.randn(J, m.ip.d)
all_data = {'solver': 'md', 'ensembles': theta.reshape(1, m.ip.d)}


def plot(i):
    # global ensembles
    # data = solver.step(m.ip, ensembles,
    #                    filename="iteration-{:04d}.npy".format(i))
    # ensembles = data.new_ensembles
    # plotter.plot(i, data._asdict())
    global theta, xis
    data = solver.step(m.ip, theta, xis,
                       filename="iteration-{:04d}.npy".format(i))
    theta, xis = data.new_theta, data.new_xis
    all_data['ensembles'] = np.vstack((all_data['ensembles'], theta))
    plotter.plot(i, all_data)
    print(theta)
    plt.pause(1)
    plt.draw()


n_figs = 500
anim = animation.FuncAnimation(plotter.fig, plot, n_figs,
                               interval=600, init_func=lambda: None)
plt.show()
