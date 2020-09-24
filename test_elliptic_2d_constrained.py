import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Forward model and solver
import lib_misc
import solvers
import model_elliptic_2d_constrained as m
import scipy.linalg as la

# Set seed to zero
np.random.seed(0)

# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
# plotter = m.MainModesPlotter(m.ip, show_weights=True)
plotter = m.AllCoeffsPlotter(m.ip)

solver_cbs = solvers.CbsSolver(
    dt=np.inf,
    parallel=True,
    adaptive=True,
    opti=True,
    dirname=m.__name__)

solver_cbo = solvers.CboSolver(
    dt=.005,
    parallel=True,
    adaptive=True,
    beta=100,
    lamda=1,
    sigma=.1,
    dirname=m.__name__,
    epsilon=1)

solver_eks = solvers.EksSolver(
    dt=10,
    reg=True,
    noise=True,
    parallel=True,
    adaptive=True,
    dirname=m.__name__,
    epsilon=10)

# Number of particles
J = 128

# Initialization of ens
ensembles = np.zeros((J, m.ip.d))
ensembles[:,0] = 2*np.random.randn(J)
ensembles[:,1] = 2*np.random.randn(J)
ensembles[:,2] = np.random.rand(J)
ensembles[:,3] = np.random.rand(J)

iter_0 = 0


niter = 1000
for i in range(iter_0, iter_0 + niter):
    data = solver_eks.step(m.ip, ensembles, filename="iteration-{:04d}.npy".format(i))
    ensembles = data.new_ensembles
    plotter.plot(i, data._asdict())
    plt.pause(1)
    plt.draw()
