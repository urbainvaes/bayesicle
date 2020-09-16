import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Forward model and solver
import lib_misc
import solvers
import model_elliptic_2d as m
import scipy.linalg as la

# Set seed to zero
np.random.seed(0)

# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
# plotter = m.MainModesPlotter(m.ip, show_weights=True)
plotter = m.AllCoeffsPlotter(m.ip)

solver = solvers.CbsSolver(
    dt=np.inf,
    parallel=True,
    adaptive=True,
    opti=False,
    dirname=m.__name__)

solver = solvers.EksSolver(
    dt=1,
    reg=True,
    noise=True,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

# solver = solver_md.MdSolver(
#     delta=.001, sigma=.001,
#     dt=.05, reg=False, noise=False,
#     parallel=True,
#     adaptive=True,
#     dt_min=1e-10,
#     dirname=m.__name__)

# Number of particles
J = 256

# Initialization of ens
ensembles = 2*np.random.randn(J, m.ip.d)
iter_0 = 0

if False:

    # Load from file
    data_file = "/solver_cbs/model_elliptic_2d/iteration-0050-cbs-extended.npy"
    data_file = "/solver_eks/model_elliptic_2d/iteration-0150.npy"
    data = np.load(lib_misc.data_root + data_file, allow_pickle=True)[()]
    ensembles = data['ensembles']
    iter_0 = 151


    niter = 1000
    for i in range(iter_0, iter_0 + niter):
        data = solver.step(m.ip, ensembles, filename="iteration-{:04d}.npy".format(i))
        ensembles = data.new_ensembles
        plotter.plot(i, data._asdict())
        plt.pause(1)
        plt.draw()


# Calculate preconditioner based on final iteration of EKS
data_file = "/solver_eks/model_elliptic_2d/preconditioner_eks.npy"
data = np.load(lib_misc.data_root + data_file, allow_pickle=True)[()]
ensembles = data['ensembles']
precond_vec = np.mean(ensembles, axis=0)
precond_mat = la.sqrtm(np.cov(ensembles.T))

# MULTISCALE METHOD {{{1
# Test MD solver
solver_md = solvers.MdSolver(
    J=8,
    delta=1e-5,
    sigma=1e-5,
    dt=.1,
    reg=True,
    noise=False,
    parallel=True,
    adaptive=False,
    dt_min=1e-7,
    dt_max=.1,
    precond_vec=precond_vec,
    precond_mat=precond_mat,
    dirname=m.__name__)

# Initial parameters
theta = precond_vec

simulation = solvers.MdSimulation(
        ip=m.ip,
        initial=theta,
        solver=solver_md)

n_iter = 200
for i in range(n_iter):
    if i > 1 and i % 1 == 0:
        plotter.plot(i, simulation.get_data())
        plt.pause(1)
        plt.draw()

    data = simulation.step()
    print("Reg least squares {}".format(data.value_func))
