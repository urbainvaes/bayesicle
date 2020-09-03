import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Forward model and solver
import model_elliptic_1d as m
import solvers

# Set seed to zero
np.random.seed(0)

# Solvers
solver_cbs = solvers.CbsSolver(
    dt=1,
    frac_min=50/100,
    frac_max=55/100,
    parallel=True,
    adaptive=True,
    dirname=m.__name__,
    opti=False)

solver_cbo = solvers.CboSolver(
    dt=.1,
    parallel=True,
    adaptive=True,
    beta=1,
    lamda=1,
    sigma=.1,
    dirname=m.__name__)

solver_eks = solvers.EksSolver(
    dt=1,
    reg=True,
    noise=True,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

use_precond = False

if use_precond:
    # Number of particles
    J = 1000

    # ensembles_x = np.random.randn(J)
    ensembles_x = np.random.randn(J)
    ensembles_y = 90 + 20*np.random.rand(J)
    ensembles = np.vstack((ensembles_x, ensembles_y)).T

    plotter = m.Plotter(m.ip, show_weights=True, cutoff=500,
                        contours=True, Lx=1, Ly=1, Lx_contours=5, Ly_contours=40)

    n_iter_precond = 500
    for i in range(n_iter_precond):
        data = solver_cbs.step(m.ip, ensembles,
                               filename="iteration-{:04d}.npy".format(i))
        ensembles = data.new_ensembles
        plotter.plot(i, data._asdict())
        if i % 1 == 0:
            plt.pause(1)
            plt.draw()
    precond_vec = np.mean(ensembles, axis=0)
    precond_mat = la.sqrtm(np.cov(ensembles.T))

else:
    precond_vec = np.array([1, 1])
    precond_mat = np.eye(len(precond_vec))

solver_md = solvers.MdSolver(
    J=3,
    delta=.001,
    sigma=.001,
    reg=True,
    noise=True,
    parallel=True,
    adaptive=False,
    dt=.1,
    dt_min=1e-7,
    dt_max=1,
    precond_vec=precond_vec,
    precond_mat=precond_mat,
    dirname=m.__name__)

# Initial parameters
data = np.mean(ensembles, axis=0)

n_iter = 50000
for i in range(n_iter):
    data = solver_md.step(m.ip, data, filename="iteration-{:04d}.npy".format(i))
    print("Reg least squares {}".format(data.value_func))
    if i % 100 == 0:
        plotter.plot(i, data)
        plt.pause(1)
        plt.draw()
