import os.path
import numpy as np
import matplotlib.pyplot as plt
import solvers
import lib_misc

import model_simple_3d as m
import scipy.linalg as la

# Make simulation deterministic
np.random.seed(0)

plotter = m.MainModesPlotter(m.ip, show_weights=True, cutoff=10000,
                             contours=False)
# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
# plotter = m.MainModesPlotter(m.ip, show_weights=True)


# PRECONDITIONING {{{1

precond_vec = np.zeros(m.ip.d)
precond_mat = np.eye(m.ip.d)

preconditioning = False
if preconditioning:

    precond_vec_file = lib_misc.data_root + "/precond_vec.npy"
    precond_mat_file = lib_misc.data_root + "/precond_mat.npy"

    if os.path.exists(precond_vec_file):
        precond_vec = np.load(precond_vec_file)
        precond_mat = np.load(precond_mat_file)

    else:

        # Number of particles
        J = 1000

        solver = solvers.EksSolver(
            dt=.5,
            reg=False,
            noise=True,
            parallel=True,
            adaptive=True,
            dirname=m.__name__)

        # Initial parameters
        ensembles = np.random.randn(J, m.ip.d)

        n_iter_precond = 1000
        for i in range(n_iter_precond):
            data = solver.step(m.ip, ensembles,
                               filename="iteration-{:04d}.npy".format(i))
            ensembles = data.new_ensembles
            plotter.plot(i, data._asdict())
            print(np.mean(ensembles, axis=0))
            if i % 100 == 0:
                plt.pause(1)
                plt.draw()

        precond_vec = np.mean(ensembles, axis=0)
        precond_mat = la.sqrtm(np.cov(ensembles.T))

        np.save(precond_vec_file, precond_vec)
        np.save(precond_mat_file, precond_mat)

# MULTISCALE METHOD {{{1
# Test MD solver
solver_md = solvers.MdSolver(
    J=8,
    delta=1e-5,
    sigma=1e-5,
    dt=(1 if preconditioning else 1/m.k**4),
    reg=False,
    noise=False,
    parallel=True,
    adaptive=False,
    dt_min=1e-7,
    dt_max=.1,
    precond_vec=precond_vec,
    precond_mat=precond_mat,
    dirname=m.__name__ + ("-precond" if preconditioning else "-noprecond"))

# Initial parameters
theta = np.zeros(m.ip.d)

simulation = solvers.MdSimulation(
        ip=m.ip,
        initial=theta,
        solver=solver_md)

n_iter = (200 if preconditioning else 4000)
for i in range(n_iter):
    if i % 100 == 0:
        plotter.plot(i, simulation.get_data())
        plt.pause(1)
        plt.draw()

    data = simulation.step()
    print("Reg least squares {}".format(data.value_func))
