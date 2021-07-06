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

preconditioning = True
if preconditioning:

    precond_vec_file = "data_julia/precond_vec.txt"
    precond_mat_file = "data_julia/precond_mat.txt"

    if os.path.exists(precond_vec_file):
        precond_vec = np.loadtxt(precond_vec_file)
        precond_mat = np.loadtxt(precond_mat_file)

precond_mat = la.sqrtm(precond_mat)

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
