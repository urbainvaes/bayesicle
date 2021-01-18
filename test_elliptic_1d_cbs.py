import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Forward model and solver
import model_elliptic_1d as m
import solvers

# Solvers
solver_cbs = solvers.CbsSolver(
    dt=1,
    # frac_min=50/100,
    # frac_max=55/100,
    parallel=True,
    beta=1,
    adaptive=False,
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

if __name__ == "__main__":

    # Plots
    plotter = m.Plotter(m.ip, show_weights=True,
                        contours=True, Lx=1, Ly=1, Lx_contours=5, Ly_contours=40)

    # Number of particles
    J = 1000

    # ensembles_x = np.random.randn(J)
    ensembles_x = np.random.randn(J)
    ensembles_y = 90 + 20*np.random.rand(J)
    ensembles = np.vstack((ensembles_x, ensembles_y)).T

    n_iter_precond = 500
    for i in range(n_iter_precond):
        data = solver_cbs.step(m.ip, ensembles,
                               filename="iteration-{:04d}.npy".format(i))
        ensembles = data.new_ensembles
        plotter.plot(i, data._asdict())
        if i % 1 == 0:
            plt.pause(1)
            plt.draw()
