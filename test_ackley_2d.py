import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# Forward model and solver
import model_ackley_2d as m
import solvers

# Set seed to zero
np.random.seed(0)

# Solvers
solver_cbs = solvers.CbsSolver(
    dt=1,
    parallel=True,
    adaptive=True,
    dirname=m.__name__,
    opti=False)

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
    dt=.1,
    dt_max=10,
    reg=True,
    noise=False,
    parallel=True,
    adaptive=True,
    dirname=m.__name__,
    epsilon = 1)

# Number of particles
J = 10000

ensembles_x = 0 + 3*np.random.randn(J)
ensembles_y = 0 + 3*np.random.randn(J)
# ensembles_x = 0 + .5*np.random.randn(J)
# ensembles_y = 0 + .5*np.random.randn(J)
ensembles = np.vstack((ensembles_x, ensembles_y)).T


# argmin = opti.basinhopping(lambda x: m.ackley_2d(*x),
#                            x0=[0, 0], T=.5, disp=True)

# Plots
plotter = m.Plotter(m.ip, show_weights=True, cutoff=500,
                    contour=True, Lx=1, Ly=1, Lx_contours=5, Ly_contours=5)

# solver, plot_step = solver_eks, 10
solver, plot_step = solver_cbo, 10

# Main loop
n_iter = 100000
for i in range(n_iter):
    print("Iteration {:04d}".format(i))
    data = solver.step(m.ip, ensembles,
                       filename="iteration-{:04d}.npy".format(i))
    ensembles = data.new_ensembles
    plotter.plot(i, data._asdict())
    if i % plot_step == 0:
        plt.pause(1)
        plt.draw()
