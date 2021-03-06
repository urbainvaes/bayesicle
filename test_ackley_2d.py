import numpy as np
import matplotlib.pyplot as plt

# Forward model and solver
import model_ackley as m
import solvers

# Set seed to zero
np.random.seed(0)

# Solvers
solver_cbs = solvers.CbsSolver(
    dt=np.inf,
    parallel=True,
    beta=1,
    adaptive=False,
    dirname=m.__name__,
    opti=True)

solver_cbo = solvers.CboSolver(
    dt=.005,
    parallel=True,
    adaptive=False,
    beta=1000,
    lamda=1,
    sigma=.1,
    dirname=m.__name__,
    epsilon=1)

solver_eks = solvers.EksSolver(
    dt=.01,
    dt_max=10,
    reg=True,
    noise=False,
    parallel=True,
    adaptive=False,
    dirname=m.__name__,
    epsilon=1)

# Plots
plotter = m.Plotter(m.ip, show_weights=True, cutoff=500, opti=True,
                    contours=True, Lx=.01, Ly=.01, Lx_contours=5, Ly_contours=5)

# Number of particles
J = 1000
# ensembles_x = -1 + 1*np.random.randn(J)
# ensembles_y = -1 + 1*np.random.randn(J)
ensembles_x = 2*np.random.randn(J)
ensembles_y = 2*np.random.randn(J)
ensembles = np.vstack((ensembles_x, ensembles_y)).T

# solver, plot_step = solver_eks, 10
solver, plot_step = solver_cbs, 1
# solver, plot_step = solver_cbo, 1

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
