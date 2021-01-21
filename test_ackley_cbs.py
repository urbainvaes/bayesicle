import numpy as np
import matplotlib.pyplot as plt

# Forward model and solver
import model_ackley_2d as m
import solvers

# Set seed to zero
np.random.seed(0)

# Number of particles
J = 100

# Plots
plotter = m.Plotter(m.ip, show_weights=True, cutoff=500, opti=True,
                    contours=True, Lx=.01, Ly=.01, Lx_contours=5, Ly_contours=5)

nsimul = 10
for isimul in range(nsimul):

    # Solvers
    solver_cbs = solvers.CbsSolver(
        dt=np.inf,
        parallel=True,
        beta=1,
        adaptive=False,
        dirname=m.__name__ + f"/simulation-{isimul}",
        opti=True)

    ensembles_x = 10*(np.random.randn() + np.random.randn(J))
    ensembles_y = 10*(np.random.randn() + np.random.randn(J))
    ensembles = np.vstack((ensembles_x, ensembles_y)).T
    solver, plot_step = solver_cbs, 1

    # Main loop
    n_iter = 50
    for i in range(n_iter):
        print(i)
        print("Iteration {:04d}".format(i))
        data = solver.step(m.ip, ensembles,
                filename="iteration-{:04d}.npy".format(i))
        ensembles = data.new_ensembles
        # plotter.plot(i, data._asdict())
        # if i % plot_step == 0:
        #     plt.pause(1)
        #     plt.draw()