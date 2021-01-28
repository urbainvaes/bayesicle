import numpy as np
import matplotlib.pyplot as plt

# Forward model and solver
import model_ackley as m
import solvers

# Set seed to zero
np.random.seed(0)

# Number of particles
J = 1000
beta = 1

# Main loop
n_iter = 701

# Plots
plotter = m.Plotter(m.op, show_weights=True, cutoff=500, opti=True,
                    contours=True, Lx=.01, Ly=.01, Lx_contours=3, Ly_contours=3)

nsimul = 10

if __name__ == "__main__":

    for isimul in range(nsimul):

        # Solvers
        solver_cbs = solvers.CbsSolver(
            dt=np.inf,
            # dt=.02,
            parallel=True,
            beta=beta,
            adaptive=True,
            dirname=m.__name__ + f"/adaptive-simulation-{isimul}",
            # adaptive=False,
            # dirname=m.__name__ + f"/simulation-{isimul}",
            opti=True)

        spread = 4
        ensembles_x = spread*(np.random.randn() + np.random.randn(J))
        ensembles_y = spread*(np.random.randn() + np.random.randn(J))
        ensembles = np.vstack((ensembles_x, ensembles_y)).T
        solver, plot_step = solver_cbs, 1

        for i in range(n_iter):
            print("Iteration {:04d}".format(i))
            data = solver.step(m.op, ensembles,
                    filename="iteration-{:04d}.npy".format(i))
            ensembles = data.new_ensembles
            plotter.plot(i, data._asdict())
            if i % plot_step == 0:
                plt.pause(1)
                plt.draw()
