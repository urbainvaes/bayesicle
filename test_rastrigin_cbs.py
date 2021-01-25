import numpy as np
import matplotlib.pyplot as plt
import model_rastrigin_nd as m
# import model_ackley_2d as m
import solvers

# Set seed to zero
np.random.seed(0)

# Number of particles
J = 1000
beta = .1

# Main loop
n_iter = 701

# Plots
Lx, Ly = 2, 2
plotter = m.Plotter(m.op, show_weights=False, cutoff=500, opti=True,
                    contours=True, Lx=.01, Ly=.01, Lx_contours=Lx, Ly_contours=Ly)

if __name__ == "__main__":

    # Solvers
    solver_cbs = solvers.CbsSolver(
        dt=np.inf,
        # frac_min=45/100,
        # frac_max=55/100,
        # dt=.02,
        parallel=True,
        beta=beta,
        dirname=m.__name__ + f"/test-simulation",
        adaptive=True,
        # adaptive=False,
        # dirname=m.__name__ + f"/simulation-{isimul}",
        opti=True)
    solver, plot_step = solver_cbs, 1

    ensembles = 10*np.random.randn(J, m.n)

    if m.n == 2:
        vec = np.array([Lx, Ly])
        ensembles = vec*np.random.rand(J, m.n)

    for i in range(n_iter):
        print("Iteration {:04d}".format(i))
        data = solver.step(m.op, ensembles,
                filename="iteration-{:04d}.npy".format(i))
        ensembles = data.new_ensembles
        plotter.plot(i, data._asdict())
        if i % plot_step == 0:
            plt.pause(1)
            plt.draw()