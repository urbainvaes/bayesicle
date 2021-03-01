import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ

# Forward model and solver
import model_ackley_2d as m

dt=.005
beta=1

def calculate_weighted_meancov(m, C):
    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    Z = integ.dblquad(lambda y, x: np.exp(-beta*m.ackley_2d(x, y)),
                      xmin, xmax, ymin, ymax)[0]


# Number of particles
J = 1000
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

