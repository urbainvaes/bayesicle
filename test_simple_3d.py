import numpy as np
import matplotlib.pyplot as plt
import solvers
import model_simple_3d as m
import scipy.linalg as la

# Make simulation deterministic
np.random.seed(0)

solver = solvers.EksSolver(
    dt=1,
    reg=True,
    noise=True,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

plotter = m.MainModesPlotter(m.ip, show_weights=True, cutoff=500,
                             contours=False)
# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
# plotter = m.MainModesPlotter(m.ip, show_weights=True)

# Number of particles
J = 1000

# PRECONDITIONING {{{1

precond_vec = np.zeros(m.ip.d)
precond_mat = np.eye(m.ip.d)

preconditioning = False
if preconditioning:
    # Initial parameters
    ensembles = np.random.randn(J, m.ip.d)

    n_iter_precond = 200
    for i in range(n_iter_precond):
        data = solver.step(m.ip, ensembles,
                           filename="iteration-{:04d}.npy".format(i))
        ensembles = data.new_ensembles
        plotter.plot(i, data._asdict())
        if i % 10 == 0:
            plt.pause(1)
            plt.draw()

    precond_vec = np.mean(ensembles, axis=0)
    precond_mat = la.sqrtm(np.cov(ensembles.T))


# MULTISCALE METHOD {{{1
# Test MD solver
solver_md = solvers.MdSolver(
    delta=1e-8, sigma=np.sqrt(1e-5),
    dt=1e-4, reg=True, noise=False,
    parallel=True,
    adaptive=False,
    dt_min=1e-7,
    precond_vec=precond_vec,
    precond_mat=precond_mat,
    dirname=m.__name__)

# Initial parameters
theta = np.zeros(m.ip.d)
xis = np.random.randn(3, m.ip.d)

# Dictionary to store all iterations
all_data = {'solver': 'md', 'ensembles': theta.reshape(1, m.ip.d)}

n_iter = 50000
for i in range(n_iter):
    data = solver_md.step(m.ip, theta, xis,
                          filename="iteration-{:04d}.npy".format(i))
    theta, xis = data.new_theta, data.new_xis
    all_data['ensembles'] = np.vstack((all_data['ensembles'], theta))
    print("Reg least squares {}".format(data.value_func))
    if i % 100 == 0:
        plotter.plot(i, all_data)
        plt.pause(1)
        plt.draw()
