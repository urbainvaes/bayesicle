import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Forward model and solver
import model_elliptic_1d as m
import solvers

# Set seed to zero
np.random.seed(0)

# Preconditioning

solver_cbs = solvers.CbsSolver(
    dt=np.inf,
    parallel=True,
    adaptive=True,
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
    noise=False,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

# Number of particles
J = 1000

ensembles_x = 3*np.random.randn(J)
ensembles_y = 90 + 40*np.random.rand(J)
ensembles = np.vstack((ensembles_x, ensembles_y)).T


# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
# plotter = m.MainModesPlotter(m.ip, show_weights=True)
plotter = m.Plotter(m.ip, show_weights=True, cutoff=500,
                    contours=False)

n_iter_precond = 1000
for i in range(n_iter_precond):
    data = solver_eks.step(m.ip, ensembles,
                           filename="iteration-{:04d}.npy".format(i))
    ensembles = data.new_ensembles
    plotter.plot(i, data._asdict())
    if i % 50 == 0:
        plt.pause(1)
        plt.draw()

precond_vec = np.mean(ensembles, axis=0)
precond_mat = la.sqrtm(np.cov(ensembles.T))
# precond_vec = np.array([1, 1])
# precond_mat = np.eye(len(precond_vec))

# Test MD solver
solver_md = solvers.MdSolver(
    delta=.1, sigma=.0001,
    dt=.01, reg=True, noise=False,
    parallel=True,
    adaptive=False,
    dt_min=1e-7,
    precond_vec=precond_vec,
    precond_mat=precond_mat,
    dirname=m.__name__)

# Initial parameters
theta, xis = np.mean(ensembles, axis=0), np.random.randn(J, m.ip.d)
theta, xis = np.array([0, 90]), np.random.randn(J, m.ip.d)

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
