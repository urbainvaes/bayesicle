import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Forward model and solver
import model_elliptic_1d as m
import solvers

# Set seed to zero
np.random.seed(0)

# Solvers
solver_cbs = solvers.CbsSolver(
    dt=1,
    frac_min=50/100,
    frac_max=55/100,
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
    noise=True,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

# Plots
plotter = m.Plotter(m.ip, show_weights=True,
                    contours=True, Lx=1, Ly=1, Lx_contours=5, Ly_contours=40)

# Preconditioning
use_precond = False

if use_precond:
    # Number of particles
    J = 1000

    # ensembles_x = np.random.randn(J)
    ensembles_x = np.random.randn(J)
    ensembles_y = 90 + 20*np.random.rand(J)
    ensembles = np.vstack((ensembles_x, ensembles_y)).T

    n_iter_precond = 500
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

else:
    precond_vec = np.array([1, 1])
    precond_mat = np.eye(len(precond_vec))

params1 = {'J': 8, 'delta': 1e-7, 'sigma': 1, 'noise': False, 'dirname': m.__name__ + "-1"}
params2 = {'J': 8, 'delta': 1e-7, 'sigma': .1, 'noise': False, 'dirname': m.__name__ + "-2"}
params2 = {'J': 8, 'delta': 1e-7, 'sigma': .01, 'noise': False, 'dirname': m.__name__ + "-3"}
params = [params2]

for p in params:

    solver_md = solvers.MdSolver(
        **p,
        reg=True,
        parallel=True,
        adaptive=False,
        dt=.001,
        precond_vec=precond_vec,
        precond_mat=precond_mat)

    # Initial parameters
    theta = np.array([1, 103])

    simulation = solvers.MdSimulation(
            ip=m.ip,
            initial=theta,
            solver=solver_md)

    n_iter = 1100
    for i in range(n_iter):
        if i % 50 == 0 and i > 0:
            plotter.plot(simulation.iteration, simulation.get_data())
            plt.pause(.1)
            plt.draw()
        data = simulation.step()
        print("Reg least squares {}".format(data.value_func))
