import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Forward model and solver
import model_elliptic_1d as m
import solvers

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


params = [{'J': 8, 'delta': 1e-7, 'sigma': 1e-8, 'noise': False},
          {'J': 8, 'delta': 1e-7, 'sigma': 1e-2, 'noise': False},
          {'J': 8, 'delta': 1e-7, 'sigma': .5, 'noise': False},
          {'J': 8, 'delta': 1e-7, 'sigma': 1, 'noise': False},]

params = [{'J': 8, 'delta': 1, 'sigma': .1, 'noise': False},
          {'J': 8, 'delta': .1, 'sigma': .1, 'noise': False},
          {'J': 8, 'delta': 1e-7, 'sigma': .1, 'noise': False},]

params = [{'J': 8, 'delta': 1e-4, 'sigma': .01, 'noise': True},]

if __name__ == "__main__":

    # Plots
    plotter = m.Plotter(m.ip, show_weights=True,
                        contours=True, Lx=1, Ly=1, Lx_contours=5, Ly_contours=40)

    # Preconditioning
    precond_vec = np.array([0, 0])
    precond_mat = np.eye(len(precond_vec))
    use_precond = True

    if use_precond:
        # Number of particles
        J = 1000

        # ensembles_x = np.random.randn(J)
        ensembles_x = .5*np.random.randn(J)
        ensembles_y = 60 + 30*np.random.rand(J)
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
        precond_vec = np.mean(ensembles, axis=0)
        precond_mat = la.sqrtm(np.cov(ensembles.T))

    for i, p in enumerate(params):

        p['dirname'] = m.__name__ + "-noise-" + str(i)

        # Reset seed
        np.random.seed(1)

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

        n_iter = 20000
        for i in range(n_iter):
            if i % 50 == 0 and i > 0:
                plotter.plot(simulation.iteration, simulation.get_data())
                plt.pause(.1)
                plt.draw()
            data = simulation.step()
            print("Reg least squares {}".format(data.value_func))
