import numpy as np
import matplotlib.pyplot as plt

# Forward model and solver
import model_simple_1d as m
import solvers

# Set seed to zero
np.random.seed(0)

# Preconditioning
solver_cbs = solvers.CbsSolver(
    dt=np.inf,
    frac_min=5/100,
    frac_max=10/100,
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
    dt=.01,
    dt_max=1,
    reg=True,
    noise=False,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

solver_md = solvers.MdSolver(
    dt=.01,
    delta=1,
    sigma=1,
    reg=True,
    noise=False,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

# Initial ensembles
J = 10000
ensembles = np.random.rand(J, 1)

n_iter = 1000
plotter = m.Plotter(m.ip)
# plotter.fig.subplots_adjust(left=.25, bottom=.25)
# ax = plotter.fig.add_axes([0.25, 0.1, 0.65, 0.03])
# ax = plotter.fig.add_axes([0.25, 0.15, 0.65, 0.03])

for i in range(n_iter):
    print("Iteration {:04d}".format(i))
    data = solver_cbs.step(m.ip, ensembles,
                           filename="iteration-{:04d}.npy".format(i))
    ensembles = data.new_ensembles
    plotter.plot(i, data._asdict())
    if i % 1 == 0:
        plt.pause(1)
        plt.draw()
