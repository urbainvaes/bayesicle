import numpy as np
import matplotlib.pyplot as plt
import lib_misc

# Forward model and solver
import model_fokker_planck as m
import solvers

# Set seed to zero
np.random.seed(0)

# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
# plotter = m.MainModesPlotter(m.ip, show_weights=True)
# plotter = m.Plotter(m.ip)
plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)

# solver = solvers.CbsSolver(
#     dt=np.inf,
#     parallel=True,
#     adaptive=True,
#     dirname=m.__name__,
#     opti=True)

solver = solvers.EksSolver(
    dt=1,
    dt_max=10**6,
    reg=True,
    noise=False,
    parallel=True,
    adaptive=True,
    dirname=m.__name__)

# Number of particles
J = 50

iter_0 = 827
data = np.load(lib_misc.data_root +
               "/solver_eks/model_fokker_planck/iteration-{:04d}.npy"
               .format(iter_0), allow_pickle=True)
ensembles = data[()]['ensembles']


# Initialization of ensembles
# ensembles = np.random.randn(J, m.ip.d)

n_iter = 500
for i in iter_0 + np.arange(n_iter):
    print("Iteration {:04d}".format(i))
    data = solver.step(m.ip, ensembles,
                       filename="iteration-{:04d}.npy".format(i))
    ensembles = data.new_ensembles
    plotter.plot(i, data._asdict())
    plt.pause(1)
    plt.draw()
