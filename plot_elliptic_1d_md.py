import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_elliptic_1d as m
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)

# Parameters
solver = 'solver_md'
model = m.__name__

# Directory of the data
param_set, it = 1, 1000
data_dir = "{}/{}/{}-{}".format(lib_misc.data_root, solver, model, param_set)

# Load data
data = np.load(data_dir + "/simulation-iteration-1000.npy", allow_pickle=True)[()]
ensembles = data['ensembles']

# Plotter
fig, ax = plt.subplots()
argmin, _ = m.ip.map_estimator()

n_grid = 400
x_plot = argmin[0] + 5*np.linspace(-1, 1, n_grid)
y_plot = argmin[1] + 6*np.linspace(-1, 1, n_grid)
X, Y = np.meshgrid(x_plot, y_plot)
Z = m.ip.least_squares_array(X, Y)
ax.contour(X, Y, -np.log(Z), levels=100, cmap='viridis')
ax.plot(ensembles[:,0], ensembles[:,1], '.-')
ax.plot(argmin[0], argmin[1], 'kx', ms=20, mew=5)
ax.set_xlim(-4.5, 2)
ax.set_ylim(101, 108)
plt.show()
