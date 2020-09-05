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


def get_ensembles(param_set):
    data_dir = "{}/{}/{}-{}".format(lib_misc.data_root, solver, model, param_set)
    data = np.load(data_dir + "/simulation-iteration-1000.npy", allow_pickle=True)[()]
    return data['ensembles']


fig, ax = plt.subplots()
argmin, _ = m.ip.map_estimator()

n_grid = 400
x_plot = argmin[0] + 5*np.linspace(-1, 1, n_grid)
y_plot = argmin[1] + 6*np.linspace(-1, 1, n_grid)
X, Y = np.meshgrid(x_plot, y_plot)
Z = m.ip.least_squares_array(X, Y)
ax.contour(X, Y, -np.log(Z), levels=100, cmap='viridis')
ax.plot(argmin[0], argmin[1], 'kx', ms=20, mew=5)

for param_set in [1, 2, 3, 4]:
    ensembles = get_ensembles(param_set)
    ax.plot(ensembles[:,0], ensembles[:,1], '.-')

ax.set_xlim(-4.5, 2)
ax.set_ylim(101, 108)
plt.show()

fig, ax = plt.subplots()
for param_set in [1, 2, 3, 4]:
    ensembles = get_ensembles(param_set)
    ax.plot(np.arange(len(ensembles)), np.linalg.norm(ensembles - argmin, axis=1))
plt.show()
