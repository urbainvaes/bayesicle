import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats


import model_elliptic_1d as m
import test_elliptic_1d as t
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(9, 7))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.0)
matplotlib.rc('figure.subplot', wspace=.03)
matplotlib.rc('figure', figsize=(15, 5))

# Parameters
solver = 'solver_cbs'
model = m.__name__

fig_dir = lib_misc.fig_root + "/" + model
os.makedirs(fig_dir, exist_ok=True)

data_dir = "{}/{}/{}".format(lib_misc.data_root, solver, model)
ensembles = np.load(data_dir + "/iteration-0100.npy", allow_pickle=True)[()]['ensembles']

mean, cov = np.mean(ensembles, axis=0), np.cov(ensembles.T)
moments = m.ip.moments_posterior()
print(mean, cov)
print(moments)


n_grid = 100
x_plot = -3.7 + 2.7*np.linspace(0, 1, n_grid)
y_plot = 102.5 + 3.5*np.linspace(0, 1, n_grid)
X, Y = np.meshgrid(x_plot, y_plot)
Z = m.ip.least_squares_array(X, Y)

fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
ax1.contour(X, Y, Z, levels=100, cmap='viridis')
ax2.contour(X, Y, Z, levels=100, cmap='viridis')
ax3.contour(X, Y, Z, levels=100, cmap='viridis')
ax1.plot(ensembles[:,0], ensembles[:,1], '.', ms=1)

rv = scipy.stats.multivariate_normal(mean, cov)
ax2.contourf(X, Y, rv.pdf(np.dstack((X, Y))), cmap='binary')
ax2.set_xticks([])
ax2.set_yticks([])

Z_post = m.ip.posterior(X, Y)
ax3.contourf(X, Y, Z_post, cmap='binary')
ax3.set_xticks([])
ax3.set_yticks([])

fig.savefig(fig_dir + '/posterior_cbs.pdf')
plt.show()
