import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

import model_ackley as m
import test_ackley_cbs as t
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(9, 7))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.0)
matplotlib.rc('figure.subplot', wspace=.03)
matplotlib.rc('figure', figsize=(12, 6))

# Parameters
solver = 'solver_cbs'
model = m.__name__

fig_dir = lib_misc.fig_root + "/" + model
os.makedirs(fig_dir, exist_ok=True)


def get_wmean_trajectory(isimul, adaptive=True):
    prefix = "adaptive-" if adaptive else ""
    data_dir = "{}/{}/{}/{}simulation-{}".format(lib_misc.data_root,
                                                 solver, model, prefix, isimul)
    wmean = np.zeros((t.n_iter, 2))
    for i in range(t.n_iter):
        data = np.load(data_dir + "/iteration-{:04d}.npy".format(i),
                       allow_pickle=True)[()]
        wmean[i] = data['weights'].dot(data['ensembles'])
    return wmean

fig, ax = plt.subplots()
for isimul in range(t.nsimul):
    wmean = get_wmean_trajectory(isimul, adaptive=True)
    ax.semilogy(np.arange(wmean.shape[0]), np.linalg.norm(wmean, axis=1), '.-')
    wmean = get_wmean_trajectory(isimul, adaptive=False)
    ax.semilogy(np.arange(wmean.shape[0]), np.linalg.norm(wmean, axis=1), '.-')
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$|M_{\beta}(\rho^J_n) - \theta_*|$")
plt.savefig("convergence_ackley.pdf")
fig.show()

fig, ax = plt.subplots()
for isimul in range(t.nsimul):
# for isimul in range(5):
    wmean = get_wmean_trajectory(isimul, adaptive=True)
    ax.plot(wmean[:, 0], wmean[:, 1], 'k.-')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
n_grid = 500
x_plot = 5*np.linspace(-1, 1, n_grid)
y_plot = 5*np.linspace(-1, 1, n_grid)
X, Y = np.meshgrid(x_plot, y_plot)
Z = m.op.objective_array(X, Y)
ax.contourf(X, Y, Z, levels=20, cmap='viridis')
ax.contour(X, Y, Z, levels=20, colors='black')
ax.set_xlim(-5, 5)
plt.savefig("convergence_ackley.pdf")
fig.show()

# mean, cov = np.mean(ensembles, axis=0), np.cov(ensembles.T)
# moments = m.ip.moments_posterior()
# print(mean, cov)
# print(moments)

# fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
# ax1.contour(X, Y, Z, levels=100, cmap='viridis')
# ax2.contour(X, Y, Z, levels=100, cmap='viridis')
# ax3.contour(X, Y, Z, levels=100, cmap='viridis')
# ax1.plot(ensembles[:,0], ensembles[:,1], '.', ms=1)

# rv = scipy.stats.multivariate_normal(mean, cov)
# ax2.contourf(X, Y, rv.pdf(np.dstack((X, Y))), cmap='binary')
# ax2.set_xticks([])
# ax2.set_yticks([])

# Z_post = m.ip.posterior(X, Y)
# ax3.contourf(X, Y, Z_post, cmap='binary')
# ax3.set_xticks([])
# ax3.set_yticks([])

# fig.savefig(fig_dir + '/posterior_cbs.pdf')
# plt.show()
