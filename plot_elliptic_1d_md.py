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

# Parameters
solver = 'solver_md'
model = m.__name__
simul = 'noise'

fig_dir = lib_misc.fig_root + "/" + model
os.makedirs(fig_dir, exist_ok=True)


def get_ensembles(param_set):
    data_dir = "{}/{}/{}-{}-{}".format(lib_misc.data_root, solver, model, simul, param_set)
    data = np.load(data_dir + "/simulation-iteration-20000.npy", allow_pickle=True)[()]
    return data['ensembles']

if t.params[0]['noise']:
    matplotlib.rc('figure', figsize=(15, 5))

    n_grid = 100
    x_plot = -3.7 + 2.7*np.linspace(0, 1, n_grid)
    y_plot = 102.5 + 3.5*np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = m.ip.least_squares_array(X, Y)

    ensembles = get_ensembles(0)
    ensembles = ensembles[1000:]
    mean, cov = np.mean(ensembles, axis=0), \
                np.cov(ensembles.T)
    moments = m.ip.moments_posterior()
    print(mean, cov)
    print(moments)

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    ax1.contour(X, Y, Z, levels=100, cmap='viridis')
    ax2.contour(X, Y, Z, levels=100, cmap='viridis')
    ax3.contour(X, Y, Z, levels=100, cmap='viridis')

    ax1.plot(ensembles[:,0], ensembles[:,1], '.', ms=1)

    kernel = scipy.stats.gaussian_kde([ensembles[:,0], ensembles[:,1]])
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    ax2.contourf(X, Y, Z, levels=60, cmap='binary')
    ax2.set_xticks([])
    ax2.set_yticks([])

    Z_post = m.ip.posterior(X, Y)
    ax3.contourf(X, Y, Z_post, levels=60, cmap='binary')
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.savefig(fig_dir + '/posterior.pdf')

else:
    n_grid = 400
    x_plot = -4 + 6.5*np.linspace(0, 1, n_grid)
    y_plot = 101 + 7*np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = m.ip.least_squares_array(X, Y)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    argmin, _ = m.ip.map_estimator()

    contour = ax1.contour(X, Y, Z, levels=100, cmap='viridis')
    ax1.plot(argmin[0], argmin[1], 'kx', ms=20, mew=5)

    for i, params in enumerate(t.params):
        ensembles = get_ensembles(i)
        error = np.linalg.norm(ensembles - argmin, axis=1)
        label = r"$\{} = {}$".format(simul, params[simul])
        ax1.plot(ensembles[:,0], ensembles[:,1], '.-', label=label)
        ax2.semilogy(np.arange(len(ensembles)), error, label=label)
        ax1.set_xlim(x_plot[0], x_plot[-1])
        ax1.set_ylim(y_plot[0], y_plot[-1])

    ax1.legend()
    ax2.legend()
    fig1.colorbar(contour)

    fig1.savefig(fig_dir + '/{}_trajectories.pdf'.format(simul))
    fig2.savefig(fig_dir + '/{}_convergence.pdf'.format(simul))

plt.show()
