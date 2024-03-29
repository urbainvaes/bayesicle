import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(9, 7))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.0)
matplotlib.rc('figure.subplot', wspace=.03)
matplotlib.rc('figure', figsize=(12, 6))

import model_ackley as ackley

# νs = [2., 1., .5, .25, .125, .0625]
νs = [10., 1., .1, .01]
εs = [1]
# νs = [10., 1., .1]
# εs = [10., 1., .1]
nparticles = 1000

for ε in εs:
    for ν in νs:
        # data = np.loadtxt(f"data/limits-nu={ν}-epsilon={ε}-J={nparticles}.txt")
        data = np.loadtxt(f"data/limits-nu={ν}-J={nparticles}.txt")
        fig, ax = plt.subplots()
        grid = np.arange(-3, 3, .01)
        xgrid, ygrid = ackley.a + grid, ackley.b + grid
        x, y = np.meshgrid(xgrid, ygrid)
        radius = 3*np.sqrt(2); constraint = lambda x, y: x**2 + y**2 - radius**2
        ax.contour(x, y, constraint(x, y), levels=[0], linewidths=5, colors='black', alpha=.2)
        ax.contourf(x, y, ackley.ackley_2d(x, y), levels=30, cmap='viridis', alpha=.6)
        ax.plot(data[0, :], data[1, :], 'r.', ms=10)
        ax.plot(3, 3, 'kx', ms=20)
        ax.set_xlim((xgrid[0], xgrid[-1]))
        ax.set_ylim((ygrid[0], ygrid[-1]))
        props = dict(boxstyle='round', facecolor='cyan', alpha=0.9)
        # ax.text(.02, .98, rf"$\nu={ν}$, $\varepsilon={ε}$, $J={nparticles}$", fontsize=18, bbox=props,
        ax.text(.02, .98, rf"$\nu={ν}$, $J={nparticles}$", fontsize=18, bbox=props,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
        ax.set_aspect('equal')
        # fig.savefig(f"figures/limits-nu={ν}-epsilon={ε}-J={nparticles}.pdf", bbox_inches="tight")
        fig.savefig(f"figures/limits-nu={ν}-J={nparticles}.pdf", bbox_inches="tight")
        plt.show()

ν, ε = 1, 1
iters = [1, 100, *range(200, 1000, 200)]
for i in iters:
    data = np.loadtxt(f"data/niter={i}.txt")
    # data = np.loadtxt(f"data/limits-nu={ν}-J={nparticles}.txt")
    fig, ax = plt.subplots()
    grid = np.arange(-5, 5, .01)
    xgrid, ygrid = ackley.a + grid, ackley.b + grid
    x, y = np.meshgrid(xgrid, ygrid)
    radius = 3*np.sqrt(2); constraint = lambda x, y: x**2 + y**2 - radius**2
    ax.contour(x, y, constraint(x, y), levels=[0], linewidths=5, colors='black', alpha=.2)
    ax.contourf(x, y, ackley.ackley_2d(x, y), levels=30, cmap='viridis', alpha=.6)
    ax.plot(data[0, :], data[1, :], 'r.', ms=10)
    ax.plot(3, 3, 'kx', ms=20)
    ax.set_xlim((xgrid[0], xgrid[-1]))
    ax.set_ylim((ygrid[0], ygrid[-1]))
    props = dict(boxstyle='round', facecolor='cyan', alpha=0.9)
    ax.text(.02, .98, rf"$i={i}$", fontsize=18, bbox=props,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes)
    ax.set_aspect('equal')
    fig.savefig(f"figures/iter-{i}.pdf", bbox_inches="tight")
    plt.show()
