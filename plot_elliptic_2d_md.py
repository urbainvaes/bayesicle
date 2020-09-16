import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sym

import model_elliptic_2d as m
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)
matplotlib.rc('image', cmap='viridis')

solver = 'solver_md'
model = m.__name__


# Directory of the data
data_dir = "{}/{}/{}".format(lib_misc.data_root, solver, model)

# MD maximum a posteriori estimate
u_md = np.load("{}/{}".format(data_dir, "iteration-0100-md.npy"), allow_pickle=True)[()]
u_md = u_md['theta']

# Truth
u_truth = m.u_truth

# Plot
fig, ax = plt.subplots(1, 2)
G = m.ForwardDarcy(int(np.sqrt(len(u_truth))), 0)
perm_truth = G(u_truth, return_permeability=True)
perm_truth = sym.lambdify(perm_truth.free_symbols, perm_truth)
perm_md = G(u_md, return_permeability=True)
perm_md = sym.lambdify(perm_md.free_symbols, perm_md)
grid = np.linspace(0, 1, 200)
X, Y = np.meshgrid(grid, grid)
ax[0].contourf(X, Y, perm_truth(X, Y))
ax[1].contourf(X, Y, perm_md(X, Y))
plt.show()


if __name__ == "__main__":
    G = ForwardDarcy(4, 10)
    u_truth = np.random.randn(len(G.indices))
    solution = G(u_truth, return_sol=True)
    p = fen.plot(solution)
    plt.colorbar(p)
    plt.show()
