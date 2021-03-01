import re
import os
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
matplotlib.rc('figure.subplot', wspace=.06)
matplotlib.rc('image', cmap='viridis')

solver = 'solver_md'
model = m.__name__


# Directory of the data
data_dir = "{}/{}/{}".format(lib_misc.data_root, solver, model)
fig_dir = lib_misc.fig_root + "/" + model
os.makedirs(fig_dir, exist_ok=True)

# MD maximum a posteriori estimate
u_md = np.load("{}/{}".format(data_dir, "iteration-0100-md-nonextended.npy"), allow_pickle=True)[()]
u_md = np.load("{}/{}".format(data_dir, "iteration-0100-md-extended.npy"), allow_pickle=True)[()]
u_md = u_md['theta']

# Truth
u_truth = m.u_truth

# Plot (Optimization)
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
G = m.ForwardDarcy(int(np.sqrt(len(u_truth))), 0)
perm_truth = G(u_truth, return_permeability=True)
variables = list(perm_truth.free_symbols)
variables.sort(key=lambda x: str(x))
perm_truth = sym.lambdify(variables, perm_truth)
perm_md = G(u_md, return_permeability=True)
perm_md = sym.lambdify(variables, perm_md)
grid = np.linspace(0, 1, 200)
X, Y = np.meshgrid(grid, grid)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
contourf0 = ax[0].contourf(X, Y, perm_truth(X, Y))
contourf1 = ax[1].contourf(X, Y, perm_md(X, Y))
contourf1.set_array(contourf0.get_array)
fig.colorbar(contourf1, ax=ax, fraction=0.022, pad=.03)
ax[0].set_xlabel('$x_0$')
ax[1].set_xlabel('$x_0$')
ax[0].set_ylabel('$x_1$')
fig.savefig(fig_dir + '/log_permeability.pdf')
plt.show()

λs = np.asarray(G.eig_v[:9], dtype=float)
norm1 = np.linalg.norm(u_truth[:9])
error1 = np.linalg.norm(u_md - u_truth[:9]) / norm1
norm2 = np.linalg.norm(u_truth[:9]*np.sqrt(λs))
error2 = np.linalg.norm((u_md - u_truth[:9])*np.sqrt(λs)) / norm2

# Plot (Sampling)
data_file = "simulation-iteration-1000-extended.npy"
data_file = "simulation-iteration-1000-nonextended.npy"
simulation_data = np.load("{}/{}".format(data_dir, data_file), allow_pickle=True)[()]

# Calculate the error in number of std deviations
u_truth = m.u_truth[:9]
std = np.std(simulation_data['ensembles'], axis=0)
diff = np.abs(np.mean(simulation_data['ensembles'], axis=0) - u_truth)
print(diff/std)

plotter = m.AllCoeffsPlotter(m.ip, show_text=False)
plotter.plot(0, simulation_data)
plotter.ax.set_xlabel("Karhunen--Loève coefficients")
plotter.fig.savefig(fig_dir + '/samples_kl_coeffs.pdf')
plt.show()
