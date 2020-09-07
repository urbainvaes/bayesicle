import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

import model_simple_3d as m
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(12, 5))
matplotlib.rc('savefig', bbox='tight')
# matplotlib.rc('figure.subplot', hspace=.0)
# msemilogyatplotlib.rc('figure.subplot', wspace=.03)

# Parameters
solver = 'solver_md'
model = m.__name__
simul = 'precond'

fig_dir = lib_misc.fig_root + "/" + model
os.makedirs(fig_dir, exist_ok=True)


data_dir_precond = "{}/{}/{}-{}".format(lib_misc.data_root, solver, model, "precond")
data_dir_noprecond = "{}/{}/{}-{}".format(lib_misc.data_root, solver, model, "noprecond")
data_precond = np.load(data_dir_precond + "/simulation-iteration-0050.npy", allow_pickle=True)[()]
data_noprecond = np.load(data_dir_noprecond + "/simulation-iteration-2000.npy", allow_pickle=True)[()]
ensembles_precond = data_precond['ensembles'][:20,:]
ensembles_noprecond = data_noprecond['ensembles']

fig, [ax1, ax2] = plt.subplots(1, 2)
error_precond = np.abs(ensembles_precond - [1, 1, 1])
error_noprecond = np.abs(ensembles_noprecond - [1, 1, 1])

indices = np.arange(len(error_noprecond))
ax1.plot(indices, error_noprecond[:, 0], '.-', label='$|\\theta_1 - 1|$')
ax1.plot(indices, error_noprecond[:, 1], '.-', label='$|\\theta_2 - 1|$')
ax1.plot(indices, error_noprecond[:, 2], '.-', label='$|\\theta_3 - 1|$')

indices = np.arange(len(error_precond))
ax2.plot(indices, error_precond[:, 0], '.-', label='$|\\theta_1 - 1|$')
ax2.plot(indices, error_precond[:, 1], '.-', label='$|\\theta_2 - 1|$')
ax2.plot(indices, error_precond[:, 2], '.-', label='$|\\theta_3 - 1|$')

ax1.legend()
ax2.legend()
ax1.set_xlabel('$n$')
ax2.set_xlabel('$n$')

fig.savefig(fig_dir + '/error_precond.pdf')
plt.show()
