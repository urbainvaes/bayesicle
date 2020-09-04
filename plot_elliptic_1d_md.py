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

# Plotter
plotter = m.Plotter(m.ip, show_weights=True, contours=True,
                    Lx=1, Ly=1, Lx_contours=5, Ly_contours=40)
import ipdb; ipdb.set_trace()

plotter.plot(it, data)
# np.save
