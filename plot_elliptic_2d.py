import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import model_elliptic_2d_constrained as m
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)

solver = 'solver_cbo'
# model = m.__name__ + '_small_sigma'
# model = m.__name__ + '_small_epsilon'
model = m.__name__


# Directory of the data
data_dir = "{}/{}/{}".format(lib_misc.data_root, solver, model)


files = glob.glob(data_dir + "/iteration-[0-9][0-9][0-9][0-9].npy")
files.sort(key=lambda f:
           int(re.search(r"iteration-([0-9]*).npy", f).group(1)))


def update_with(plotter=None):
    if plotter is None:
        fig, ax = plt.subplots()
    else:
        fig = plotter.fig

    def update(i):
        print(i)
        it_data = np.load(files[i], allow_pickle=True)[()]
        iteration = re.search(r"iteration-([0-9]*).npy", files[i]).group(1)
        if plotter is None:
            ax.clear()
            ax.set_aspect('equal')
            ensembles = it_data['ensembles']
            theta = np.linspace(0, 2*np.pi)
            ax.plot(.5 + .2*np.cos(theta), .5 + .2*np.sin(theta))
            ax.plot(ensembles[:, 2], ensembles[:, 3], '.')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            plotter.plot(iteration, it_data)
        plt.pause(.5)
    return update, fig


animate = animation.FuncAnimation

plotter_2 = m.AllCoeffsPlotter(m.ip)
update, fig = update_with(plotter_2)
anim_1 = animate(fig, update, len(files), repeat=False)
plt.show()

update, fig = update_with(None)
anim_1 = animate(fig, update, len(files), repeat=False)
plt.show()

