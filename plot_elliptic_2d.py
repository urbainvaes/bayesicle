import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# import model_elliptic_2d as m
import model_simple_3d as m
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)

solver = 'solver_cbs'
model = m.__name__


# Directory of the data
data_dir = "{}/{}/{}".format(lib_misc.data_root, solver, model)


files = glob.glob(data_dir + "/iteration-[0-9]*.npy")
files.sort(key=lambda f:
           int(re.search(r"iteration-([0-9]*).npy", f).group(1)))


def update_with(plotter):
    def update(i):
        it_data = np.load(files[i], allow_pickle=True)[()]
        iteration = re.search(r"iteration-([0-9]*).npy", files[i]).group(1)
        plotter.plot(iteration, it_data)
        plt.pause(.5)
    return update


animate = animation.FuncAnimation
plotter_1 = m.MainModesPlotter(m.ip.unknown, show_weights=True)
anim_1 = animate(plotter_1.fig, update_with(plotter_1), len(files),
                 repeat=False)
plt.show()

plotter_2 = m.AllCoeffsPlotter(m.u, solver, coeffs=[0, 3, 4])
anim_2 = animate(plotter_2.fig, update_with(plotter_2), len(files),
                 repeat=False)
plt.show()
