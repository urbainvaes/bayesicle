import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import model_elliptic_2d as m
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)

solver = 'solver_cbs'
# model = m.__name__ + '_small_sigma'
# model = m.__name__ + '_small_epsilon'
# model = m.__name__
model = "model_elliptic_2d"

# Directory of the data
data_dir = "{}/{}/{}".format(lib_misc.data_root, solver, model)


files = glob.glob(data_dir + "/iteration-[0-9][0-9][0-9][0-9].npy")
files.sort(key=lambda f:
           int(re.search(r"iteration-([0-9]*).npy", f).group(1)))

# data_file = "/solver_eks/model_elliptic_2d/preconditioner_eks.npy"
data_file = "/solver_eks/model_elliptic_2d/preconditioner_eks.npy"
eks_data = np.load(lib_misc.data_root + data_file, allow_pickle=True)[()]
# ensembles = data['ensembles']

plotter = m.AllCoeffsPlotter(m.ip)
for i, f in enumerate(files):
    print(i)
    if i % 1 == 0:
        it_data = np.load(f, allow_pickle=True)[()]
        iteration = re.search(r"iteration-([0-9]*).npy", files[i]).group(1)
        plotter.plot(iteration, it_data)
        plt.draw()
        plt.pause(.5)

plotter.plot(iteration, eks_data)
plotter.ax.set_title("EKS")
plt.draw()
plt.pause(.5)

def update_with(plotter=None):
    fig = plotter.fig
    def update(i):
        print(i)
        it_data = np.load(files[i], allow_pickle=True)[()]
        iteration = re.search(r"iteration-([0-9]*).npy", files[i]).group(1)
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

