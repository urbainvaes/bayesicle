import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import model_fokker_planck as m
import lib_misc

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)

G = m.ForwardFokkerPlanck(4)
data = np.load("/home/urbain/postdoc/cbs/data/solver_cbs/" +
               "model_fokker_planck/iteration-0051.npy",
               allow_pickle=True)[()]
control = np.mean(data['ensembles'], axis=0)
control = np.reshape(control, (G.nc, G.N))
G.time = np.linspace(0, G.T, 200)
result = G.solve_state(control)

# fig, ax = plt.subplots()
# for i, r in enumerate(result):
#     if i % 10 != 0:
#         continue
#     ax.clear()
#     G.quad_vi.plot(r, ax=ax)
#     ax.set_title("Time: {}".format(G.time[i]))
#     plt.draw()
#     plt.pause(.2)

# import ipdb; ipdb.set_trace()

# solver, step = 'solver_eks', 100
solver, step = 'solver_cbs', 1
model = m.__name__


# Directory of the data
data_dir = "{}/{}/{}".format(lib_misc.data_root, solver, model)


files = glob.glob(data_dir + "/iteration-[0-9]*.npy")
files.sort(key=lambda f:
           int(re.search(r"iteration-([0-9]*).npy", f).group(1)))

def update_with(a_plotter):
    def update(i):
        # iter_0 = 800
        # i = step*i
        # i = iter_0 + i
        it_data = np.load(files[i], allow_pickle=True)[()]
        iteration = re.search(r"iteration-([0-9]*).npy", files[i]).group(1)
        a_plotter.plot(iteration, it_data)
        plt.pause(.5)
    return update


animate = animation.FuncAnimation
plotter = m.Plotter(m.ip, show_weights=True)
# plotter = m.AllCoeffsPlotter(m.ip, show_weights=True)
anim = animate(plotter.fig, update_with(plotter), len(files),
               repeat=False)
plt.show()
