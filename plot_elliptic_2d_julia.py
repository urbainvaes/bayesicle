import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)

# Directory of the data
data_dir = "data_julia/cbs/model_elliptic_2d/"


files = glob.glob(data_dir + "/ensemble-*.txt")
files.sort(key=lambda f:
           int(re.search(r"ensemble-([0-9]*).txt", f).group(1)))

fig, ax = plt.subplots()
N = 3
d = N**2
indices = [(m, n) for m in range(0, N) for n in range(0, N)]

for i, f in enumerate(files):
    print(i)
    iteration = re.search(r"ensemble-([0-9]*).txt", f).group(1)
    ensembles = np.loadtxt(f).T

    ax.clear()
    ax.set_xticks(np.arange(len(indices)))
    ax.set_xticklabels((str(i) for i in indices))
    for i, u_i in enumerate(ensembles):
        ax.plot(range(d), u_i, '.', ms=10)
    ax.plot(range(d), np.mean(ensembles, axis=0), 'bx', ms=20, mew=5)
    for i in range(d):
        ens = ensembles[:, i]
        mean_dir = np.mean(ens)
        std_dir = np.std(ens)
        y_plot = np.linspace(mean_dir - 4*std_dir, mean_dir + 4*std_dir)
        x_plot = (1/2) * np.exp(-(y_plot - mean_dir)**2/(2*std_dir**2))
        ax.plot(i + x_plot, y_plot, c='gray')
    plt.draw()
    plt.pause(.1)
