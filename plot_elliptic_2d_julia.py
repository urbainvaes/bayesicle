import re
import glob
import numpy as np
import sympy as sym
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import array_to_latex as a2l

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(18, 11))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.3)

# Directory of the data
data_dir = "data_julia/cbs/model_elliptic_2d/"
data_dir_eks = "data_julia/eks/model_elliptic_2d/"
data_dir_aldi = "data_julia/aldi/model_elliptic_2d/"
data_dir_multiscale = "data_julia/multiscale/model_elliptic_2d/"
data_dir_pCN = "data_julia/pCN/model_elliptic_2d/"


files = glob.glob(data_dir + "/ensemble-*.txt")
files.sort(key=lambda f:
           int(re.search(r"ensemble-([0-9]*).txt", f).group(1)))

files_eks = glob.glob(data_dir_eks + "/all_ensembles-*.txt")
files_eks.sort(key=lambda f:
               int(re.search(r"all_ensembles-([0-9]*).txt", f).group(1)))

files_multiscale = glob.glob(data_dir_multiscale + "/ensemble-*.txt")
files_multiscale.sort(key=lambda f:
               int(re.search(r"ensemble-([0-9]*).txt", f).group(1)))

files_pCN = glob.glob(data_dir_pCN + "/ensemble-*.txt")
files_pCN.sort(key=lambda f:
               int(re.search(r"ensemble-([0-9]*).txt", f).group(1)))

files_aldi = glob.glob(data_dir_aldi + "/ensemble-*.txt")
files_aldi.sort(key=lambda f:
               int(re.search(r"ensemble-([0-9]*).txt", f).group(1)))

utruth = np.loadtxt(data_dir + "/utruth.txt")
umap = np.mean(np.loadtxt(files[-1]), axis=1)
umap_eks = np.mean(np.loadtxt(files_eks[-1]), axis=1)

a2l.to_ltx(utruth, frmt = '{:6.2f}', arraytype = 'array')
a2l.to_ltx(umap, frmt = '{:6.2f}', arraytype = 'array')
a2l.to_ltx(umap_eks, frmt = '{:6.2f}', arraytype = 'array')

fig, ax = plt.subplots()
d = len(utruth)
N = int(np.sqrt(d))
indices = [(m, n) for m in range(0, N) for n in range(0, N)]
indices.sort(key=lambda i: (max(i), sum(i), i[0]))

def update(index):
    f = files[index]
    ax.clear()
    iteration = re.search(r"ensemble-([0-9]*).txt", f).group(1)
    ensembles = np.loadtxt(f).T

    def plot_marginals(ensembles, label, shape, color, plot_points=True, gaussian=False):

        if isinstance(ensembles, str):
            ensembles = np.loadtxt(ensembles).T

        for i in range(d):
            ens = ensembles[:, i]
            mean_dir, std_dir = np.mean(ens), np.std(ens)
            y_plot = np.linspace(mean_dir - 4*std_dir, mean_dir + 4*std_dir)
            kernel = scipy.stats.gaussian_kde(ens)
            x_plot = kernel([y_plot]).T
            x_plot = (1/2) * x_plot / np.max(x_plot)
            kwarg = {'label': label} if i == 0 else {}
            ax.plot(i + x_plot, y_plot, shape, c=color, **kwarg)
            ax.plot(i + 0*x_plot, y_plot, '-', c='gray')



    # Truth
    ax.plot(range(d), utruth, 'bx', ms=20, mew=5, label="Truth")

    # ax.clear()
    # ax.set_xticks(np.arange(len(indices)))
    # ax.set_xticklabels((str(i) for i in indices))
    # ax.set_title(f"Iteration {iteration}")
    # for i, u_i in enumerate(ensembles):
    #     ax.plot(range(d), u_i, '.', color='blue', ms=5)
    # # ax.plot(range(d), np.mean(ensembles, axis=0), 'bx', ms=20, mew=5)
    # for i in range(d):
    #     ens = ensembles[:, i]
    #     mean_dir = np.mean(ens)
    #     std_dir = np.std(ens)
    #     y_plot = np.linspace(mean_dir - 4*std_dir, mean_dir + 4*std_dir)
    #     x_plot = (1/2) * np.exp(-(y_plot - mean_dir)**2/(2*std_dir**2))
    #     kwarg = {'label': "CBS"} if i == 0 else {}
    #     ax.plot(i + x_plot, y_plot, c='blue', **kwarg)

    # f = files_eks[100]
    # ensembles = np.loadtxt(f).T
    # ax.set_xticks(np.arange(len(indices)))
    # ax.set_xticklabels((str(i) for i in indices))
    # for i, u_i in enumerate(ensembles):
    #     ax.plot(np.arange(d) - .1, u_i, '.', color='orange', ms=5)
    # for i in range(d):
    #     ens = ensembles[:, i]
    #     mean_dir = np.mean(ens)
    #     std_dir = np.std(ens)
    #     y_plot = np.linspace(mean_dir - 4*std_dir, mean_dir + 4*std_dir)
    #     kernel = scipy.stats.gaussian_kde(ens)
    #     x_plot = kernel([y_plot]).T
    #     x_plot = (1/2) * x_plot / np.max(x_plot)
    #     kwarg = {'label': "Approximate posterior (EKS)"} if i == 0 else {}
    #     ax.plot(i + x_plot, y_plot, '--', c='orange', **kwarg)

    # ax.plot(range(d), umap, '+', color='darkgreen', ms=20, mew=5, label="MAP CBS")

    ensembles = np.loadtxt(f).T
    ax.set_xticks(np.arange(len(indices)))
    ax.set_xticklabels((str(i) for i in indices))
    ax.set_xlabel("Karhunen--Loève coefficients")
    # for i, u_i in enumerate(ensembles):
        # ax.plot(np.arange(d), u_i, '.', ms=5)

    f = files_multiscale[-1]
    plot_marginals(f, label="Approximate posterior (multiscale)", shape="--",
                   color="black", plot_points=False, gaussian=False)

    f = files_pCN[-1]
    plot_marginals(f, label="Approximate posterior (MCMC)", shape="-",
                   color="green", plot_points=False, gaussian=False)

    # f = files_aldi[index]
    all_ensembles = np.vstack([np.loadtxt(f).T for f in files_aldi])
    J = np.shape(np.loadtxt(files_aldi[0]))[0]
    all_ensembles = all_ensembles[100*J:]

    # all_ensembles = all_ensembles[300:,:]

    argument = all_ensembles
    plot_marginals(argument, label="Approximate posterior (ALDI)", shape="-",
                   color="red", plot_points=False, gaussian=False)

    # f = files_eks[-1]
    # plot_marginals(f, label="Approximate posterior (EKS)", shape="-",
    #                color="red", plot_points=False, gaussian=False)

    # f = files_eks[100]
    # ensembles = np.loadtxt(f).T

    # for i in range(d):
    #     ens = ensembles[:, i]
    #     mean_dir = np.mean(ens)
    #     std_dir = np.std(ens)
    #     y_plot = np.linspace(mean_dir - 4*std_dir, mean_dir + 4*std_dir)
    #     kernel = scipy.stats.gaussian_kde(ens)
    #     x_plot = kernel([y_plot]).T
    #     x_plot = (1/2) * x_plot / np.max(x_plot)
    #     kwarg = {'label': "Approximate posterior (multiscale)"} if i == 0 else {}
    #     ax.plot(i + x_plot, y_plot, '--', c='black', **kwarg)
    #     ax.plot(i + 0*x_plot, y_plot, '-', c='gray')

    # MAP
    # ax.plot(range(d), umap_eks, '.', color="red", ms=10, mew=5, label="Approximate MAP estimator (multiscale)")

    plt.legend()
    plt.draw()
    plt.pause(.1)

update(100)
plt.savefig("posterior_multiscale_elliptic2d.pdf")

animate = animation.FuncAnimation
anim = animate(fig, update, len(files), repeat=False)
plt.show()

# writer = animation.writers['ffmpeg'](fps=2, bitrate=500, codec='libvpx-vp9')
# anim.save(f'elliptic_new.webm', writer=writer, dpi=500)

# Plot (Optimization)
x, y = sym.symbols('x[0], x[1]', real=True, positive=True)
f = sym.Function('f')(x, y)

# Precision (inverse covariance) operator, when α = 1
tau, alpha = 3, 2
precision = (- sym.diff(f, x, x) - sym.diff(f, y, y) + tau**2*f)

# Eigenfunctions of the covariance operator
eig_f = [sym.cos(i[0]*sym.pi*x) * sym.cos(i[1]*sym.pi*y)
         for i in indices]

# Eigenvalues of the covariance operator
eig_v = [1/(precision.subs(f, e).doit()/e).simplify()**alpha
         for e in eig_f]

# Functions
functions = [f*np.sqrt(float(v)) for f, v in zip(eig_f, eig_v)]

# Assembling diffusivity
utruth = np.loadtxt(data_dir + "/utruth.txt")
uapprox = np.mean(np.loadtxt(files[-1]), axis=1)

from matplotlib import cm
from matplotlib.ticker import LinearLocator

# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
perm_truth = sum([ui*fi for ui, fi in zip(utruth, functions)], 0)
variables = list(perm_truth.free_symbols)
variables.sort(key=lambda x: str(x))
perm_truth = sym.lambdify(variables, perm_truth)
perm_approx = sum([ui*fi for ui, fi in zip(uapprox, functions)], 0)
perm_approx = sym.lambdify(variables, perm_approx)
grid = np.linspace(0, 1, 200)
X, Y = np.meshgrid(grid, grid)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
contourf0 = ax[0].contourf(X, Y, perm_truth(X, Y))
contourf1 = ax[1].contourf(X, Y, perm_approx(X, Y))
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf0 = ax.plot_surface(X, Y, np.exp(perm_truth(X, Y)), linewidth=0, antialiased=False, cmap=cm.viridis)
# plt.savefig("/home/urbain/true.png", bbox_inches="tight", transparent=True)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf1 = ax.plot_surface(X, Y, np.exp(perm_approx(X, Y)), linewidth=0, antialiased=False, cmap=cm.viridis)
# plt.savefig("/home/urbain/reconstructed.png", bbox_inches="tight", transparent=True)
contourf1.set_array(contourf0.get_array)
fig.colorbar(contourf1, ax=ax, fraction=0.022, pad=.03)
ax[0].set_xlabel('$x_0$')
ax[1].set_xlabel('$x_0$')
ax[0].set_ylabel('$x_1$')
fig.savefig('log_permeability.pdf')
plt.show()
