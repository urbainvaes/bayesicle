import re
import glob
import numpy as np
import sympy as sym
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
N = 4
d = N**2
indices = [(m, n) for m in range(0, N) for n in range(0, N)]
indices.sort(key=lambda i: (max(i), sum(i), i[0]))
utruth = np.loadtxt(data_dir + "/utruth.txt")

def update(i):
    print(i)
    f = files[i]
    iteration = re.search(r"ensemble-([0-9]*).txt", f).group(1)
    ensembles = np.loadtxt(f).T

    ax.clear()
    ax.set_xticks(np.arange(len(indices)))
    ax.set_xticklabels((str(i) for i in indices))
    ax.set_title(f"Iteration {i}")
    for i, u_i in enumerate(ensembles):
        ax.plot(range(d), u_i, '.', ms=10)
    # ax.plot(range(d), np.mean(ensembles, axis=0), 'bx', ms=20, mew=5)
    ax.plot(range(d), utruth, 'kx', ms=20, mew=5)
    for i in range(d):
        ens = ensembles[:, i]
        mean_dir = np.mean(ens)
        std_dir = np.std(ens)
        y_plot = np.linspace(mean_dir - 4*std_dir, mean_dir + 4*std_dir)
        x_plot = (1/2) * np.exp(-(y_plot - mean_dir)**2/(2*std_dir**2))
        ax.plot(i + x_plot, y_plot, c='gray')
    # plt.draw()
    # plt.pause(.1)

update(len(files) - 1)
plt.savefig("posterior_cbs.pdf")


# animate = animation.FuncAnimation
# anim = animate(fig, update, len(files), repeat=False)
# writer = animation.writers['ffmpeg'](fps=2, bitrate=500, codec='libvpx-vp9')
# anim.save(f'elliptic.webm', writer=writer, dpi=500)

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

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
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
contourf1.set_array(contourf0.get_array)
fig.colorbar(contourf1, ax=ax, fraction=0.022, pad=.03)
ax[0].set_xlabel('$x_0$')
ax[1].set_xlabel('$x_0$')
ax[0].set_ylabel('$x_1$')
fig.savefig('log_permeability.pdf')
plt.show()
