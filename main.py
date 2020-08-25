import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

# Forward model
import model_simple_1d
import model_elliptic_1d
import model_elliptic_2d
import model_simple_3d

import solver_cbs
import solver_eks

m = model_elliptic_2d
# m = model_simple_3d
# m = model_elliptic_1d
solver = solver_cbs
# solver = solver_eks

# Plotting options
matplotlib.rc('animation', bitrate=-1)
matplotlib.rc('animation', codec='libvpx-vp9')
matplotlib.rc('animation', writer='ffmpeg')
matplotlib.rc('font', family='serif')
matplotlib.rc('font', size=17)
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('lines', markersize=12)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('figure.subplot', hspace=.01)
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('text', usetex=True)

# Set seed to zero
np.random.seed(0)

# Dimensions of the model
d, K = m.d, m.K

# Covariance of noise and prior
Γ, Σ = m.Γ, m.Σ

# Compute inverses
inv_Γ = np.linalg.inv(m.Γ)
inv_Σ = np.linalg.inv(m.Σ)

# Unknown and observation
u, y = m.u, m.y

# Number of ensembles and initial ensembles
J, ensembles = m.J, m.ensembles


# Forward model (For some reason — not sure exactly why — this is neccesary for
# multiprocessing to work.
def forward(u):
    return m.forward(u)


# Least squares functional
def f(u):
    diff = m.forward(u) - y
    return (1/2)*diff.dot(inv_Γ.dot(diff)) + (1/2)*u.dot(inv_Σ.dot(u))


# Initialize plot
m_plot = m.Plotter(u, **m.plot_settings)


def plot(i):
    global ensembles

    n_iter = 1
    print("Iteration {}".format(i))
    for n in range(n_iter):

        if solver.__name__ == "solver_cbs":
            dt = .1
            dt = np.inf
            ensembles_new, weights, β = solver.step(f, ensembles, dt)

        elif solver.__name__ == "solver_eks":
            dt = 1
            solver_eks.settings['noise'] = False
            solver_eks.settings['reg'] = False
            ensembles_new = solver.step(forward, ensembles, y, inv_Γ, inv_Σ)
            weights, β = None, None

    # Save to file
    data_dir = "data/{}/{}".format(m.__name__, solver.__name__)
    os.makedirs(data_dir, exist_ok=True)
    filename = "{}/iteration-{:04d}.npy".format(data_dir, i)
    data = {'ensembles': np.array(ensembles), 'weights': weights, 'beta': β}
    np.save(filename, data)

    # Plot iterates
    m_plot.plot(i*n_iter, ensembles, weights=weights, beta=β)
    ensembles = ensembles_new

    plt.draw()
    plt.pause(.1)


n_figs = 500
anim = animation.FuncAnimation(m_plot.fig, plot, n_figs,
                               interval=600, init_func=lambda: None)
# anim.save('{}.webm'.format(m.__name__))
plt.show()
