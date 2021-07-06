import numpy as np
import scipy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Forward model and solver
import model_elliptic_1d as m
import solvers

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.0)
matplotlib.rc('figure.subplot', wspace=.03)

# Solvers
α = .8
solver_cbs = solvers.CbsSolver(
    dt=-np.log(α),
    # frac_min=50/100,
    # frac_max=55/100,
    parallel=True,
    beta=.5,
    adaptive=False,
    dirname=m.__name__,
    opti=False)

if __name__ == "__main__":

    # Plots
    plotter = m.Plotter(m.ip, show_weights=True, adapt_size=True,
                        contours=True, Lx=1, Ly=1, Lx_contours=5, Ly_contours=40)

    # Number of particles
    J = 1000

    # ensembles_x = np.random.randn(J)
    # ensembles_x = np.random.randn(J)
    # ensembles_y = 90 + 20*np.random.rand(J)

    ensembles_x = -4 + 3*np.random.rand(J)
    ensembles_y = 102 + 5*np.random.rand(J)
    ensembles = np.vstack((ensembles_x, ensembles_y)).T

    def update(i):
        global ensembles
        print("Iteration {:04d}".format(i))
        data = solver_cbs.step(m.ip, ensembles,
                               filename="iteration-{:04d}.npy".format(i))
        ensembles = data.new_ensembles
        plotter.plot(i, data._asdict())
    anim = animation.FuncAnimation(plotter.fig, update, 51, init_func=lambda: None, repeat=False)
    # plt.show()
    writer = animation.writers['ffmpeg'](fps=2, bitrate=500, codec='libvpx-vp9')
    anim.save(f'{m.__name__}_α={α}.webm', writer=writer, dpi=500)
