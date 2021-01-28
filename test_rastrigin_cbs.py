import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import solvers
import model_rastrigin as m
# import model_ackley as m

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.0)
matplotlib.rc('figure.subplot', wspace=.03)

# Set seed to zero
np.random.seed(5)

# Number of particles
J = 100
beta = .05

# Main loop
n_iter = 701

# Plots
L = 3
plotter = m.Plotter(m.op, show_weights=False, cutoff=500, opti=True, relative=False,
                    contours=True, Lx=.01, Ly=.01, Lx_contours=3, Ly_contours=3)

if __name__ == "__main__":

    # Solvers
    α = .9
    α = .000000001
    dt = - np.log(α)
    solver_cbs = solvers.CbsSolver(
        dt=dt,
        # dt=.2,
        # frac_min=45/100,
        # frac_max=50/100,
        # dt=.02,
        parallel=True,
        # beta=beta,
        dirname=m.__name__ + f"/test-simulation",
        # adaptive=False,
        adaptive=True,
        # dirname=m.__name__ + f"/simulation-{isimul}",
        opti=True)
    solver, plot_step = solver_cbs, 5

    vec = L*np.ones(m.n)
    # ensembles = -vec + 2*vec*np.random.rand(J, m.n)
    ensembles = 5*np.random.rand(J, m.n)

    def update(i):
        global ensembles
        print("Iteration {:04d}".format(i))
        for j in range(plot_step):
            data = solver.step(m.op, ensembles,
                   filename="iteration-{:04d}.npy".format(i))
            ensembles = data.new_ensembles
        plotter.plot(plot_step*i, data._asdict())
        if i % plot_step == 0:
            plt.pause(1)
            plt.draw()

    anim = animation.FuncAnimation(plotter.fig, update, 120, init_func=lambda: None, repeat=False)
    plt.show()
    # writer = animation.writers['ffmpeg'](fps=4, bitrate=500, codec='libvpx-vp9')
    # anim.save(f'{m.__name__}_{m.n}d_α={α}.webm', writer=writer, dpi=500)
