import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import lib_inverse_problem
import lib_opti_problem
import math

# cmap = 'spring'
# cmap = 'gray_r'
cmap = 'jet'
matplotlib.rc('image', cmap=cmap)


def set_text(iteration, data, text):
    the_text = "Iteration {}".format(iteration)
    if data['solver'] == 'cbs':
        the_text += "\n $\\beta = {:.3f}$\n $J_{{\\rm eff}}/J = {:.4f}$"\
                    .format(data['beta'], data['ess'], len(data['weights']))
    text.set_text(the_text)


class AllCoeffsPlotter():
    def __init__(self, ip, **config):
        if isinstance(ip, lib_inverse_problem.InverseProblem):
            self.u = ip.unknown
        elif isinstance(ip, lib_opti_problem.OptimizationProblem):
            self.u = ip.argmin
        self.show_weights = config.get('show_weights', False)
        self.show_text = config.get('show_text', True)
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter([], [], cmap=cmap)
        self.cb = None

    def set_text(self, iteration, data):
        if not self.show_text:
            return
        props = dict(boxstyle='round', facecolor='cyan', alpha=0.5)
        text = self.ax.text(
            .02, .98, "Ready", fontsize=18, bbox=props,
            horizontalalignment='left', verticalalignment='top',
            transform=self.ax.transAxes)
        set_text(iteration, data, text)

    def plot(self, iteration, data):
        plot_args = {}
        self.ax.clear()
        ensembles = data['ensembles']
        if len(ensembles) == 0:
            return
        x_plot = range(len(ensembles[0]))
        if data['solver'] == 'cbs' and self.show_weights:
            if self.cb is None:
                self.cb = self.fig.colorbar(self.scatter, ax=self.ax)
            max_weights = np.max(data['weights'])
            self.scatter.set_clim((0, max_weights))
        for i, u_i in enumerate(ensembles):
            if data['solver'] == 'cbs' and self.show_weights:
                my_cmap = plt.cm.get_cmap(cmap)
                plot_args['c'] = my_cmap(data['weights'][i]/max_weights)
            self.ax.plot(x_plot, u_i, '.', ms=10, **plot_args)
        self.ax.plot(range(len(self.u)), np.mean(ensembles, axis=0), 'bx', ms=20, mew=5)
        for i in range(len(ensembles[0])):
            ens = ensembles[:, i]
            mean_dir = np.mean(ens)
            std_dir = np.std(ens)
            y_plot = np.linspace(mean_dir - 4*std_dir, mean_dir + 4*std_dir)
            if data['solver'] == 'cbs':
                x_plot = (1/2) * np.exp(-(y_plot - mean_dir)**2/(2*std_dir**2))
            else:
                # kernel = scipy.stats.gaussian_kde(ens, bw_method=.1)
                kernel = scipy.stats.gaussian_kde(ens)
                x_plot = kernel([y_plot]).T
                x_plot = (1/2) * x_plot / np.max(x_plot)
            self.ax.plot(i + x_plot, y_plot, c='gray')
        if self.u is not None:
            self.ax.plot(range(len(self.u)), self.u, 'kx', ms=20, mew=5)
        self.set_text(iteration, data)


class MainModesPlotter:

    def __init__(self, ip, **config):
        if isinstance(ip, lib_inverse_problem.InverseProblem):
            self.u = ip.unknown
        elif isinstance(ip, lib_opti_problem.OptimizationProblem):
            self.u = ip.argmin
        self.coeffs = config.get('coeffs', [0, 1, 2])
        self.show_weights = config.get('show_weights', True)
        c0, c1, c2 = self.coeffs
        self.fig, self.ax = plt.subplots(1, 2)
        self.ax[0].plot(self.u[c0], self.u[c1], 'kx', ms=20, mew=5)
        self.ax[1].plot(self.u[c1], self.u[c2], 'kx', ms=20, mew=5)
        self.ax[0].set_xlabel(r'$u_{}$'.format(c0))
        self.ax[0].set_ylabel(r'$u_{}$'.format(c1))
        self.ax[1].set_xlabel(r'$u_{}$'.format(c1))
        self.ax[1].set_ylabel(r'$u_{}$'.format(c2))
        scatter_1 = self.ax[0].scatter([], [], s=15, cmap=cmap)
        scatter_2 = self.ax[1].scatter([], [], s=15, cmap=cmap)
        self.scatters = [scatter_1, scatter_2]
        self.cb = None

        ax_text = self.ax[0]
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.text = ax_text.text(
            .03, .98, "Ready", fontsize=18, bbox=props,
            horizontalalignment='left', verticalalignment='top',
            transform=ax_text.transAxes)
        self.config = config

    def plot(self, iteration, data):

        ensembles = data['ensembles']
        if data['solver'] == 'md':
            cutoff = self.config.get('cutoff', 10**10)
            if cutoff < len(ensembles):
                ensembles = ensembles[-cutoff:]

        if len(data['ensembles']) == 0:
            return

        def update_scatter(an_ax, scatter, i1, i2):
            scatter.set_offsets(ensembles[:, [i1, i2]])
            x_plot, y_plot = ensembles[:, i1], ensembles[:, i2]
            xmin = min(self.u[i1], np.min(x_plot))
            xmax = max(self.u[i1], np.max(x_plot))
            ymin = min(self.u[i2], np.min(y_plot))
            ymax = max(self.u[i2], np.max(y_plot))
            Delta_x = max(xmax - xmin, 1)
            Delta_y = max(ymax - ymin, 1)
            an_ax.set_xlim(xmin - .1*Delta_x, xmax + .1*Delta_x)
            an_ax.set_ylim(ymin - .1*Delta_y, ymax + .1*Delta_y)
            if data['solver'] == 'cbs' and self.show_weights:
                scatter.set_array(data['weights'])
                scatter.set_clim((0, np.max(data['weights'])))
                if self.cb is None:
                    self.cb = self.fig.colorbar(self.scatters[1], ax=self.ax)

        coeff_0, coeff_1, coeff_2 = self.coeffs
        update_scatter(self.ax[0], self.scatters[0], coeff_0, coeff_1)
        update_scatter(self.ax[1], self.scatters[1], coeff_1, coeff_2)
        set_text(iteration, data, self.text)


class TwoDimPlotter:
    def __init__(self, ip, **config):
        self.fig, self.ax = plt.subplots()
        n_grid = 800
        self.argmin, _ = ip.map_estimator()
        # Width for plot of posterior
        self.Lx = config.get('Lx', 1.5)
        self.Ly = config.get('Ly', 1.5)
        x_plot = self.argmin[0] + self.Lx*np.linspace(-1, 1, n_grid)
        y_plot = self.argmin[1] + self.Ly*np.linspace(-1, 1, n_grid)
        # self.ax.plot(ip.argmin[0], ip.argmin[1], 'kx', ms=20, mew=5)
        self.ax.set_xlim(-4, -1)
        self.ax.set_ylim(102, 107)
        # self.ax.set_xlabel('$x$')
        # self.ax.set_ylabel('$y$')
        # self.mean = self.ax.plot([], [], 'x', color='violet', ms=20, mew=5)
        if not config.get('opti', False):
            X, Y = np.meshgrid(x_plot, y_plot)
            Z = (1/ip.normalization()) * ip.posterior(X, Y)
            cmap_posterior = plt.cm.get_cmap('binary')
            self.ax.contourf(X, Y, Z, levels=100, cmap=cmap_posterior)
        if config.get('contours', True):
            Lx_contour = config.get('Lx_contours', 1)
            Ly_contour = config.get('Ly_contours', 1)
            relative = config.get('relative', True)
            addx = self.argmin[0] if relative else 0
            addy = self.argmin[1] if relative else 0
            x_plot = addx + Lx_contour*np.linspace(-1, 1, n_grid)
            y_plot = addy + Ly_contour*np.linspace(-1, 1, n_grid)
            X, Y = np.meshgrid(x_plot, y_plot)
            Z = ip.least_squares_array(X, Y)
            if isinstance(ip, lib_opti_problem.OptimizationProblem):
                cont = self.ax.contourf(X, Y, Z, levels=100, cmap='rainbow')
                # self.ax.contour(X, Y, Z, levels=20, colors='black')
                # self.fig.colorbar(cont, orientation="horizontal")
                self.fig.colorbar(cont)
            else:
                cont = self.ax.contour(X, Y, -np.log(Z), levels=200, cmap='rainbow')
                self.fig.colorbar(cont)
            constraint = None
            if ip.eq_constraint is not None:
                constraint = ip.eq_constraint
            elif ip.ineq_constraint is not None:
                constraint = ip.ineq_constraint
            if constraint is not None:
                Z = constraint((X, Y))
                self.ax.contour(X, Y, Z, levels=[0])
        # self.scatter = self.ax.scatter([], [], cmap=cmap)
        if config.get('show_weights', True):
            kwargs = {'cmap': cmap}
        else:
            kwargs = {'c': 'black', 's': 30, 'edgecolors': 'black'}
        self.scatter = self.ax.scatter([], [], **kwargs)
        # cbar = self.fig.colorbar(self.scatter)
        # cbar.set_ticks([])
        self.my_plot = self.ax.plot([], [], 'k')
        self.config = config

        # Text on plot
        props = dict(boxstyle='round', facecolor='cyan', alpha=0.9)
        self.text = self.ax.text(
            .02, .98, "Ready", fontsize=18, bbox=props,
            horizontalalignment='left', verticalalignment='top',
            transform=self.ax.transAxes)

    def plot(self, iteration, data):
        title = "Iteration {}".format(iteration)
        title = ""
        ensembles = data['ensembles']
        if data['solver'] == 'md':
            cutoff = self.config.get('cutoff', 10**10)
            if cutoff < len(ensembles):
                ensembles = data['ensembles'][-cutoff:]
        if len(ensembles) == 0:
            return
        self.scatter.set_offsets(ensembles)
        x_plot, y_plot = ensembles[:, 0], ensembles[:, 1]
        relative = self.config.get('relative', True)
        xmin = min(self.argmin[0] - self.Lx, np.min(x_plot))
        xmax = max(self.argmin[0] + self.Lx, np.max(x_plot))
        ymin = min(self.argmin[1] - self.Ly, np.min(y_plot))
        ymax = max(self.argmin[1] + self.Ly, np.max(y_plot))
        delta_x, delta_y = xmax - xmin, ymax - ymin
        if self.config.get('adapt_size', False):
            min_step_x = 1
            min_step_y = 1
            # self.ax.set_xlim(math.floor(xmin - .1*delta_x), math.ceil(xmax + .1*delta_x))
            # self.ax.set_ylim(math.floor(ymin - .1*delta_y), math.ceil(ymax + .1*delta_y))
            self.ax.set_xlim(-4, -1)
            self.ax.set_ylim(102, 107)
        if data['solver'] == 'cbs':
            # xmean = np.sum(data['weights']*x_plot)
            # ymean = np.sum(data['weights']*y_plot)
            xmean = np.mean(x_plot)
            ymean = np.mean(y_plot)
            # self.mean[0].set_data([xmean], [ymean])
        if data['solver'] == 'cbs' and self.config.get('show_weights', True):
            # title += r": $\beta = {:.3f}$, ESS = {:.2f}"\
            #          .format(data['beta'], data['ess'])
            # self.scatter.set_array(data['weights'])
            sizes = 5 + 40*data['weights']/np.max(data['weights'])
            self.scatter.set_sizes(sizes)
            # self.scatter.set_clim((0, np.max(data['weights'])))
        if data['solver'] == 'md' and self.config.get('show_weights', True):
            n = len(ensembles)
            self.scatter.set_array(np.arange(n)/n)
            self.scatter.set_clim((0, 1))
            self.my_plot[0].set_data(ensembles[:, 0], ensembles[:, 1])
        self.ax.set_title(title)
        set_text(iteration, data, self.text)


class OneDimPlotter(object):

    def __init__(self, ip, **config):
        self.fig, self.ax = plt.subplots()
        n_grid = 400
        x_plot = 6*np.linspace(-1, 1, n_grid)
        if isinstance(ip, lib_inverse_problem.InverseProblem):
            posterior = (1/ip.normalization()) * ip.posterior(x_plot)
            self.ax.plot(x_plot, posterior, label="Posterior")
            self.ax.plot(x_plot, -np.log(posterior), label=r"$\Phi_R$")
        elif isinstance(ip, lib_opti_problem.OptimizationProblem):
            self.ax.plot(x_plot, ip.objective_array(x_plot), label=r"Objective function")
        self.my_plot = self.ax.plot(x_plot, 0*x_plot, ".-", label="CBS")[0]
        self.x_plot = x_plot
        self.ax.set_ylim(0, 2)
        plt.legend()

    def plot(self, iteration, data):
        ens = data['ensembles']
        x_particles = ens.T[0]
        x_particles = np.sort(x_particles)
        mean, cov = np.mean(x_particles), np.cov(x_particles)
        # x_plot = self.x_plot
        x_plot = x_particles
        self.my_plot.set_data(x_plot, 1/np.sqrt(2*np.pi*cov) *
                              np.exp(-(x_plot - mean)**2/(2*cov)))

    def kernel_plot(self, iteration, data):
        ens, J = data['ensembles'], len(data['ensembles'])
        kernel = scipy.stats.gaussian_kde(ens.reshape(J))
        y_plot = kernel([self.x_plot]).T
        self.my_plot.set_data(self.x_plot, y_plot)
