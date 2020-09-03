import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

cmap = 'spring'
matplotlib.rc('image', cmap=cmap)


def set_text(iteration, data, text):
    the_text = "Iteration {}".format(iteration)
    if data['solver'] == 'cbs':
        the_text += "\n $\\beta = {:.3f}$\n ESS = {:.2f}/{}"\
                    .format(data['beta'], data['ess'], len(data['weights']))
    text.set_text(the_text)


class AllCoeffsPlotter():
    def __init__(self, ip, **config):
        self.u = ip.unknown
        self.show_weights = config.get('show_weights', False)
        self.fig, self.ax = plt.subplots()
        self.scatter = self.ax.scatter([], [], cmap=cmap)
        self.cb = None

    def set_text(self, iteration, data):
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text = self.ax.text(
            .03, .98, "Ready", fontsize=18, bbox=props,
            horizontalalignment='left', verticalalignment='top',
            transform=self.ax.transAxes)
        set_text(iteration, data, text)

    def plot(self, iteration, data):
        plot_args = {}
        self.ax.clear()
        ensembles = data['ensembles']
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
            self.ax.plot(x_plot, u_i, '.', ms=15, **plot_args)
        for i in range(len(ensembles[0])):
            mean_dir = np.mean(ensembles[:, i])
            std_dir = np.std(ensembles[:, i])
            print(std_dir)
            y_plot = np.linspace(mean_dir - 3*std_dir, mean_dir + 3*std_dir)
            x_plot = i + .5*np.exp(-(y_plot - mean_dir)**2/(2*std_dir**2))
            self.ax.plot(x_plot, y_plot, c='gray')
        if self.u is not None:
            self.ax.plot(range(len(self.u)), self.u, 'kx', ms=20, mew=5)
        self.set_text(iteration, data)


class MainModesPlotter:

    def __init__(self, ip, **config):
        self.u = ip.unknown
        self.coeffs = config.get('coeffs', [0, 1, 2])
        self.show_weights = config.get('show_weights', True)
        c0, c1, c2 = self.coeffs
        self.fig, self.ax = plt.subplots(1, 2)
        self.ax[0].plot(self.u[c0], self.u[c1], 'kx', ms=20, mew=5)
        self.ax[1].plot(self.u[c0], self.u[c2], 'kx', ms=20, mew=5)
        self.ax[0].set_xlabel(r'$u_{}$'.format(c0))
        self.ax[0].set_ylabel(r'$u_{}$'.format(c1))
        self.ax[1].set_xlabel(r'$u_{}$'.format(c0))
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

    def plot(self, iteration, data):

        def update_scatter(an_ax, scatter, i1, i2):
            scatter.set_offsets(data['ensembles'][:, [i1, i2]])
            x_plot, y_plot = data['ensembles'][:, i1], data['ensembles'][:, i2]
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
        update_scatter(self.ax[1], self.scatters[1], coeff_0, coeff_2)
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
        X, Y = np.meshgrid(x_plot, y_plot)
        Z = (1/ip.normalization()) * ip.posterior(X, Y)
        cmap_posterior = plt.cm.get_cmap('binary')
        self.ax.contourf(X, Y, Z, levels=100, cmap=cmap_posterior)
        if config.get('contours', True):
            Lx_contour = config.get('Lx_contours', 1)
            Ly_contour = config.get('Ly_contours', 1)
            x_plot = self.argmin[0] + Lx_contour*np.linspace(-1, 1, n_grid)
            y_plot = self.argmin[1] + Ly_contour*np.linspace(-1, 1, n_grid)
            X, Y = np.meshgrid(x_plot, y_plot)
            Z = (1/ip.normalization()) * ip.posterior(X, Y)
            self.ax.contour(X, Y, -np.log(Z), levels=50, cmap='jet')
            constraint = None
            if ip.eq_constraint is not None:
                constraint = ip.eq_constraint
            elif ip.ineq_constraint is not None:
                constraint = ip.ineq_constraint
            if constraint is not None:
                Z = constraint((X, Y))
                self.ax.contour(X, Y, Z, levels=[0])
        self.scatter = self.ax.scatter([], [], cmap=cmap)
        self.config = config

        # Text on plot
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.text = self.ax.text(
            .03, .98, "Ready", fontsize=18, bbox=props,
            horizontalalignment='left', verticalalignment='top',
            transform=self.ax.transAxes)

    def plot(self, iteration, data):
        title = "Iteration {}".format(iteration)
        ensembles = data['ensembles']
        if data['solver'] == 'md':
            cutoff = self.config.get('cutoff', 200)
            if cutoff < len(ensembles):
                ensembles = data['ensembles'][-cutoff:]
        self.scatter.set_offsets(ensembles)
        x_plot, y_plot = ensembles[:, 0], ensembles[:, 1]
        xmin = min(self.argmin[0] - self.Lx, np.min(x_plot))
        xmax = max(self.argmin[0] + self.Lx, np.max(x_plot))
        ymin = min(self.argmin[1] - self.Ly, np.min(y_plot))
        ymax = max(self.argmin[1] + self.Ly, np.max(y_plot))
        delta_x, delta_y = xmax - xmin, ymax - ymin
        self.ax.set_xlim(xmin - .1*delta_x, xmax + .1*delta_x)
        self.ax.set_ylim(ymin - .1*delta_y, ymax + .1*delta_y)
        print(ymin - .1*delta_y, ymax + .1*delta_y)
        if data['solver'] == 'cbs' and self.config.get('show_weights', True):
            title += r": $\beta = {:.3f}$, ESS = {:.2f}"\
                     .format(data['beta'], data['ess'])
            self.scatter.set_array(data['weights'])
            sizes = 1 + 40*data['weights']/np.max(data['weights'])
            self.scatter.set_sizes(sizes)
            self.scatter.set_clim((0, np.max(data['weights'])))
        if data['solver'] == 'md' and self.config.get('show_weights', True):
            n = len(ensembles)
            self.scatter.set_array(np.arange(n)/n)
            self.scatter.set_clim((0, 1))
        self.ax.set_title(title)
        set_text(iteration, data, self.text)


class OneDimPlotter(object):

    def __init__(self, ip):
        self.fig, self.ax = plt.subplots()
        n_grid = 400
        x_plot = 6*np.linspace(-1, 1, n_grid)
        posterior = (1/ip.normalization()) * ip.posterior(x_plot)
        self.ax.plot(x_plot, posterior, label="Posterior")
        # self.ax.plot(x_plot, -np.log(posterior), label=r"$\Phi_R$")
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
