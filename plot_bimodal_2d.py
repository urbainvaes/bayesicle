import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_bimodal_2d as m
import scipy.stats

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('figure', figsize=(9, 7))
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure.subplot', hspace=.0)
matplotlib.rc('figure.subplot', wspace=.03)
matplotlib.rc('figure', figsize=(15, 5))

# data = np.loadtxt("data_julia/model_bimodal_2d/aldi/all_ensembles.txt")
data = np.loadtxt("data_julia/model_bimodal_2d/multiscale/all_ensembles.txt")

n_grid = 100
x_plot = -4 + 8*np.linspace(0, 1, n_grid)
y_plot = -4 + 8*np.linspace(0, 1, n_grid)
X, Y = np.meshgrid(x_plot, y_plot)
Z = m.ip.least_squares_array(X, Y)

fig, [ax2, ax3] = plt.subplots(1, 2)
# ax1.contour(X, Y, Z, levels=100, cmap='viridis')
# ax2.contour(X, Y, Z, levels=100, cmap='viridis')
# ax3.contour(X, Y, Z, levels=100, cmap='viridis')

# ax1.plot(data[0], data[1], '.', ms=1)

kernel = scipy.stats.gaussian_kde(data)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kernel(positions).T, X.shape)
ax2.contourf(X, Y, Z, levels=60, cmap='binary')
ax2.set_yticks([])

Z_post = m.ip.posterior(X, Y)
ax3.contourf(X, Y, Z_post, levels=60, cmap='binary')
ax3.set_xticks([])
ax3.set_yticks([])

fig.savefig('figures/posterior_bimodal_2d.pdf')

plt.show()
