import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_bimodal_2d as m
import scipy.stats

matplotlib.rc('font', size=20)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('savefig', bbox='tight')
# matplotlib.rc('figure.subplot', hspace=.0)
# matplotlib.rc('figure.subplot', wspace=.03)
# matplotlib.rc('figure', figsize=(15, 5))

iter = 10000
data = np.loadtxt(f"data_julia/model_bimodal_2d/cbs_metro/ensembles-{iter}.txt")

n_grid = 100
x_plot = -4 + 8*np.linspace(0, 1, n_grid)
y_plot = -4 + 8*np.linspace(0, 1, n_grid)
X, Y = np.meshgrid(x_plot, y_plot)
Z = m.ip.least_squares_array(X, Y)

fig, ax = plt.subplots(1, 1)

# kernel = scipy.stats.gaussian_kde(data, bw_method=.1)
kernel = scipy.stats.gaussian_kde(data, bw_method=.17)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kernel(positions).T, X.shape)
cnt2 = ax.contourf(X, Y, Z, levels=60, cmap='viridis')
ax.set_xticks([])
ax.set_yticks([])
for c in cnt2.collections:
    c.set_edgecolor("face")
# ax.axis("equal")
fig.savefig(f"figures/posterior_bimodal_2d-{iter}.pdf")

fig, ax = plt.subplots(1, 1)
Z_post = m.ip.posterior(X, Y)
# ax2.plot(data[0], data[1], '.', ms=1)
cnt3 = ax.contourf(X, Y, Z_post, levels=60, cmap='viridis')
ax.set_xticks([])
ax.set_yticks([])
for c in cnt3.collections:
    c.set_edgecolor("face")
# ax.axis("equal")
fig.savefig(f"figures/posterior_bimodal_2d-exact.pdf")

plt.show()
