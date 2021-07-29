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

data_aldi = np.loadtxt("data_julia/model_bimodal_2d/aldi/all_ensembles.txt")
data = np.loadtxt("data_julia/model_bimodal_2d/multiscale/all_ensembles.txt")

n_grid = 100
x_plot = -4 + 8*np.linspace(0, 1, n_grid)
y_plot = -4 + 8*np.linspace(0, 1, n_grid)
X, Y = np.meshgrid(x_plot, y_plot)
Z = m.ip.least_squares_array(X, Y)

fig, [ax1, ax2, ax3] = plt.subplots(1, 3)

# kernel = scipy.stats.gaussian_kde(data, bw_method=.1)
kernel = scipy.stats.gaussian_kde(data, bw_method=.04)
kernel_aldi = scipy.stats.gaussian_kde(data_aldi, bw_method=.07)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kernel(positions).T, X.shape)
Z_aldi = np.reshape(kernel_aldi(positions).T, X.shape)
cnt1 = ax1.contourf(X, Y, Z_aldi, levels=60, cmap='viridis')
cnt2 = ax2.contourf(X, Y, Z, levels=60, cmap='viridis')
ax2.set_xticks([])
ax2.set_yticks([])

Z_post = m.ip.posterior(X, Y)
# ax3.plot(data[0], data[1], '.', ms=1)
cnt3 = ax3.contourf(X, Y, Z_post, levels=60, cmap='viridis')
ax3.set_xticks([])
ax3.set_yticks([])

for c in cnt1.collections:
    c.set_edgecolor("face")
for c in cnt2.collections:
    c.set_edgecolor("face")
for c in cnt3.collections:
    c.set_edgecolor("face")

fig.savefig('figures/posterior_bimodal_2d.pdf')
plt.show()
