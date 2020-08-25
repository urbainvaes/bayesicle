import numpy as np

import lib_plotters
import lib_inverse_problem


np.random.seed(0)
plot_settings = {
        # 'plot_type': 'main_modes',
        'plot_type': 'all_coeffs',
        'method': 'cbs'
        }

# Dimensions of the model
# d, K = 6, 6
d, K = 3, 3


# Forward model
def forward(u):
    # return np.array([u[0], 10*u[1], 100*u[2], u[3], 10*u[4], 100*u[5]])
    return np.array([u[0], 10*u[1], 100*u[2]])


# Covariance of noise and prior
γ, σ = .1, 2
Γ = np.diag([γ**2]*K)
Σ = np.diag([σ**2]*d)

# Unknown
# u = np.array([2, 2, 2])
u = σ*np.random.randn(d)

# Observation
np.random.seed(0)
y = forward(u)
# y = y + sp.linalg.sqrtm(Γ).dot(np.random.randn(K))

# Number of particles
J = 10
# J = 20

# Initial ensembles
ensembles = 5*σ*np.random.rand(J, d)

# Inverse problem
ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y, unknown=u)

AllCoeffsPlotter = lib_plotters.AllCoeffsPlotter
MainModesPlotter = lib_plotters.MainModesPlotter
