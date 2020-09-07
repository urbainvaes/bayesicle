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
d, K, k = 3, 3, 5


# Forward model
def forward(u):
    # return np.array([u[0], 10*u[1], 100*u[2], u[3], 10*u[4], 100*u[5]])
    # return np.array([u[0], 10*u[1], 100*u[2]])
    return np.array([u[0], k*u[1], k**2*u[2]])


# Covariance of noise and prior
γ, σ = 1, 2
Γ = np.diag([γ**2]*K)
Σ = np.diag([σ**2]*d)

# Unknown
u = σ*np.random.randn(d)
u = np.array([1, 1, 1])

# Observation
y = forward(u)

# y = np.array([1, k, k**2])
# np.random.seed(0)
# y = y + sp.linalg.sqrtm(Γ).dot(np.random.randn(K))

# Inverse problem
ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y, unknown=u)

AllCoeffsPlotter = lib_plotters.AllCoeffsPlotter
MainModesPlotter = lib_plotters.MainModesPlotter
