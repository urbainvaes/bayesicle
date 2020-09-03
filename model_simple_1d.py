import lib_plotters
import lib_inverse_problem
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stats

# Dimensions of the model
d, K = 1, 2

# Covariance of noise and prior
γ, σ = 1, 100
Γ = np.diag([γ**2]*K)
Σ = np.diag([σ**2]*d)


# Forward model
def forward(u):
    # return np.array([u[0], .5*np.sin(5*np.pi*u[0])])
    return np.array([u[0]**2, .1*u[0]])
    # return np.array([u[0], 2*u[0]])


# Unknown
u = np.array([1])

# Observation
np.random.seed(0)
# y = forward(u) + sp.linalg.sqrtm(Γ).dot(np.random.randn(K))
# y = np.array([1.353, 0.08])
y = np.array([1, .1])

# Inverse problem
ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y, unknown=u)

Plotter = lib_plotters.OneDimPlotter
