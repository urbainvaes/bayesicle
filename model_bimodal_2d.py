import numpy as np
import lib_inverse_problem
import lib_plotters

# Dimensions of the model
d, K = 2, 1

# Forward model
def forward(u):
    return np.array([(u[1] - u[0])**2])

# Covariance of noise and prior
γ, σ = 1, 1e20
noise_cov = np.diag([1]*K)
prior_cov = np.diag([1]*d)

# Observation
y = np.array([4.2297])

ip = lib_inverse_problem.InverseProblem(forward, noise_cov, prior_cov, y)
Plotter = lib_plotters.TwoDimPlotter
