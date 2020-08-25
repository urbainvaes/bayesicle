import numpy as np
import lib_inverse_problem
import lib_plotters

settings = {
        'method': 'eks'
        }

# Dimensions of the model
d, K = 2, 2

# Covariance of noise and prior
γ, σ = .1, 10
Γ = np.diag([γ**2]*K)
Σ = np.diag([σ**2]*d)


# Forward model
def forward(θ):

    def solution(x):
        return θ[1]*x + np.exp(-θ[0])*(-x**2/2 + x/2)

    x1, x2 = .25, .75
    return np.array([solution(x1), solution(x2)])


# Observation
u = None
y = np.array([27.5, 79.7])


# Number of particles
J = 100

# Initialization of ensembles
ensembles_x = 3*np.random.randn(J)
ensembles_y = 90 + 20*np.random.rand(J)
# ensembles_x = np.random.randn(J)
# ensembles_y = 90 + 10*np.random.rand(J)

ensembles = np.vstack((ensembles_x, ensembles_y)).T

# Width for plot of posterior
Lx = 1.5
Ly = 1.5

# Inverse problem
ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y)
Plotter = lib_plotters.TwoDimPlotter
