import numpy as np
import lib_inverse_problem
import lib_plotters
import sympy as sym

a, b = 2.5, 2.5

def ackley_2d(x, y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(np.power(x-a, 2)+np.power(y-b, 2)))) \
           - np.exp(0.5*(np.cos(2*np.pi*(x-a))+np.cos(2*np.pi*(y-b)))) + 20 + np.e

def rastrigin_function_2d(x, y):
    n, x1, x2, A = 2, 2, 4, 10
    return A*n + (x1**2 - A*np.cos(2*np.pi*x1)) + (x2**2 - A*np.cos(2*np.pi*x2))

def forward(u):
    fun = rastrigin_function_2d
    # fun = ackley_2d
    return np.array([rastrigin_function_2d(*u)])


# Dimensions of the model
d, K = 2, 1

# Data = lower bound
y = np.array([-.1])

# Covariance of noise and prior
# γ, σ = 1, .5
γ, σ = 1, 1000000
Γ = np.diag([γ**2]*K)
Σ = np.diag([σ**2]*d)

# Constraint
vx, vy = sym.symbols('x y', real=True)
constraint = vx**2 + vy**2 - (a**2 + b**2)
# constraint = vx**2 + vy**2 + (vx + vy)**2/(.5 + (vx - vy)**2) - 2**2
# constraint = sym.cos(vx) - vy
grad_constraint = [constraint.diff(vx), constraint.diff(vy)]
constraint = sym.lambdify((vx, vy), constraint)
grad_constraint = sym.lambdify((vx, vy), grad_constraint)

constraints = {
        'eq_constraint': lambda x: constraint(*x),
        'eq_constraint_grad': lambda x: np.array(grad_constraint(*x)),
        }

# constraints = {
#         'ineq_constraint': lambda x: constraint(*x),
#         'ineq_constraint_grad': lambda x: np.array(grad_constraint(*x)),
#         }

ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y, **constraints)
Plotter = lib_plotters.TwoDimPlotter
