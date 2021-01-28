import numpy as np
import lib_inverse_problem
import lib_opti_problem
import lib_plotters
import sympy as sym

n = 2
shift = 2
a, b = shift, shift

def ackley_2d(x, y):
    return -20*np.exp(-0.2*np.sqrt(0.5*(np.power(x-a, 2)+np.power(y-b, 2)))) \
           - np.exp(0.5*(np.cos(2*np.pi*(x-a))+np.cos(2*np.pi*(y-b)))) + 20 + np.e


def rastrigin_2d(x, y):
    n, A = 2, 10
    return A*n + (x**2 - A*np.cos(2*np.pi*x)) + (y**2 - A*np.cos(2*np.pi*y))


def rosenbrock_2d(x, y):
    return 100*(y-x**2)**2 + (1-x)**2


# fun, argmin, fmin = rosenbrock_2d, np.array([1, 1]), 0
# fun, argmin, fmin = rastrigin_2d, np.array([0, 0]), 0
fun, argmin, fmin = ackley_2d, np.array([a, b]), 0


def forward(u):
    # Sqrt for inverse problem!
    return np.sqrt(np.array([fun(*u)]))


# Dimensions of the model
d, K = 2, 1

# Data = lower bound
y = np.array([0])

# Constraint
vx, vy = sym.symbols('x y', real=True)
constraint = vx**2 + vy**2 - ((.9*a)**2 + (.9*b)**2)
# constraint = vx**2 + vy**2 + (vx + vy)**2/(.5 + (vx - vy)**2) - 2**2
# constraint = sym.cos(vx) - vy
grad_constraint = [constraint.diff(vx), constraint.diff(vy)]
constraint = sym.lambdify((vx, vy), constraint)
grad_constraint = sym.lambdify((vx, vy), grad_constraint)

epsilon = .1
def forward_constrained(u):
    return forward(u) + 1/epsilon * np.array([constraint(*u)**2])

# constraints = {
#         'eq_constraint': lambda x: constraint(*x),
#         'eq_constraint_grad': lambda x: np.array(grad_constraint(*x)),
#         }

# constraints = {
#         'ineq_constraint': lambda x: constraint(*x),
#         'ineq_constraint_grad': lambda x: np.array(grad_constraint(*x)),
#         }

constraints = {}

# ip = lib_inverse_problem.InverseProblem(forward_constrained, Γ, Σ, y,
#                                         argmin=argmin, fmin=0, **constraints)
# ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y,
#                                         argmin=argmin, fmin=0, **constraints)
def objective(u):
   return np.array(ackley_2d(*u))

op = lib_opti_problem.OptimizationProblem(2, objective, argmin=argmin, fmin=0)
Plotter = lib_plotters.TwoDimPlotter
