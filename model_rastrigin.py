import numpy as np
import lib_inverse_problem
import lib_opti_problem
import lib_plotters
import sympy as sym

n = 2
shift = 2
rootpow = 1

def rastrigin_nd(x):
    assert len(x) == n
    A = 10
    result = A*n
    for i in range(n):
        z = rootpow**i * (x[i] - shift)
        result += z**2 - A*np.cos(2*np.pi*z)
    return result


def objective(u):
   return np.array(rastrigin_nd(u))


argmin = np.ones(n) * shift
op = lib_opti_problem.OptimizationProblem(n, objective, argmin=argmin, fmin=0)

if n == 1:
    Plotter = lib_plotters.OneDimPlotter

elif n == 2:
    Plotter = lib_plotters.TwoDimPlotter

else:
    Plotter = lib_plotters.AllCoeffsPlotter
    # Plotter = lib_plotters.MainModesPlotter
