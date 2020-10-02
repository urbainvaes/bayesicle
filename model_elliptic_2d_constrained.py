import numpy as np
import sympy as sym
import fenics as fen
import matplotlib
import matplotlib.pyplot as plt

import lib_inverse_problem
import lib_plotters

matplotlib.rc('figure', figsize=(14, 8))
matplotlib.rc('figure.subplot', hspace=.1)
matplotlib.rc('font', size=14)
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('text', usetex=True)

# Fix seed
np.random.seed(20)

# Definition of the forward model {{{1
# ====================================


class ForwardDarcy():

    def __init__(self, K):
        # Variables
        x, y = sym.symbols('x[0], x[1]', real=True, positive=True)
        f = sym.Function('f')(x, y)

        grid = np.linspace(0, 1, K + 2)[1:-1]
        x_obs, y_obs = np.meshgrid(grid, grid)
        self.x_obs, self.y_obs = x_obs.reshape(K*K), y_obs.reshape(K*K)

        # --- THIS PART USED TO NOT BE PARALELLIZABLE --- #
        # Create mesh and define function space
        n_mesh = 80
        self.mesh = fen.UnitSquareMesh(n_mesh, n_mesh)
        self.f_space = fen.FunctionSpace(self.mesh, 'P', 2)

        # Define boundary condition
        # u_boundary = fen.Expression('x[0] + x[1]', degree=2)
        u_boundary = fen.Expression('0', degree=2)
        self.bound_cond = fen.DirichletBC(
           self.f_space, u_boundary,
           lambda x, on_boundary: on_boundary)

        # Define variational problem
        self.trial_f = fen.TrialFunction(self.f_space)
        self.test_f = fen.TestFunction(self.f_space)
        rhs = fen.Constant(50)

        self.lin_func = rhs*self.test_f*fen.dx
        # ----------------------------------------------- #

    def __call__(self, u_test, return_sol=False, return_permeability=False):

        # Variables
        x, y = sym.symbols('x[0], x[1]', real=True, positive=True)

        # Diffusivities
        diff_1 = np.exp(u_test[0])
        diff_2 = np.exp(u_test[1])

        # Interfaces
        ix = u_test[2]
        iy = u_test[3]

        # Diffusivity
        diff = sym.Piecewise((sym.Piecewise((diff_1, y <= iy), (diff_2, True)), x <= ix), (diff_2, True))
        # if return_permeability:
            # return sym.lambdify((x, y), diff)

        ccode_coeff = sym.ccode(diff)
        diff = fen.Expression(ccode_coeff, degree=2)
        if return_permeability:
            return diff

        # Define bilinear form in variational problem
        a_bil = diff*fen.dot(fen.grad(self.trial_f),
                             fen.grad(self.test_f))*fen.dx

        # Compute solution
        sol = fen.Function(self.f_space)
        fen.solve(a_bil == self.lin_func, sol, self.bound_cond)

        evaluations = [sol(xi, yi) for xi, yi in zip(self.x_obs, self.y_obs)]
        return sol if return_sol else np.array(evaluations)


# Test code {{{1
# ==============
if __name__ == "__main__":
    G = ForwardDarcy(10)
    u_truth = [1, 2, .5, .6]
    solution = G(u_truth, return_sol=True)
    p = fen.plot(solution)
    plt.colorbar(p)
    plt.show()
    solution = G(u_truth, return_permeability=True)
    p = fen.plot(solution, mesh=G.mesh, extend='both')
    plt.colorbar(p)
    plt.show()

# Parameters of the inverse problem {{{1
# ======================================

# Covariance of noise and prior
γ, σ = .01, 4

# Initialize forward model
Kx = 10
G = ForwardDarcy(Kx)

# Truth
u_truth = [0, 1, .4, .6]

# Forward model
forward = G.__call__

# Observation without noise
y = forward(u_truth)
K = len(y)

# Approximation
d = len(u_truth)

# Covariance of noise and prior
Σ = np.diag([σ**2, σ**2, 10**6, 10**6])
Γ = np.diag([γ**2]*K)

# Square root of covariance matrix
rtΓ = np.diag([γ]*K)

# Noisy observation
y = y + rtΓ.dot(np.random.randn(K))

# Constraints
diff_1, diff_2, ix, iy = sym.symbols('dx dy ix iy', real=True)
constraint = (ix - .5)**2 + (iy - .5)**2 - .2**2
grad_constraint = [constraint.diff(v) for v in [diff_1, diff_2, ix, iy]]
constraint = sym.lambdify((diff_1, diff_2, ix, iy), constraint)
grad_constraint = sym.lambdify((diff_1, diff_2, ix, iy), grad_constraint)
constraints = {
        'eq_constraint': lambda x: constraint(*x),
        'eq_constraint_grad': lambda x: np.array(grad_constraint(*x)),
        }

# Inverse problem
ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y, unknown=u_truth, **constraints)

# Delete local variables to make them unaccessible outside
del γ, σ, Kx, u_truth, d, forward, Σ, Γ, rtΓ, y

# Plotters
AllCoeffsPlotter = lib_plotters.AllCoeffsPlotter
MainModesPlotter = lib_plotters.MainModesPlotter
