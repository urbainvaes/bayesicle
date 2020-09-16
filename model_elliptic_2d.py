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

    def __init__(self, N, K):
        # Variables
        x, y = sym.symbols('x[0], x[1]', real=True, positive=True)
        f = sym.Function('f')(x, y)

        # Precision (inverse covariance) operator, when α = 1
        tau, alpha = 3, 2
        precision = (- sym.diff(f, x, x) - sym.diff(f, y, y) + tau**2*f)

        indices = [(m, n) for m in range(0, N) for n in range(0, N)]
        # indices = indices[:-1]
        self.indices = sorted(indices, key=sum)
        eig_f = [sym.cos(i[0]*sym.pi*x) * sym.cos(i[1]*sym.pi*y)
                 for i in self.indices]

        # Eigenvalues of the covariance operator
        eig_v = [1/(precision.subs(f, e).doit()/e).simplify()**alpha
                 for e in eig_f]

        grid = np.linspace(0, 1, K + 2)[1:-1]
        x_obs, y_obs = np.meshgrid(grid, grid)
        self.x_obs, self.y_obs = x_obs.reshape(K*K), y_obs.reshape(K*K)

        # Basis_functions
        self.functions = [f*np.sqrt(float(v)) for f, v in zip(eig_f, eig_v)]

        # --- THIS PART USED TO NOT BE PARALELLIZABLE --- #
        # Create mesh and define function space
        n_mesh = 80
        mesh = fen.UnitSquareMesh(n_mesh, n_mesh)
        self.f_space = fen.FunctionSpace(mesh, 'P', 2)

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
        # rhs = fen.Expression('sin(x[0])*sin(x[1])', degree=2)
        self.lin_func = rhs*self.test_f*fen.dx
        # ----------------------------------------------- #

    def __call__(self, u_test, return_sol=False, return_permeability=False):

        # Pad with zeros
        u_test = np.pad(u_test, (0, len(self.indices) - len(u_test)))

        # Assembling diffusivity
        log_coeff = sum([ui*fi for ui, fi in zip(u_test, self.functions)], 0)
        if return_permeability:
            return log_coeff

        ccode_coeff = sym.ccode(sym.exp(log_coeff))
        diff = fen.Expression(ccode_coeff, degree=2)

        # Define bilinear form in variational problem
        a_bil = diff*fen.dot(fen.grad(self.trial_f),
                             fen.grad(self.test_f))*fen.dx

        # Compute solution
        sol = fen.Function(self.f_space)
        fen.solve(a_bil == self.lin_func, sol, self.bound_cond)

        evaluations = [sol(xi, yi) for xi, yi in zip(self.x_obs, self.y_obs)]
        # print("Evaluating G...:", np.array(evaluations))
        return sol if return_sol else np.array(evaluations)


# Test code {{{1
# ==============
if __name__ == "__main__":
    G = ForwardDarcy(4, 10)
    u_truth = np.random.randn(len(G.indices))
    solution = G(u_truth, return_sol=True)
    p = fen.plot(solution)
    plt.colorbar(p)
    plt.show()

# Parameters of the inverse problem {{{1
# ======================================

# Covariance of noise and prior
γ, σ = .01, 1

# Initialize forward model
dx, Kx = 6, 10
G = ForwardDarcy(dx, Kx)

# Fourier coefficients of the true (scalar) diffusivity
u = σ*np.random.randn(len(G.indices))

# Forward model
forward = G.__call__

# Observation without noise
y = forward(u)
K = len(y)

# Forward model for approximation
dx, Kx = 3, 10
G = ForwardDarcy(dx, Kx)

# Approximation
d = len(G.indices)
u_truth = u
u = u[:d]

# Covariance of noise and prior
Σ = np.diag([σ**2]*d)
Γ = np.diag([γ**2]*K)

# Square root of covariance matrix
rtΓ = np.diag([γ]*K)

# Noisy observation
y = y + rtΓ.dot(np.random.randn(K))

# Inverse problem
ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y, unknown=u)

# Plotters {{{1
# ==========


class MainModesPlotter(lib_plotters.MainModesPlotter):

    def __init__(self, ip, **config):
        super().__init__(ip, **config)
        ix = [G.indices[c] for c in self.coeffs]
        ax0, ax1 = self.ax[0], self.ax[1]
        ax0.set_title("Fourier coefficients {} and {}".format(ix[0], ix[1]))
        ax1.set_title("Fourier coefficients {} and {}".format(ix[0], ix[2]))
        ax0.set_xlabel("Coefficient {}".format(ix[0]))
        ax0.set_ylabel("Coefficient {}".format(ix[1]))
        ax1.set_xlabel("Coefficient {}".format(ix[0]))
        ax1.set_ylabel("Coefficient {}".format(ix[2]))
        ax0.set_xlim([-6, 6])
        ax0.set_ylim([-6, 6])
        ax1.set_xlim([-6, 6])
        ax1.set_ylim([-6, 6])


class AllCoeffsPlotter(lib_plotters.AllCoeffsPlotter):

    def plot_all_coeffs(self, iteration, data):
        self.ax.clear()
        super().plot(iteration, data)
        self.ax.set_xticks(np.arange(len(G.indices)))
        self.ax.set_xticklabels((str(i) for i in G.indices))


# Delete local variables to make them unaccessible outside
del γ, σ, dx, Kx, u, d, forward, Σ, Γ, rtΓ, y
