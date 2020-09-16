import numpy as np
import lib_inverse_problem
import lib_plotters
import sympy as sym
import matplotlib.pyplot as plt

n_grid = 100
z = np.linspace(0, 1, n_grid)
Δz = z[1] - z[0]

cs0 = 100
k = 50
z0 = .1
n = 1
z1 = .5
alpha = 5

# def forward(cs0, k, z0, n, z1, alpha):

cs = np.where(z <= z0, cs0,
              np.where(z <= z1, cs0 * (1 + k*(z - z0))**n,
                       alpha * cs0  * (1 + k*(z1 - z0))**n))

cs = 0*z + cs0

F = .1
def forcing(t):
    return F*np.sin(20*np.pi*t)


# Initial condition
u = z*0
v = z*0  # v = dudt

# Time step and number of iterations
n_periods = 1
# iters_per_period = 1000000
# n_iter = n_periods*iters_per_period
Δt = .001 * Δz/cs[-1]
n_iter = int(1/Δt)

# Initial time
t = 0

fig, ax = plt.subplots()

for i in range(n_iter):
    print(i)

    if i % (n_iter // 100) == 0:
        ax.clear()
        ax.set_ylim(1.2*min(-F, np.min(u)), 1.2*max(F, np.max(u)))
        ax.plot(z, u, '.-')
        ax.set_title("Time: {} [sec]".format(t))
        # ax.plot(z, v)
        plt.draw()
        plt.pause(.01)

    # Add artificial data at z = 0 (extende u)
    u0_u = np.insert(u, 0, u[0])

    # Discretize second derivative
    du = (1/Δz) * np.insert(cs**2, 0, 0)[:-1] * (u0_u[1:] - u0_u[:-1])
    ddu = (1/Δz) * (du[1:] - du[:-1])

    # Time
    t = (i+1)*Δt

    # Update u
    u += v*Δt
    u[-1] = forcing(t)

    # Update v
    v[:-1] += ddu*Δt



# Solution of the wave equation


# Dimensions of the model
d, K = 2, 1

# Data = lower bound
y = np.array([-.1])

# Covariance of noise and prior
# γ, σ = 1, .5
γ, σ = 1, 1000000
Γ = np.diag([γ**2]*K)
Σ = np.diag([σ**2]*d)

# # Constraint
# vx, vy = sym.symbols('x y', real=True)
# constraint = vx**2 + vy**2 - (2*np.sqrt(2))**2
# # constraint = vx**2 + vy**2 + (vx + vy)**2/(.5 + (vx - vy)**2) - 2**2
# # constraint = sym.cos(vx) - vy
# grad_constraint = [constraint.diff(vx), constraint.diff(vy)]
# constraint = sym.lambdify((vx, vy), constraint)
# grad_constraint = sym.lambdify((vx, vy), grad_constraint)

# # constraints = {
# #         'eq_constraint': lambda x: constraint(*x),
# #         'eq_constraint_grad': lambda x: np.array(grad_constraint(*x)),
# #         }

# constraints = {
#         'ineq_constraint': lambda x: constraint(*x),
#         'ineq_constraint_grad': lambda x: np.array(grad_constraint(*x)),
#         }

# ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y, **constraints)
# Plotter = lib_plotters.TwoDimPlotter

