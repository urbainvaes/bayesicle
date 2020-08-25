import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import scipy.integrate as integrate
import hermipy as hm
import lib_inverse_problem
import lib_plotters

matplotlib.rc('font', size=16)
matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex=True)
matplotlib.rc('savefig', bbox='tight')
matplotlib.rc('figure', figsize=(14, 8))

hm.settings['cache'] = False
hm.settings['tensorize'] = True

flat = False


class ForwardFokkerPlanck:
    def __init__(self, N):
        self.fig, self.ax = None, None
        x = sym.symbols('x')
        f = sym.Function('f')(x)

        if flat:
            V = sym.Rational(1)
        else:
            V = (x**4/4 - x**2/2)  # b = - ∇V

        β = sym.Rational(1)  # a = 1/β

        # Interaction potential
        θ = 1
        V1 = θ*x**2/2
        V2 = - θ*x

        # Normalization
        if not flat:
            Z = integrate.quad(sym.lambdify(x, sym.exp(-V)), -7, 7)[0]
            V = sym.log(Z) + V
            assert abs(integrate.quad(sym.lambdify(x, sym.exp(-V)), -7, 7)[0] - 1) < 1e-5

        self.N = N
        x = sym.symbols('x')
        f = sym.Function('f')(x)
        fp = (V.diff(x)*f).diff(x) + (1/β)*f.diff(x, x)

        # Parameters of spectral discretization
        f_degree = 1
        degree = int(30 * f_degree)
        n_points = 2*degree + 1
        scaling = sym.Rational(.5) / np.sqrt(f_degree)

        # weight_fp = sym.exp(-β*V)
        weight_fp = sym.Rational(1)
        weight_gaussian = 1/sym.sqrt(2*sym.pi*scaling**2) \
            / sym.exp(x*x/(2*scaling**2))
        factor_fp = sym.sqrt(weight_gaussian*weight_fp)
        q = hm.Quad.gauss_hermite
        self.quad_fp = q(n_points, factor=factor_fp, cov=[[scaling**2]])
        self.quad_vi = hm.Quad.newton_cotes(n_points=[200], extrema=[4])
        self.mat_fp = self.quad_fp.discretize_op(fp, degree=degree)

        # Nonlocal interaction part
        fp1 = (V1.diff(x)*f).diff(x)
        fp2 = (V2.diff(x)*f).diff(x)
        self.mat_fp1 = self.quad_fp.discretize_op(fp1, degree=degree)
        self.mat_fp2 = self.quad_fp.discretize_op(fp2, degree=degree)

        # Integral and moment operators
        w = self.quad_fp.factor * self.quad_fp.factor \
            / self.quad_fp.position.weight()
        self.m0 = self.quad_fp.transform(w, degree=degree)
        self.m1 = self.quad_fp.transform(w * x, degree=degree)
        self.m2 = self.quad_fp.transform(w * x**2, degree=degree)

        # Initial condition
        mi = -1
        initial = sym.exp(-β*(V1 + mi*V2))
        initial = self.quad_fp.transform(initial, degree=degree)
        self.initial = 1/float(self.m0*initial) * initial

        self.T = 2
        self.time = np.linspace(0, self.T, N + 1)
        shape_control = [0*x + 1, x]
        # shape_control = [0*x + 1]
        self.nc = len(shape_control)

        control_fp = [-1*(s*f).diff(x) for s in shape_control]
        self.mat_control_fp = []
        for i in range(len(shape_control)):
            self.mat_control_fp.append(self.quad_fp.discretize_op(
                control_fp[i], degree=degree))

    def construct_controls(self, u):
        # Interpolation
        # controls = [interpolate.interp1d(self.time, s, kind='previous',
        #                                  fill_value='extrapolate') for s in u]
        # Legendre
        # controls = [lambda x: npp.legendre.legval(-1 + 2*x/self.T, s)
        #             for s in u]
        # Cosine-I
        controls = [lambda x, control=s: sum((c/(n+1)*np.cos(n*np.pi*x/self.T)
                                             for n, c in enumerate(control)))
                    for s in u]

        return controls

    def solve_state(self, u):
        """ Solve state equation

        :initial: Initial condition (Hermite series)
        :returns: An list of Hermite series

        """

        # Control signals as functions
        # Other 'kinds': cubic, left, right...
        controls = self.construct_controls(u)

        def matrix_fp(t, m):
            result = self.mat_fp.matrix \
                    + self.mat_fp1.matrix \
                    + m*self.mat_fp2.matrix
            for ci, mi in zip(controls, self.mat_control_fp):
                result = result + ci(t)*mi.matrix
            return result

        def dfdt(t, y):
            series = self.quad_fp.series(y)
            m = float(self.m1*series)
            return matrix_fp(t, m).dot(y)

        result = integrate.solve_ivp(dfdt, [0, self.T], self.initial.coeffs,
                                     'RK45', t_eval=self.time, atol=1e-11,
                                     rtol=1e-11)
        result = [self.quad_fp.series(y) for y in result.y.T]
        return result

    def __call__(self, u, symbolic=False):
        u = np.reshape(u, (self.nc, self.N))
        # u = np.hstack((u, np.array([[0], [0]])))
        result = self.solve_state(u)
        final = result[-1]
        m1 = float(self.m1*final)
        m2 = float(self.m2*final) - m1**2
        print("Evaluating G... Moments:", m1, m2)
        return m1, m2

    def plot(self, us, iteration, interactive=None):
        if not iteration % 1 == 0:
            return

        if self.fig is None and self.ax is None:
            self.fig, self.ax = plt.subplots(self.nc)
            if interactive:
                plt.ion()


# Parameters of the inverse problem {{{1
# ======================================

# Initialize forward model
G = ForwardFokkerPlanck(4)

# Forward model
forward = G.__call__

# Parameters
d = G.nc*G.N

# Desired moments
# y = np.array([1, 1])
y = np.array([0, 1])

# Covariance of noise and prior
γ, σ = 1, 10

# Covariance of noise and prior
Σ = np.diag([σ**2]*d)
Γ = np.diag([γ**2]*len(y))

# Inverse problem
ip = lib_inverse_problem.InverseProblem(forward, Γ, Σ, y)

# Plotters {{{1
# ==========


class Plotter:
    def __init__(self, ip, **config):
        self.fig, self.ax = plt.subplots(G.nc)
        if G.nc == 1:
            self.ax = [self.ax]

    def plot(self, iteration, data):
        us = data['ensembles']
        us = [np.reshape(u, (G.nc, G.N)) for u in us]
        # us = [np.hstack((u, np.array([[0], [0]]))) for u in us]
        controls = [G.construct_controls(u) for u in us]
        fine_time = np.linspace(0, G.T, 100)

        for i in range(G.nc):
            self.ax[i].clear()

        for c in controls:
            for i, u in enumerate(c):
                self.ax[i].plot(fine_time, u(fine_time))
        self.ax[0].set_title("Iteration: {}".format(iteration))


AllCoeffsPlotter = lib_plotters.AllCoeffsPlotter

# Delete local variables to make them unaccessible outside
del γ, σ, d, forward, Σ, Γ, y
