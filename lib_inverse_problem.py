import scipy.integrate as integ
import scipy.optimize as opti
import numpy as np


def direct_min(f, x0):
    argmin = opti.basinhopping(f, x0)
    if argmin.lowest_optimization_result.success:
        argmin, fmin = argmin.x, argmin.fun
    else:
        print("Warning: Could not locate minimum!")
        argmin, fmin = argmin.x, argmin.fun
    return argmin, fmin


class InverseProblem:

    def __init__(self, forward, Γ, Σ, y, unknown=None, **constraints):
        self.forward = forward
        self.d = len(Σ)
        self.K = len(Γ)
        self.y = y
        self.inv_Γ = np.linalg.inv(Γ)
        self.inv_Σ = np.linalg.inv(Σ)
        self.unknown = unknown

        self.Z_normal = None
        self.argmin = None
        self.fmin = None

        self.eq_constraint = constraints.get('eq_constraint', None)
        self.eq_constraint_grad = constraints.get('eq_constraint_grad', None)
        self.ineq_constraint = constraints.get('ineq_constraint', None)
        self.ineq_constraint_grad = constraints.get('ineq_constraint_grad', None)

    def reg_least_squares(self, u):
        diff = self.forward(u) - self.y
        return (1/2)*diff.dot(self.inv_Γ.dot(diff)) \
            + (1/2)*u.dot(self.inv_Σ.dot(u))

    def map_estimator(self):
        if self.argmin is None:
            self.argmin, self.fmin = direct_min(self.reg_least_squares,
                                                np.zeros(self.d))
        return self.argmin, self.fmin

    def least_squares(self, u):
        diff = self.forward(u) - self.y
        return (1/2)*diff.dot(self.inv_Γ.dot(diff))

    def least_squares_array(self, *args):
        if isinstance(args[0], float):
            args = tuple(np.array([a]) for a in args)
        shape = args[0].shape
        args = tuple(a.reshape(np.prod(shape)) for a in args)
        xs = np.vstack(args).T
        f = self.reg_least_squares
        result = np.array([f(x) for x in xs])
        return result.reshape(shape)

    def posterior(self, *args):
        argmin, fmin = self.map_estimator()
        assert len(args) == self.d
        if isinstance(args[0], float):
            args = tuple(np.array([a]) for a in args)
        shape = args[0].shape
        args = tuple(a.reshape(np.prod(shape)) for a in args)
        xs = np.vstack(args).T
        f = self.reg_least_squares
        result = np.array([np.exp(-(f(x) - fmin)) for x in xs])
        return result.reshape(shape)

    def normalization(self):
        if self.Z_normal is not None:
            return self.Z_normal

        if self.d == 1:
            bound = 5
            self.Z_normal = integ.quad(self.posterior, -bound, bound)[0]
        elif self.d == 2:
            Lx, Ly = 1.5, 1.5
            xmin, xmax = self.argmin[0] - Lx, self.argmin[0] + Lx
            ymin, ymax = self.argmin[1] - Ly, self.argmin[1] + Ly
            self.Z_normal = integ.dblquad(lambda y, x: self.posterior(x, y),
                                          xmin, xmax, ymin, ymax)[0]
        return self.Z_normal
