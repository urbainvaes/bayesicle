import scipy.integrate as integ
import scipy.optimize as opti
import numpy as np
import lib_misc

class OptimizationProblem:

    def __init__(self, d, objective, argmin=None, fmin=None, **constraints):
        self.objective = objective
        self.argmin = argmin
        self.fmin = fmin
        self.reg_least_squares = self.objective
        self.d = d

        # For now
        self.eq_constraint = None
        self.ineq_constraint = None

    def solution(self):
        if self.argmin is None:
            self.argmin, self.fmin = lib_misc.direct_min(self.objective,
                                                         np.zeros(self.d))
        return self.argmin, self.fmin

    def objective_array(self, *args):
        if isinstance(args[0], float):
            args = tuple(np.array([a]) for a in args)
        shape = args[0].shape
        args = tuple(a.reshape(np.prod(shape)) for a in args)
        xs = np.vstack(args).T
        f = self.objective
        result = np.array([f(x) for x in xs])
        return result.reshape(shape)

    def map_estimator(self):
        return self.solution()

    def least_squares_array(self, *args):
        return self.objective_array(*args)
