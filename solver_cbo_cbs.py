"""Consensus-based solvers
"""

import os
import multiprocessing as mp
import collections
import numpy as np
import scipy as sp
import lib_misc

import lib_inverse_problem
import lib_opti_problem

default_settings = {
        'verbose': False,
        'parallel': True,
        'adaptive': True,
        'beta': .01,
        'dirname': 'test',
}


class CbSolver:

    def __init__(self, **opts):
        self.dt = opts['dt']
        self.adaptive = opts.get('adaptive', False)
        self.beta = opts.get('beta', default_settings['beta'])  # Last β
        self.parallel = opts.get('parallel', default_settings['parallel'])
        self.ess = opts.get('ess', 1/2)
        self.reg = opts.get('reg', True)
        self.verbose = opts.get('verbose', default_settings['verbose'])

    def f_ensembles(self, problem, ensembles):
        if isinstance(problem, lib_inverse_problem.InverseProblem):
            the_function = problem.reg_least_squares if self.reg \
                else problem.least_squares
        elif isinstance(problem, lib_opti_problem.OptimizationProblem):
            the_function = problem.objective

        # Strange but seemingly necessary to avoid pickling issue? \_(")_/
        global function

        def function(u):
            return the_function(u)
        # -------------------------------- #

        if self.parallel:
            pool = mp.Pool(4)
            f_ensembles = np.array(pool.map(function, ensembles))
            pool.close()
        else:
            f_ensembles = np.array([function(u) for u in ensembles])
        return f_ensembles

    def calculate_weights(self, fensembles):
        J = len(fensembles)
        fensembles = fensembles - np.min(fensembles)

        def get_ess(β):
            weights = np.exp(- β*fensembles)
            return weights, np.sum(weights)**2/np.sum(weights**2)/J

        if self.adaptive:
            β1, β2 = 0, 1e20
            β = β2
            weights, ess = get_ess(β2)
            if ess > self.ess:
                print("Can't find β")
                return β

            while abs(ess - self.ess) > 1e-3:
                β = (β1 + β2)/2
                weights, ess = get_ess(β)
                if ess > self.ess:
                    β1 = β
                else:
                    β2 = β
            self.beta = β
        else:
            weights, ess = get_ess(self.beta)
        return weights / np.sum(weights), ess

    def step(self, problem, ensembles, filename=None):
        raise NotImplementedError


CbsIterationData = collections.namedtuple(
    'CbsIterationData', ['opti', 'solver', 'ensembles', 'f_ensembles', 'beta',
                         'weights', 'ess', 'dt', 'new_ensembles'])


class CbsSolver(CbSolver):

    def __init__(self, **opts):
        super().__init__(**opts)
        self.adaptive = opts.get('adaptive', default_settings['adaptive'])
        self.opti = opts.get('opti', False)
        self.reg = opts.get('reg', True)
        dirname = opts.get('dirname', default_settings['dirname'])
        data_root = "{}/solver_cbs".format(lib_misc.data_root)
        self.data_dir = "{}/{}".format(data_root, dirname)
        os.makedirs(self.data_dir, exist_ok=True)

    def step(self, problem, ensembles, filename=None):
        J = ensembles.shape[0]
        f_ensembles = self.f_ensembles(problem, ensembles)
        weights, ess = self.calculate_weights(f_ensembles)
        mean = np.sum(ensembles*weights.reshape(J, 1), axis=0)
        diff = ensembles - mean
        cov = [x.reshape(problem.d, 1) * x.reshape(1, problem.d) for x in diff]
        cov = np.sum(np.array(cov) * weights.reshape(J, 1, 1), axis=0)
        new_ensembles = np.zeros((J, problem.d))
        coeff_noise = np.sqrt((1 - np.exp(-2*self.dt))/2) \
            * sp.linalg.sqrtm(2*cov*(1 if self.opti else (1+self.beta)))
        if J <= problem.d:
            coeff_noise = np.real(coeff_noise)
        coeff_noise = np.real(coeff_noise)
        for j in range(J):
            new_ensembles[j] = mean + np.exp(-self.dt)*diff[j] \
                               + coeff_noise.dot(np.random.randn(problem.d))

        data = CbsIterationData(
            solver='cbs', opti=self.opti, ensembles=ensembles,
            f_ensembles=f_ensembles, beta=self.beta,
            weights=weights.reshape(J), ess=ess, dt=self.dt,
            new_ensembles=new_ensembles)

        if filename is not None:
            np.save("{}/{}".format(self.data_dir, filename),
                    data._asdict())

        return data


CboIterationData = collections.namedtuple(
    'CboIterationData', [
        'solver', 'ensembles', 'f_ensembles', 'beta',
        'weights', 'ess', 'dt', 'new_ensembles',
        'lamda', 'sigma'])


class CboSolver(CbSolver):

    def __init__(self, **opts):
        super().__init__(**opts)
        self.lamda = opts.get('lamda', 1)
        self.sigma = opts.get('sigma', 1)
        dirname = opts.get('dirname', default_settings['dirname'])
        data_root = "{}/solver_cbo".format(lib_misc.data_root)
        self.data_dir = "{}/{}".format(data_root, dirname)
        os.makedirs(self.data_dir, exist_ok=True)

        # Constraint
        self.constraint_eps = opts.get('epsilon', .1)

    def step(self, problem, ensembles, filename=None):
        J = ensembles.shape[0]
        f_ensembles = self.f_ensembles(problem, ensembles)
        weights, ess = self.calculate_weights(f_ensembles)
        mean = np.sum(ensembles*weights.reshape(J, 1), axis=0)
        diff = ensembles - mean
        new_ensembles = np.zeros((J, problem.d))

        for j in range(J):
            new_ensembles[j] = ensembles[j] - self.lamda * self.dt*diff[j] \
                + np.sqrt(self.dt)*self.sigma*diff[j]*np.random.randn(problem.d)

        if problem.eq_constraint is not None:
            for i, _ in enumerate(new_ensembles):
                new_ensembles[i] -= self.dt*(1/self.constraint_eps) \
                                * problem.eq_constraint(ensembles[i]) \
                                * problem.eq_constraint_grad(ensembles[i])

        if problem.ineq_constraint is not None:
            for i, _ in enumerate(new_ensembles):
                new_ensembles[i] -= self.dt*(1/self.constraint_eps) \
                                * max(problem.ineq_constraint(ensembles[i]), 0) \
                                * problem.ineq_constraint_grad(ensembles[i])

        data = CboIterationData(
            solver='cbo', ensembles=ensembles, f_ensembles=f_ensembles,
            beta=self.beta, weights=weights.reshape(J), ess=ess,
            dt=self.dt, new_ensembles=new_ensembles, sigma=self.sigma,
            lamda=self.lamda)

        if filename is not None:
            np.save("{}/{}".format(self.data_dir, filename),
                    data._asdict())

        return data
