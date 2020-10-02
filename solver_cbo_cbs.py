"""Consensus-based solvers
"""

import os
import multiprocessing as mp
import collections
import numpy as np
import scipy as sp
import lib_misc

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
        self.frac_min = opts.get('frac_min', 1/5)
        self.frac_max = opts.get('frac_max', 1/2)
        self.reg = opts.get('reg', True)
        self.verbose = opts.get('verbose', default_settings['verbose'])

    def f_ensembles(self, ip, ensembles):
        the_function = ip.reg_least_squares if self.reg \
            else ip.least_squares

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

    def calculate_weights(self, f_ensembles):
        J = len(f_ensembles)
        n_iter_max = 1000
        # This helps to avoid numerical issues
        f_ensembles = f_ensembles - np.min(f_ensembles)
        for _ in range(n_iter_max):
            weights = np.exp(- self.beta * f_ensembles)
            sum_ess = np.sum(weights**2)
            ess = 0 if sum_ess == 0 else np.sum(weights)**2/sum_ess
            my_print = print if self.verbose else lambda s: None
            if self.adaptive and ess < int(self.frac_min*J):
                self.beta /= 1.1
                my_print("ESS = {} too small, decreasing β to {}"
                         .format(ess, self.beta))
            elif self.adaptive and ess > int(self.frac_max*J):
                self.beta *= 1.1
                my_print("ESS = {} too large, increasing β to {}"
                         .format(ess, self.beta))
            else:
                break
        else:
            print("Could not find suitable β")
        weights = weights / np.sum(weights)
        return weights, ess

    def step(self, ip, ensembles, filename=None):
        raise NotImplementedError


CbsIterationData = collections.namedtuple(
    'CbsIterationData', [
        'solver', 'ensembles', 'f_ensembles', 'beta',
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

    def step(self, ip, ensembles, filename=None):
        J = ensembles.shape[0]
        f_ensembles = self.f_ensembles(ip, ensembles)
        weights, ess = self.calculate_weights(f_ensembles)
        mean = np.sum(ensembles*weights.reshape(J, 1), axis=0)
        diff = ensembles - mean
        cov = [x.reshape(ip.d, 1) * x.reshape(1, ip.d) for x in diff]
        cov = np.sum(np.array(cov) * weights.reshape(J, 1, 1), axis=0)
        new_ensembles = np.zeros((J, ip.d))
        if self.opti:
            coeff_noise = sp.linalg.sqrtm(cov)
            if J <= ip.d:
                coeff_noise = np.real(coeff_noise)
            for j in range(J):
                new_ensembles[j] = mean \
                    + coeff_noise.dot(np.random.randn(ip.d))
        else:
            coeff_noise = np.sqrt((1 - np.exp(-2*self.dt))/2) \
                * sp.linalg.sqrtm(2*(1+self.beta)*cov)
            if J <= ip.d:
                coeff_noise = np.real(coeff_noise)
            coeff_noise = np.real(coeff_noise)
            for j in range(J):
                new_ensembles[j] = mean + np.exp(-self.dt)*diff[j] \
                                   + coeff_noise.dot(np.random.randn(ip.d))

        data = CbsIterationData(
            solver='cbs', ensembles=ensembles, f_ensembles=f_ensembles,
            beta=self.beta, weights=weights.reshape(J), ess=ess,
            dt=self.dt, new_ensembles=new_ensembles)

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

    def step(self, ip, ensembles, filename=None):
        J = ensembles.shape[0]
        f_ensembles = self.f_ensembles(ip, ensembles)
        weights, ess = self.calculate_weights(f_ensembles)
        mean = np.sum(ensembles*weights.reshape(J, 1), axis=0)
        diff = ensembles - mean
        new_ensembles = np.zeros((J, ip.d))

        for j in range(J):
            new_ensembles[j] = ensembles[j] - self.lamda * self.dt*diff[j] \
                + np.sqrt(self.dt)*self.sigma*diff[j]*np.random.randn(ip.d)

        if ip.eq_constraint is not None:
            for i, _ in enumerate(new_ensembles):
                new_ensembles[i] -= self.dt*(1/self.constraint_eps) \
                                * ip.eq_constraint(ensembles[i]) \
                                * ip.eq_constraint_grad(ensembles[i])

        if ip.ineq_constraint is not None:
            for i, _ in enumerate(new_ensembles):
                new_ensembles[i] -= self.dt*(1/self.constraint_eps) \
                                * max(ip.ineq_constraint(ensembles[i]), 0) \
                                * ip.ineq_constraint_grad(ensembles[i])

        data = CboIterationData(
            solver='cbo', ensembles=ensembles, f_ensembles=f_ensembles,
            beta=self.beta, weights=weights.reshape(J), ess=ess,
            dt=self.dt, new_ensembles=new_ensembles, sigma=self.sigma,
            lamda=self.lamda)

        if filename is not None:
            np.save("{}/{}".format(self.data_dir, filename),
                    data._asdict())

        return data
