import os
import collections
import multiprocessing as mp
import numpy as np
import scipy.linalg as la
import lib_misc

data_root = "{}/solver_eks".format(lib_misc.data_root)
default_settings = {
        'verbose': False,
        'parallel': True,
        'adaptive': True,
        'dirname': 'test',
        'dt_max': 100000,
}

EksIterationData = collections.namedtuple(
    'CbsIterationData', [
        'solver', 'ensembles', 'g_ensembles', 'dt', 'new_ensembles'])


class EksSolver:

    def __init__(self, **opts):
        self.dt = opts['dt']
        self.noise = opts['noise']
        self.reg = opts['reg']
        self.parallel = opts.get('parallel', default_settings['parallel'])
        self.adaptive = opts.get('adaptive', default_settings['adaptive'])
        self.verbose = opts.get('verbose', default_settings['verbose'])
        if self.adaptive:
            self.dt_max = opts.get('dt_max', default_settings['dt_max'])
        dirname = opts.get('dirname', default_settings['dirname'])
        self.data_dir = "{}/{}".format(data_root, dirname)
        os.makedirs(self.data_dir, exist_ok=True)

        # Constraint
        self.constraint_eps = opts.get('epsilon', .1)

    def g_ensembles(self, ip, ensembles):
        # Strange but seemingly necessary to avoid pickling issue? \_(")_/
        global forward

        def forward(u):
            return ip.forward(u)
        # -------------------------------- #
        if self.parallel:
            pool = mp.Pool(4)
            g_ensembles = pool.map(forward, ensembles)
            pool.close()
        else:
            g_ensembles = [ip.forward(u) for u in ensembles]
        return np.array(g_ensembles)

    def step(self, ip, ensembles, filename, **opts):

        """Performs a step of the EKS / EKI

        Note:
            The behavior depends on the options 'noise' and 'reg'
            If (noise == True and reg == True) -> sample from posterior
            Elif(noise == False and reg == True) -> find the MAP, i.e. the mode
                of the posterior, or minimizer of the regularized least-squares
                functional.
            Elif(noise == False and reg == False) -> find minimum of Φ, i.e. of
                the non-regularized mean-square functional.

        Args:
            ip: An InverseProblem
            ensembles: The ensembles, of shape (J x d)

        Returns:
            Ensembles at the next time step

        """

        J, d = ensembles.shape
        g_ensembles = self.g_ensembles(ip, ensembles)
        mean_ensembles = np.mean(ensembles, axis=0)
        mean_g_ensembles = np.mean(g_ensembles, axis=0)
        my_print = print if self.verbose else lambda s: None

        if self.noise or self.reg:
            theta_bar = np.tile(np.mean(ensembles, axis=0), (J, 1))
            diff_ensembles_mean = ensembles - theta_bar
            C_theta = (1/J) * (diff_ensembles_mean.T).dot(diff_ensembles_mean)

            if self.noise:
                sqrtCtheta = la.sqrtm(2*C_theta)
                if J <= d:
                    sqrtCtheta = np.real(sqrtCtheta)
                dW = np.random.randn(J, d)

        if not self.noise:
            my_print("Agreement of mean G(θ): {}".format(
                la.norm(mean_g_ensembles - ip.y, 2)))

        # !! 'drifts' is in fact the negative drift
        diff_data = g_ensembles - ip.y
        diff_mean_forward = g_ensembles - mean_g_ensembles
        diff_mean = ensembles - mean_ensembles
        coeffs = diff_mean_forward.dot(ip.inv_Γ).dot(diff_data.T)
        drifts = (1/J)*np.tensordot(coeffs, diff_mean, [[0], [0]])
        if self.reg:
            _mat = C_theta.dot(ip.inv_Σ)
            drifts += np.tensordot(ensembles, _mat, [[1], [1]])

        # drifts = []
        # for (j, u), g in zip(enumerate(ensembles), g_ensembles):
        #     coeffs = (g_ensembles - mean_g_ensembles).dot(ip.inv_Γ.dot(g - ip.y))
        #     coeffs = np.array([coeffs]).T  # needed for the multiplication
        #     drift = np.mean(coeffs*(ensembles - mean_ensembles), axis=0) \
        #         + (C_theta.dot(ip.inv_Σ.dot(u)) if self.reg else 0)
        #     drifts.append(drift)

        # if ip.eq_constraint is not None:
        #     for i, _ in enumerate(drifts):
        #         drifts[i] += (1/self.constraint_eps) \
        #                       * ip.eq_constraint(ensembles[i]) \
        #                       * ip.eq_constraint_grad(ensembles[i])

        # if ip.ineq_constraint is not None:
        #     for i, _ in enumerate(drifts):
        #         drifts[i] += (1/self.constraint_eps) \
        #                       * ip.ineq_constraint(ensembles[i]) \
        #                       * max(ip.ineq_constraint(ensembles[i]), 0) \
        #                       * ip.ineq_constraint_grad(ensembles[i])

        my_dt = self.dt
        if self.adaptive:
            norm_drift = np.sqrt(sum(la.norm(d, 2)**2 for d in drifts))
            my_dt = self.dt/(self.dt/self.dt_max + norm_drift)
            print("Norm of drift: {}".format(norm_drift))
            print("New time step: {}".format(my_dt))

        # new_ensembles = ensembles - my_dt*drifts \
        #     + (np.sqrt(my_dt) * dW.dot(sqrtCtheta) if self.noise else 0)

        # constraint = ensembles[:, 0]**2 + ensembles[:, 1]**2 - (2*2.5**2)
        # constraint = np.reshape(constraint, (len(constraint), 1))

        new_ensembles = ensembles - my_dt*drifts \
            + (np.sqrt(my_dt) * dW.dot(sqrtCtheta) if self.noise else 0)
        # new_ensembles = new_ensembles / (1 + constraint*my_dt/self.constraint_eps)

        # import ipdb; ipdb.set_trace()
        # new_ensembles = np.zeros((J, ip.d))
        # for (j, u), drift in zip(enumerate(ensembles), drifts):
        #     new_ensembles[j] = u - my_dt*drift \
        #         + (np.sqrt(my_dt) * sqrtCtheta.dot(dW[j])
        #            if self.noise else 0)

        data = EksIterationData(
            solver='eks', ensembles=ensembles, g_ensembles=g_ensembles,
            dt=my_dt, new_ensembles=new_ensembles)

        if filename is not None:
            np.save("{}/{}".format(self.data_dir, filename),
                    data._asdict())

        return data
