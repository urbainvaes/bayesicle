import os
import numpy as np
import numpy.random as r
import scipy.linalg as la
import multiprocessing as mp
import collections
import lib_misc

data_root = "{}/solver_md".format(lib_misc.data_root)
default_settings = {
        'parallel': True,
        'adaptive': True,
        'dt_min': 1e-5,
        'dt_max': 1e3,
        'dirname': 'test',
        }

MdIterationData = collections.namedtuple(
    'MdIterationData', [
        'solver', 'theta', 'xis', 'delta', 'sigma', 'dt',
        'new_theta', 'new_xis', 'value_func', 'all_thetas'])



class MdSolver:

    def __init__(self, **opts):
        self.J = opts['J']
        self.delta = opts['delta']
        self.sigma = opts['sigma']
        self.dt = opts['dt']
        self.noise = opts['noise']
        self.reg = opts['reg']
        dirname = opts.get('dirname', default_settings['dirname'])
        self.data_dir = "{}/{}".format(data_root, dirname)
        os.makedirs(self.data_dir, exist_ok=True)
        self.parallel = opts.get('parallel', default_settings['parallel'])
        self.adaptive = opts.get('adaptive', default_settings['adaptive'])
        self.precond_vec = opts.get('precond_vec', None)
        self.precond_mat = opts.get('precond_mat', None)
        if self.precond_mat is not None:
            self.inv_precond_mat = la.inv(self.precond_mat)
        if self.adaptive:
            self.dt_min = opts.get('dt_min', default_settings['dt_min'])
            self.dt_max = opts.get('dt_max', default_settings['dt_max'])

    def precond_map(self, u):
        if self.precond_vec is None:
            return u
        return self.inv_precond_mat.dot(u - self.precond_vec)

    def precond_unmap(self, u):
        if self.precond_vec is None:
            return u
        return self.precond_vec + self.precond_mat.dot(u)

    def g_ensembles(self, ip, ensembles):
        # Strange but seemingly necessary to avoid pickling issue? \_(")_/
        global forward

        def forward(u):
            return ip.forward(self.precond_unmap(u))
        # -------------------------------- #
        if self.parallel:
            pool = mp.Pool(4)
            g_ensembles = pool.map(forward, ensembles)
            pool.close()
        else:
            g_ensembles = np.array([forward(u) for u in ensembles])
        return g_ensembles

    def step(self, ip, data, filename=None):
        if isinstance(iterand, MdIterationData):
            theta = data.theta
            xis = data.xis
            all_thetas = data.all_thetas
        elif isinstance(iterand, np.ndarray):
            theta = iterand
            xis = np.random.randn(self.J, ip.d)
            all_thetas = np.reshape(theta, (1, ip.d))
        else:
            print("Invalid argument!")
            raise TypeError

        # Preconditioning
        unmapped_theta = theta
        theta = self.precond_map(theta)

        J, dim_u = self.J, ip.d
        g_theta = ip.forward(unmapped_theta)

        func = ip.reg_least_squares if self.reg else ip.least_squares
        value_func = func(unmapped_theta)

        # Calculation of the LHS in the inner product
        ensembles = np.tile(theta, (J, 1)) + self.sigma*xis
        g_thetas = self.g_ensembles(ip, ensembles)
        grads_approx = (1/self.sigma)*np.array([g - g_theta for g in g_thetas])

        if self.noise or self.reg:
            Cxi = (1/J) * xis.T.dot(xis)
            if self.noise:
                sqrt2Cxi = la.sqrtm(2*Cxi)
                if J <= dim_u:
                    sqrt2Cxi = np.real(sqrt2Cxi)
                dW = r.randn(dim_u)

        drift = - (Cxi.dot(ip.inv_Σ.dot(theta)) if self.reg else 0)
        inner_product = ip.inv_Γ.dot(g_theta - ip.y)
        for grad_approx, xi in zip(grads_approx, xis):
            coeff = grad_approx.dot(inner_product)
            drift -= (1/J) * coeff * xi

        my_dt = self.dt
        if self.adaptive:
            dt_0, dt_min, dt_max = self.dt, self.dt_min, self.dt_max
            my_dt = dt_0/(dt_0/dt_max + la.norm(drift, 2))
            my_dt = max(my_dt, dt_min)
            print("New time step: {}".format(my_dt))

        new_theta = theta + drift*my_dt + \
            (np.sqrt(my_dt)*sqrt2Cxi.dot(dW) if self.noise else 0)
        alpha = np.exp(-my_dt/self.delta**2)
        new_xis = alpha * xis + np.sqrt(1-alpha**2) * r.randn(J, dim_u)

        # Undo preconditioning
        unmapped_new_theta = self.precond_unmap(new_theta)

        data = MdIterationData(
            solver='md', theta=unmapped_theta, xis=xis, delta=self.delta,
            sigma=self.sigma, dt=my_dt, new_theta=unmapped_new_theta,
            new_xis=new_xis, value_func=value_func, )

        if filename is not None:
            np.save("{}/{}".format(self.data_dir, filename),
                    data._asdict())

        return data


class MdSimulation():

    def init(self, ip, label, initial, **opts):
        self.ip = ip
        self.label = label
        self.theta = initial

        self.all_thetas = np.reshape(initial, (1, ip.d))
        self.iteration = 0
        self.solver = MdSolver(**opts)

    def step(self):
        self.step(m.ip, data, filename="iteration-{:04d}.npy".format(i))
