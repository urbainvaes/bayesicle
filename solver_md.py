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
        'new_theta', 'new_xis', 'value_func'])



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
            self.sqrt_precond_mat = la.sqrtm(self.precond_mat)
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

    def step(self, ip, theta, xis, filename=None):

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

        drift = 0
        if self.reg:
            inv_Σ = ip.inv_Σ
            prior_μ = np.zeros(len(theta))
            if self.precond_mat is not None:
                prior_μ = prior_μ - self.precond_vec
                inv_Σ = self.sqrt_precond_mat.dot(inv_Σ).dot(self.sqrt_precond_mat)
            drift = - Cxi.dot(inv_Σ).dot(theta - prior_μ)

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
            new_xis=new_xis, value_func=value_func,)

        if filename is not None:
            np.save("{}/{}".format(self.data_dir, filename),
                    data._asdict())

        return data


class MdSimulation:

    def __init__(self, ip, initial, solver, save_step=50):
        self.ip = ip
        self.theta = initial
        self.solver = solver
        self.save_step = save_step

        self.xis = np.random.randn(solver.J, ip.d)
        self.all_thetas = []
        self.all_fthetas = []
        self.iteration = 0

    def step(self):
        data = self.solver.step(self.ip, self.theta, self.xis,
                                filename="iteration-{:04d}.npy".format(self.iteration))
        self.iteration += 1
        self.all_thetas.append(data.theta)
        self.all_fthetas.append(data.value_func)
        print("Step: {}".format(np.linalg.norm(self.theta - data.new_theta)))
        self.theta = data.new_theta
        self.xis = data.new_xis

        plot_step = 50
        if self.iteration % self.save_step == 0:
            filename="simulation-iteration-{:04d}.npy".format(self.iteration)
            np.save("{}/{}".format(self.solver.data_dir, filename),
                    self.get_data())

        return data

    def get_data(self):
        all_thetas = np.asarray(self.all_thetas).reshape(self.iteration, self.ip.d)
        all_fthetas = np.asarray(self.all_thetas).reshape(self.iteration, self.ip.d)
        return {'solver': 'md', 'ensembles': all_thetas, 'f_ensembles': all_fthetas}
