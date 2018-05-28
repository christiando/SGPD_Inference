"""
Copyright (C) 2018, Christian Donner

This file is part of SGPD_Inference.

SGPD_Inference is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Foobar is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SGPD_Inference.  If not, see <http://www.gnu.org/licenses/>.
"""
__author__ = 'Christian Donner'
__email__ = 'christian.donner(at)bccn-berlin.de'
__license__ = 'gpl-3.0'

import numpy
from pypolyagamma import PyPolyaGamma, pgdrawvpar
import time
from .basemeasures import BaseMeasure
from copy import copy

class GS_SGPD():
    """ Gibbs sampler for SGPD model.
    """

    def __init__(self, X, cov_params,
                 base_measure, lmbda=None, burnin=1000,
                 num_integration_points=1000, max_iterations=2000,
                 update_hyperparams=True, update_basemeasure=True, nthreads=1,
                 gp_mu=0, sample_hyperparams_iter=10):
        """ Initialises class of Gibbs sampler. Sampled data is saved in
        dictionary 'self.data'.

        The dictionary self.data contains all the sampled data. 'X' are the
        locations (observations and latent), 'g' the GP at these locations,
        'lmbda' the max rate of latent Poisson process, 'cov_params' the kernel
        parameters, 'M' the number of latent events, 'time' the time for
        samples, 'bm_params' the base measure parameters, 'gp_mu' the mean of
        the GP prior.

        :param X: Data.
        :type X: numpy.ndarray [instances x features]
        :param cov_params: Kernel hyperparameters. List with first entry the
        prefactor and second a D-dimensional array with length scales.
        :type cov_params: list
        :param base_measure:
        :type base_measure: BaseMeasure
        :param lmbda: Initial value for max. Poisson rate. If None
        it will be equal to number of data points. Default is None.
        :type lmbda: float
        :param burnin: Number of iteration before the posterior will be
        sampled. Default=1000.
        :type burnin: int
        :param num_integration_points: Number of integration points. Only
        used for predictive likelihood. Default=1000.
        :type num_integration_points: int
        :param max_iterations: Number of iterations the posterior is sampled.
        Default=2000.
        :type max_iterations: int
        :param update_hyperparams: Whether GP hyperparameters should be
        sampled. Default=True.
        :type update_hyperparams: bool
        :param update_basemeasure: Whether base measure parameters should be
        sampled. Can only be done for certain base measure ('normal',
        'laplace', 'standard_t'). Default=True.
        :type update_basemeasure: bool
        :param nthreads: Number of threads used for PG sampling. Default=1.
        :type nthreads: int
        :param gp_mu: Mean of GP prior.
        :type gp_mu: float
        :param sample_hyperparams_iter: Every x^th step hyperparameters are
        sampler. Default=0.
        :type sample_hyperparams_iter: float
        """
        self.max_iterations = int(max_iterations)
        self.D = X.shape[1]
        self.cov_params = cov_params
        self.X = X
        self.N = self.X.shape[0]
        self.base_measure = base_measure
        self.noise = 1e-4

        if lmbda is None:
            self.lmbda = self.N / 1.
        else:
            self.lmbda = lmbda
        seeds = numpy.random.randint(2 ** 16, size=nthreads)
        self.pg = [PyPolyaGamma(seed) for seed in seeds]
        self.M = int(self.lmbda)
        self.M_save = numpy.empty(self.max_iterations)
        # Position of all events (first N are the actual observed ones)
        self.X_all = numpy.empty([self.N+self.M, self.D])
        self.X_all[:self.N] = self.X
        self.X_all[self.N:] = base_measure.sample_density(self.M)
        self.marks = numpy.empty(self.N + self.M)
        self.K = self.cov_func(self.X_all,self.X_all)
        self.K += self.noise * numpy.eye(self.K.shape[0])
        self.L = numpy.linalg.cholesky(self.K)
        self.L_inv = numpy.linalg.solve(self.L, numpy.eye(self.L.shape[0]))
        self.K_inv = self.L_inv.T.dot(self.L_inv)
        self.gp_mu = gp_mu
        self.pred_log_likelihood = []
        self.g = numpy.zeros([self.N + self.M])
        # Probability of insertion or deletion proposal
        self.num_iterations = 0
        self.burnin = int(burnin)
        self.num_integration_points = num_integration_points
        self.place_integration_points()
        self.update_hyperparams = update_hyperparams
        self.update_basemeasure = update_basemeasure
        self.update_hyperparams_iter = sample_hyperparams_iter


        self.data = {'X':[], 'g':[], 'lmbda':[], 'cov_params':[], 'M':[],
                       'time': [], 'bm_params':[], 'gp_mu': []}

    def run(self):
        """ Runs initialised Gibbs sampler.
        """

        self.data['time'].append(time.perf_counter())

        for i in range(self.max_iterations + self.burnin):
            self.sample_marks()
            self.sample_g()
            if self.update_hyperparams and i % self.update_hyperparams_iter == 0:
                self.sample_kernel_params()
                self.sample_GP_mean_prior()
            if self.update_basemeasure and i % self.update_hyperparams_iter == 0:
                self.sample_basemeasure()
            if i % 100 == 0:
                print('%d Iterations of %d -- Currently %d latent events'
                      %(i, self.max_iterations+self.burnin, self.M))
            self.sample_lmbda()
            self.sample_latent_poisson()
            if i >= self.burnin:
                self.data['M'].append(numpy.copy(self.M))
                self.data['lmbda'].append(numpy.copy(self.lmbda))
                self.data['X'].append(numpy.copy(self.X_all))
                self.data['g'].append(numpy.copy(self.g))
                self.data['bm_params'].append(numpy.copy(
                    self.base_measure.params))
                self.data['cov_params'].append(copy(
                    self.cov_params))
                self.data['gp_mu'].append(self.gp_mu)
            self.data['time'].append(time.perf_counter())

    def sample_marks(self):
        """ Samples Polya-Gamma variables (at observed and latent events).

        :return:
        """
        self.marks = numpy.empty(self.N + self.M)
        pgdrawvpar(self.pg, numpy.ones(self.N + self.M), self.g, self.marks)

    def sample_g(self):
        """ Samples GP at Observations and latent events.

        :return:
        """
        Sigma_g_inv = numpy.diag(self.marks) + self.K_inv
        L_inv = numpy.linalg.cholesky(Sigma_g_inv)
        L = numpy.linalg.solve(L_inv, numpy.eye(L_inv.shape[0]))
        Sigma_g = L.T.dot(L)
        u = numpy.empty(self.N + self.M)
        u[:self.N] = .5
        u[self.N:] = -.5
        mu_g = numpy.dot(Sigma_g, u + numpy.sum(self.K_inv*self.gp_mu, axis=0))
        self.g = mu_g + numpy.dot(L.T,numpy.random.randn(len(u)))

    def sample_lmbda(self):
        """ Samples lambda (max. intensity of Poisson process).

        :return:
        """

        alpha = self.N + self.M
        beta = 1.
        shape = 1./beta
        self.lmbda = numpy.random.gamma(alpha, shape)

    def sample_latent_poisson(self):
        """ Samples new latent Poisson process location via thinning.

        :return:
        """

        num_events = numpy.random.poisson(self.lmbda, 1)[0]
        X_unthinned = self.base_measure.sample_density(num_events)
        g_unthinned = self.sample_from_cond_GP(X_unthinned)
        inv_intensity = 1./(1. + numpy.exp(g_unthinned))
        rand_nums = numpy.random.rand(len(X_unthinned))
        thinned_idx = numpy.where(inv_intensity > rand_nums)[0]
        self.M = len(thinned_idx)
        self.X_all = numpy.empty([self.N + self.M, self.D])
        self.X_all[:self.N] = self.X
        self.X_all[self.N:] = X_unthinned[thinned_idx]
        g_X = self.g[:self.N]
        self.g = numpy.empty(self.N + self.M)
        self.g[:self.N] = g_X
        self.g[self.N:] = g_unthinned[thinned_idx]
        self.update_kernels()

    def cov_func(self, x, x_prime, only_diagonal=False, cov_params=None):
        """ Computes the covariance functions (squared exponential) between x
        and x_prime.

        :param x: Contains coordinates for points of x
        :type x: numpy.ndarray [num_points x D]
        :param x_prime: Contains coordinates for points of x_prime
        :type x_prime: numpy.ndarray [num_points_prime x D]
        :param only_diagonal: If true only diagonal is computed (Works only
        if x and x_prime are the same, Default=False)
        :type only_diagonal: bool

        :return: Kernel matrix.
        :rtype: numpy.ndarray [num_points x num_points_prime]

        """

        if cov_params is None:
            theta_1, theta_2 = self.cov_params[0], self.cov_params[1]
        else:
            theta_1, theta_2 = cov_params[0], cov_params[1]

        if only_diagonal:
            return theta_1*numpy.ones(x.shape[0])

        else:
            dx = numpy.subtract(x[:, None], x_prime[None])
            h = .5 * dx ** 2 / (theta_2[None,None]) ** 2
            return theta_1 * numpy.exp(- numpy.sum(h, axis=2))

    def update_kernels(self):
        """ Recalculates all kernel relevant parameters.

        :return:
        """
        self.K = self.cov_func(self.X_all, self.X_all)
        self.K += self.noise * numpy.eye(self.K.shape[0])
        self.L = numpy.linalg.cholesky(self.K)
        self.L_inv = numpy.linalg.solve(self.L, numpy.eye(self.L.shape[0]))
        self.K_inv = self.L_inv.T.dot(self.L_inv)

    def sample_from_cond_GP(self, xprime):
        """ Samples GP conditioned on GP at observations and thinned events.

        :param xprime: Positions where GP should be sampled.
        :type xprime: numpy.ndarray [num_of_points x D]
        :return: GP values
        :rtype: numpy.ndarray [num_of_points]
        """

        k = self.cov_func(self.X_all, xprime)
        mean = self.gp_mu + k.T.dot(self.K_inv.dot(self.g - self.gp_mu))
        kprimeprime = self.cov_func(xprime, xprime)
        Sigma = (kprimeprime - k.T.dot(self.K_inv.dot(k)))
        Sigma += self.noise*numpy.eye(Sigma.shape[0])
        L = numpy.linalg.cholesky(Sigma)
        gprime = mean + numpy.dot(L.T,numpy.random.randn(xprime.shape[0]))
        return gprime

    def predictive_log_likelihood(self, X_test):
        """ Given test set, log test likelihood is sampled.

        :param X_test: Observations of test set.
        :type X_test: numpy.ndarray [num_of_points x D]

        :return: Returns the log mean likelihood
        :rtype: float
        """

        num_test_points = X_test.shape[0]
        test_log_likelihood = numpy.empty(self.max_iterations)
        for iiter in range(self.max_iterations):
            if iiter % 100 == 0:
                print('%d of %d Iterations' %(iiter, self.max_iterations))
            self.base_measure = BaseMeasure(self.D, self.base_measure.type,
                                            self.data['bm_params'][iiter])
            self.integration_points = self.base_measure.sample_density(
                self.num_integration_points)
            X = numpy.vstack([self.integration_points, X_test])
            log_base_measure = numpy.sum(numpy.log(
                self.base_measure.evaluate_density(X_test)))
            self.X_all = self.data['X'][iiter]
            self.g = self.data['g'][iiter]
            self.lmbda = self.data['lmbda'][iiter]
            if self.update_hyperparams:
                self.cov_params = self.data['cov_params'][iiter]
                self.gp_mu = self.data['gp_mu'][-1]
            self.update_kernels()
            g_sample = self.sample_from_cond_GP(X)
            integral = 1./(
                1. + numpy.exp(-g_sample[:self.num_integration_points]))
            log_denominator = num_test_points*numpy.log(numpy.mean(integral))
            log_nominator = log_base_measure - \
                            numpy.sum(numpy.log(1. + numpy.exp(
                                -g_sample[self.num_integration_points:])))
            test_log_likelihood[iiter] = log_nominator - log_denominator

        self._reset()
        mean_test_log_likelihood = numpy.log(numpy.mean(numpy.exp(
                test_log_likelihood - numpy.amax(test_log_likelihood)))) + \
                                   numpy.amax(test_log_likelihood)

        return mean_test_log_likelihood

    def predictive_posterior(self, X_grid, dx):
        """ Samples the mean predictive density.

        :param X_grid: The regular grid on the space, on which density is
        evaluated.
        :type X_grid: numpy.ndarray [num_grid_points x D]
        :param dx: Area of bins.
        :type dx: float

        :return: Sample mean density at grid points
        :rtype: numpy.ndarray [num_grid_points]
        """

        predictive_density = numpy.empty([X_grid.shape[0],
                                      self.max_iterations])
        for iiter in range(self.max_iterations):
            if iiter % 100 == 0:
                print('%d of %d Iterations' %(iiter, self.max_iterations))
            self.base_measure = BaseMeasure(self.D, self.base_measure.type,
                                            self.data['bm_params'][iiter])
            base_measure_grid = self.base_measure.evaluate_density(X_grid)
            self.X_all = self.data['X'][iiter]
            self.g = self.data['g'][iiter]
            self.lmbda = self.data['lmbda'][iiter]
            if self.update_hyperparams:
                self.cov_params = self.data['cov_params'][iiter]
                self.gp_mu = self.data['gp_mu'][iiter]
            self.update_kernels()
            g_pred = self.sample_from_cond_GP(X_grid)
            predictive_density[:,iiter] = self.lmbda * base_measure_grid / (
                1. + numpy.exp(-g_pred))

        predictive_density /= numpy.sum(predictive_density, axis=0)[None]*dx
        mean_density = numpy.mean(predictive_density, axis=1)

        self._reset()

        return mean_density

    def _reset(self):
        """ Resets the sampler to the last sample.

        :return:
        """
        self.M = self.data['M'][-1]
        self.lmbda = self.data['lmbda'][-1]
        self.X_all = self.data['X'][-1]
        self.g = self.data['g'][-1]

        if self.update_hyperparams:
            self.cov_params = self.data['cov_params'][-1]
            self.gp_mu = self.data['gp_mu'][-1]
        if self.update_basemeasure:
            self.base_measure = BaseMeasure(self.D, self.base_measure.type,
                                            self.data['bm_params'][-1])

    def place_integration_points(self):
        """ Places the integration points and updates all related kernels.

        :return:
        """
        self.integration_points = self.base_measure.sample_density(
            self.num_integration_points)

    def sample_basemeasure(self):
        """ Samples base measure parameters with Metropolis Algorithm.

        :return:
        """

        old_log_prob = numpy.sum(numpy.log(self.base_measure.evaluate_density(
            self.X_all)))
        proposed_base_measure = self._propose_new_basemeasure()
        new_log_prob = numpy.sum(
            numpy.log(proposed_base_measure.evaluate_density(self.X_all)))
        rand_log_num = numpy.log(numpy.random.rand(1))

        if rand_log_num < new_log_prob - old_log_prob:
            self.base_measure = proposed_base_measure

    def _propose_new_basemeasure(self):
        """ Proposals for Metropolis sampling.

        :return:
        """

        if self.base_measure.type is 'normal':
            old_log_params = numpy.log(self.base_measure.sigmas)
            new_log_params = old_log_params + .1*numpy.random.randn(self.D)
            new_params = numpy.exp(new_log_params)
        elif self.base_measure.type is 'laplace':
            old_log_params = numpy.log(self.base_measure.scale)
            new_log_params = old_log_params + .1*numpy.random.randn(self.D)
            new_params = numpy.exp(new_log_params)
        if self.base_measure.type is 'standard_t':
            old_log_params = numpy.log(self.base_measure.df - 1.)
            new_log_params = old_log_params + .1 * numpy.random.randn(
                self.D)
            new_params = numpy.exp(new_log_params) + 1.
        return BaseMeasure(self.D, params=new_params,
                           type=self.base_measure.type)

    def sample_GP_mean_prior(self):
        """ Samples mean of the GP prior.

        :return:
        """
        sigma_mu_gp = 1./numpy.sum(self.K_inv)
        mean_mu_gp = sigma_mu_gp*numpy.sum(self.K_inv.dot(self.g))
        self.gp_mu = mean_mu_gp + numpy.sqrt(sigma_mu_gp)*numpy.random.randn(1)

    def sample_kernel_params(self):
        """ Samples kernel parameters with Metropolis sampling.

        :return:
        """

        logp_old = self.compute_kernel_param_prop(self.K_inv, self.L)
        theta1 = numpy.exp(numpy.log(self.cov_params[0]) + .05 * \
                                                       numpy.random.randn(1))
        theta2 = numpy.exp(numpy.log(self.cov_params[1]) + .05 *
                           numpy.random.randn(self.D))
        K = self.cov_func(self.X_all, self.X_all, cov_params=[theta1, theta2])
        K += self.noise * numpy.eye(K.shape[0])
        L = numpy.linalg.cholesky(K)
        L_inv = numpy.linalg.solve(L, numpy.eye(L.shape[0]))
        K_inv = L_inv.T.dot(L_inv)
        logp_new = self.compute_kernel_param_prop(K_inv, L)
        rand_num_accept = numpy.random.rand(1)
        accept_p = numpy.exp(logp_new - logp_old)

        if rand_num_accept < accept_p:
            self.cov_params = [theta1, theta2]
            self.K = K
            self.L = L
            self.L_inv = L_inv
            self.K_inv = K_inv

    def compute_kernel_param_prop(self, K_inv, L):
        """ Computes the log probability (plus constant) given Kernel
        parameters.

        :param K_inv: Inverse kernel matrix.
        :type K_inv: numpy.ndarray [num_of_points x num_of_points]
        :param L: Cholesky decomposition of Kernel matrix.
        :type L: numpy.ndarray [num_of_points x num_of_points]
        :return: log probability plus constant
        :rtype: float
        """

        logp = -.5*self.g.T.dot(K_inv.dot(self.g)) - \
            numpy.sum(numpy.log(L.diagonal()))
        return logp