"""
Copyright (C) 2018, Christian Donner

This file is part of SGPD_Inference.

SGPD_Inference is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SGPD_Inference is distributed in the hope that it will be useful,
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
from scipy.special import digamma, gammaln
from scipy.integrate import quadrature
from .basemeasures import BaseMeasure
import time
from sklearn.cluster import KMeans
from scipy.linalg import solve_triangular

class VI_SGPD():
    """ Variational inference for SGPD model.
    """

    def __init__(self, X, cov_params, base_measure, num_inducing_points=100,
                 conv_crit=1e-3, num_integration_points=1000, output=False,
                 update_hyperparams=True, update_basemeasure=True, gp_mu = 0,
                 epsilon=1e-2):
        """ Initialises class of variational Bayes.

        :param X: Data.
        :type X: numpy.ndarray [instances x features]
        :param cov_params: Kernel hyperparameters. List with first entry the
        prefactor and second a D-dimensional array with length scales.
        :type cov_params: list
        :param base_measure:
        :type base_measure: BaseMeasure
        :param num_inducing_points: Num. of inducing points used for sparse GP.
        Default=100
        :type num_inducing_points: int
        :param conv_crit: Convergence criterion (absolute). Default=1e-3
        :type conv_crit: float
        :param num_integration_points: How many samples are used for
        importance sampling. Default=1000.
        :type num_integration_points: int
        :param output: Whether it prints messages during fit or not.
        Default=False
        :type output: bool
        :param update_hyperparams: Whether GP hyperparameters (kernel+mean)
        are optimised. Default=True.
        :type update_hyperparams: bool
        :param update_basemeasure: Whether base measure parameters are
        optimised. Default=True.
        :type update_basemeasure: bool
        :param gp_mu: Initial value of mean GP prior.
        :type gp_mu: float
        :param epsilon: Step size of ADAM gradient.
        :type epsilon: float
        """

        self.D = X.shape[1]
        self.cov_params = cov_params
        self.noise = 1e-4
        self.num_integration_points = num_integration_points
        self.num_inducing_points = num_inducing_points  # must be power of D
        self.X = X
        self.gp_mu = gp_mu
        self.mu_g_X = self.gp_mu*numpy.ones(X.shape[0])
        self.mu_g2_X = 1e-10*numpy.ones(X.shape[0])
        self.base_measure = base_measure
        self.place_inducing_points()
        self.Ks = self.cov_func(self.induced_points, self.induced_points)
        self.Ks += self.noise * numpy.eye(self.Ks.shape[0])
        L = numpy.linalg.cholesky(self.Ks)
        L_inv = solve_triangular(L, numpy.eye(L.shape[0]), lower=True,
                                 check_finite=False)
        self.Ks_inv = L_inv.T.dot(L_inv)
        self.logdet_Ks = 2. * numpy.sum(numpy.log(L.diagonal()))
        self.place_integration_points()
        self.mu_g_int_points = self.gp_mu*numpy.ones(num_integration_points)
        self.mu_g2_int_points = 1e-10 * numpy.ones(num_integration_points)
        self.ks_X = self.cov_func(self.induced_points, self.X)
        self.LB_list = []
        self.times = []

        self.kappa_X = self.Ks_inv.dot(self.ks_X)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)
        self.epsilon = epsilon/self.X.shape[0]
        self.lmbda_q1 = self.X.shape[0] / 1.
        self.log_lmbda_q1 = digamma(self.X.shape[0])
        self.alpha_q1 = self.X.shape[0]
        self.beta_q1 = 1.
        self.convergence = numpy.inf
        self.conv_crit = conv_crit
        self.num_iterations = 0
        self.output = output
        self.update_hyperparams = update_hyperparams
        self.update_basemeasure = update_basemeasure

        # ADAM parameters
        self.beta1_adam = .9
        self.beta2_adam = .99
        self.epsilon_adam = 1e-5
        self.m_hyper_adam = numpy.zeros(self.D + 2)
        self.v_hyper_adam = numpy.zeros(self.D + 2)
        self.m_bm_adam = numpy.zeros(self.D)
        self.v_bm_adam = numpy.zeros(self.D)

    def place_inducing_points(self):
        """ Places the inducing points (sparse GP). Half according to base
        measure half to kmeans.

        :return:
        """
        self.induced_points = numpy.zeros([self.num_inducing_points, self.D])
        random_inducing_points = int(.5*self.num_inducing_points)
        if self.num_inducing_points - random_inducing_points > self.X.shape[0]:
            random_inducing_points = self.num_inducing_points - self.X.shape[0]
            self.induced_points[:random_inducing_points] = \
                self.base_measure.sample_density(random_inducing_points)
            self.induced_points[random_inducing_points:] = self.X
        else:
            self.induced_points[:random_inducing_points] = \
                self.base_measure.sample_density(random_inducing_points)
            kmeans_inducing_points = self.num_inducing_points - \
                                     random_inducing_points
            kmeans = KMeans(n_clusters=kmeans_inducing_points)
            kmeans.fit(self.X)
            self.induced_points[random_inducing_points:] = kmeans.cluster_centers_

    def run(self):
        """ Main function that runs the fit.

        :return:
        """
        self.times.append(time.perf_counter())
        self.calculate_PG_expectations()
        self.calculate_posterior_intensity()
        converged = False
        while not converged:
            self.num_iterations += 1
            self.calculate_postrior_GP()
            self.update_lmbda()
            if self.update_basemeasure:
                self.place_integration_points()
            self.update_predictive_posterior()
            self.calculate_PG_expectations()
            self.calculate_posterior_intensity()
            if self.update_hyperparams:
                self.update_hyperparameters()
            if self.update_basemeasure:
                self.update_bm()
            self.LB_list.append(self.calculate_lower_bound())
            if self.update_basemeasure and self.num_iterations > 50:
                self.convergence = numpy.mean(self.LB_list[-50:]) - \
                                   numpy.mean(self.LB_list[-100:-50])
                converged = self.convergence < self.conv_crit
            elif not self.update_basemeasure and self.num_iterations > 1:
                self.convergence = self.LB_list[-1] - self.LB_list[-2]
                converged = self.convergence < self.conv_crit
            self.times.append(time.perf_counter())
            if self.output and self.num_iterations % 100 == 0:
                self.print_info()

    def print_info(self):
        """ Prints info.

        :return:
        """
        print((' +-----------------+ ' +
              '\n |  Iteration %4d |' +
              '\n |  Conv. = %.4f |' +
              '\n +-----------------+') %(self.num_iterations,
                                           self.convergence))

    def place_integration_points(self):
        """ Places the integration points and updates all related kernels.

        :return:
        """
        self.integration_points = self.base_measure.sample_density(
            self.num_integration_points)
        self.ks_int_points = self.cov_func(self.induced_points,
                                           self.integration_points)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)

    def calculate_posterior_intensity(self):
        """ The rate of the latent Poisson process is updated.

        :return:
        """

        self.lmbda_q2 = .5*numpy.exp(-.5 * self.mu_g_int_points +
                                     self.log_lmbda_q1) / \
                        numpy.cosh(.5*self.c_int_points)
        self.log_lmbda_q2 = \
            numpy.log(self.lmbda_q2) + \
            numpy.log(self.base_measure.evaluate_density(
            self.integration_points))

    def calculate_PG_expectations(self):
        """ The Polya-Gamma posterior is updated at observations and
        integration points.

        :return:
        """

        self.c_X = numpy.sqrt(self.mu_g2_X)
        self.mu_omega_X = .5/self.c_X*numpy.tanh(
            .5*self.c_X)
        self.c_int_points = numpy.sqrt(self.mu_g2_int_points)
        self.mu_omega_int_points = .5/self.c_int_points \
            * numpy.tanh(.5*self.c_int_points)

    def calculate_predictive_posterior_intensity(self, X_prime):
        """ Predictive Poisson intensity at new poinst

        :param X_prime: Points where intensity is evaluated.
        :type X_prime: numpy.ndarray [num_of_points x D]
        :return: Intensity at X_prime.
        :rtype: numpy.ndarray [num_of_points]
        """
        mu_g, var_g = self.predictive_posterior_GP(X_prime)
        mu_g = mu_g
        mu_g2 = var_g + mu_g ** 2
        c = numpy.sqrt(mu_g2)
        pred_lmbda_q2 = .5 * numpy.exp(-.5 * mu_g + self.log_lmbda_q1) / \
                        numpy.cosh(.5 * c)
        return pred_lmbda_q2

    def calculate_postrior_GP(self):
        """ Updates the sparse GP posterior.

        :return:
        """

        A_int_points = self.lmbda_q2 * self.mu_omega_int_points
        A_X = self.mu_omega_X
        kAk = self.kappa_X.dot(A_X[:,numpy.newaxis] * self.kappa_X.T) + \
                self.kappa_int_points.dot(A_int_points[:,numpy.newaxis] *
                                       self.kappa_int_points.T) \
                / self.num_integration_points
        self.Sigma_g_s_inv =  kAk + self.Ks_inv
        L_inv = numpy.linalg.cholesky(self.Sigma_g_s_inv)
        L = solve_triangular(L_inv, numpy.eye(L_inv.shape[0]), lower=True,
                             check_finite=False)
        self.Sigma_g_s = L.T.dot(L)
        self.logdet_Sigma_g_s = 2*numpy.sum(numpy.log(L.diagonal()))
        b_int_points = (-.5 - (self.gp_mu -
                        self.gp_mu*numpy.sum(self.kappa_int_points,axis=0)
                               )*self.mu_omega_int_points) * self.lmbda_q2
        b_X = .5 - self.mu_omega_X*(self.gp_mu -
                                    self.gp_mu*numpy.sum(self.kappa_X, axis=0))
        kb = self.kappa_X.dot(b_X) + self.kappa_int_points.dot(b_int_points) /\
                             self.num_integration_points
        self.mu_g_s = self.Sigma_g_s.dot(
            kb + self.gp_mu*numpy.sum(self.Ks_inv, axis=0))

    def predictive_posterior_GP(self, x_prime, points=None):
        """ Computes the predictive posterior for given points.

        :param x_prime: Points, which should be predicted for.
        :type x_prime: numpy.ndarray [num_of_points x D]
        :param points: If 'int_points' or 'X' posterior for integration
        points or observation points is calculated, respectively (Default=None).
        :type points: str
        :returns: mean of predictive posterior and variance of predictive
        posterior
        :rtype: list
        """
        if points is None:
            ks_x_prime = self.cov_func(self.induced_points, x_prime)
            kappa = self.Ks_inv.dot(ks_x_prime)
        elif points is 'int_points':
            ks_x_prime = self.ks_int_points
            kappa = self.kappa_int_points
        elif points is 'X':
            ks_x_prime = self.ks_X
            kappa = self.kappa_X

        mu_g_x_prime = self.gp_mu + kappa.T.dot(self.mu_g_s - self.gp_mu)
        K_xx = self.cov_func(x_prime, x_prime, only_diagonal=True)
        var_g_x_prime = K_xx - numpy.sum(
            kappa*(ks_x_prime - kappa.T.dot(self.Sigma_g_s).T),axis=0)
        return mu_g_x_prime, var_g_x_prime

    def cov_func(self, x, x_prime, only_diagonal=False):
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

        theta_1, theta_2 = self.cov_params[0], self.cov_params[1]
        if only_diagonal:
            return theta_1*numpy.ones(x.shape[0])

        else:
            h = numpy.zeros([x.shape[0],x_prime.shape[0]])
            for idim in range(self.D):
                dx = numpy.subtract(x[:, None,idim], x_prime[None,:,idim])
                h += .5*dx ** 2 / (theta_2[idim]) ** 2
            return theta_1 * numpy.exp(-h)

    def update_lmbda(self):
        """ Updates the posterior for the maximal intensity.
        """
        self.alpha_q1 = self.X.shape[0] + numpy.sum(
            self.lmbda_q2)/self.num_integration_points
        self.beta_q1 = 1.
        self.lmbda_q1 = self.alpha_q1 / self.beta_q1
        self.log_lmbda_q1 = digamma(self.alpha_q1) - \
                            numpy.log(self.beta_q1)

    def update_kernels(self):
        """ Updates all kernels (for induced, observed and integration points).

        :return:
        """
        self.ks_int_points = self.cov_func(self.induced_points,
                                           self.integration_points)
        self.ks_X = self.cov_func(self.induced_points, self.X)
        self.Ks = self.cov_func(self.induced_points, self.induced_points)
        self.Ks += self.noise * numpy.eye(self.Ks.shape[0])
        L = numpy.linalg.cholesky(self.Ks)
        L_inv = solve_triangular(L, numpy.eye(L.shape[0]), lower=True,
                                 check_finite=False)
        self.Ks_inv = L_inv.T.dot(L_inv)
        self.logdet_Ks = 2. * numpy.sum(numpy.log(L.diagonal()))
        self.kappa_X = self.Ks_inv.dot(self.ks_X)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)

    def update_bm(self):
        """ Does one ADAM step for parameters of base measure.

        :return:
        """

        dL_dbm = self.calculate_bm_derivative()
        old_params_bm = self.base_measure.params
        dL_dlogbm = old_params_bm*dL_dbm
        self.m_bm_adam = self.beta1_adam * self.m_bm_adam + \
                               (1. - self.beta1_adam) * dL_dlogbm
        self.v_bm_adam = self.beta2_adam * self.v_bm_adam + \
                               (1. - self.beta2_adam) * dL_dlogbm ** 2
        m_hat = self.m_bm_adam / (1. - self.beta1_adam)
        v_hat = self.v_bm_adam / (1. - self.beta2_adam)
        log_params_new = numpy.log(old_params_bm) + \
                         self.epsilon*m_hat/(numpy.sqrt(v_hat) +
                                             self.epsilon_adam)
        params_new = numpy.exp(log_params_new)
        self.base_measure = BaseMeasure(self.D, self.base_measure.type,
                                        params_new)

    def calculate_bm_derivative(self):
        """ Computes the derivative of variational lower bound wrt base
        measure parameters.

        :return: Derivative
        :rtype: numpy.ndarray [num_of_parameters]
        """

        deriv_X = self.base_measure.log_derivative(self.X)
        deriv_int = self.base_measure.log_derivative(self.integration_points)
        deriv_bm = numpy.sum(deriv_X,axis=0) + numpy.mean(
            deriv_int*self.lmbda_q2[:,None], axis=0)

        return deriv_bm

    def calculate_hyperparam_derivative(self):
        """ Calculates the derivative of the lower bound with respect to the
        kernel hyperparameters.

        :return: Derivative
        :rtype: numpy.ndarray [num_of_parameters]
        """
        Sigma_s_mugmug = self.Sigma_g_s + numpy.outer(self.mu_g_s, self.mu_g_s)

        theta1, theta2 = self.cov_params[0], numpy.copy(
            self.cov_params[1])
        dks_X = numpy.empty([self.ks_X.shape[0], self.ks_X.shape[1],
                             1 + theta2.shape[0]])
        dks_int_points = numpy.empty(
            [self.ks_int_points.shape[0], self.ks_int_points.shape[1],
             1 + theta2.shape[0]])
        dKs = numpy.empty([self.Ks.shape[0], self.Ks.shape[1],
                           1 + theta2.shape[0]])
        dKss = numpy.zeros([1 + theta2.shape[0]])
        dKss[0] = 1.

        # kernel derivatives wrt theta1
        dks_X[:, :, 0] = self.ks_X / theta1
        dks_int_points[:, :, 0] = self.ks_int_points / theta1
        dKs[:, :, 0] = self.Ks / theta1
        # kernel derivatives wrt theta2
        dx = numpy.subtract(self.induced_points[:, None],
                            self.X[None])
        dks_X[:, :, 1:] = self.ks_X[:, :, None] * (dx ** 2) / \
                          (theta2[None, None] ** 3)
        dx = numpy.subtract(self.induced_points[:, None],
                            self.integration_points[None])
        dks_int_points[:, :, 1:] = self.ks_int_points[:, :, None] * \
                                   (dx ** 2) / (theta2[None, None] ** 3)
        dx = numpy.subtract(self.induced_points[:, None],
                            self.induced_points[None])
        dKs[:, :, 1:] = self.Ks[:, :, None] * (dx ** 2) / (theta2[None,
                                                                  None] ** 3)
        dL_dtheta = numpy.zeros(1 + len(theta2))

        for itheta in range(1 + len(theta2)):
            dKs_inv = -self.Ks_inv.dot(dKs[:, :, itheta].dot(self.Ks_inv))

            dkappa_X = self.Ks_inv.dot(dks_X[:, :, itheta]) + dKs_inv.dot(
                self.ks_X)
            dkappa_int_points = self.Ks_inv.dot(
                dks_int_points[:, :, itheta]) + dKs_inv.dot(
                self.ks_int_points)

            dKtilde_X = dKss[itheta] - numpy.sum(
                dks_X[:, :, itheta] * self.kappa_X, axis=0) - numpy.sum(
                self.ks_X * dkappa_X, axis=0)
            dKtilde_int_points = dKss[itheta] - numpy.sum(
                dks_int_points[:, :, itheta] * self.kappa_int_points,
                axis=0) - numpy.sum(self.ks_int_points * dkappa_int_points,
                                    axis=0)

            dg1_X = (self.mu_g_s - self.gp_mu).dot(dkappa_X)
            dg1_int_points = (self.mu_g_s - self.gp_mu).dot(dkappa_int_points)

            dg2_X = dKtilde_X*self.mu_omega_X
            dg2_X += 2.*numpy.sum(self.kappa_X * self.Sigma_g_s.dot(dkappa_X),
                axis=0)*self.mu_omega_X
            dg2_X +=  2.*self.mu_g_X*dg1_X * self.mu_omega_X

            dg2_int_points = dKtilde_int_points * self.mu_omega_int_points
            dg2_int_points += 2. * numpy.sum(self.kappa_int_points *
                            self.Sigma_g_s.dot(dkappa_int_points), axis=0) * \
                              self.mu_omega_int_points
            dg2_int_points += 2.*self.mu_g_int_points*dg1_int_points * \
                              self.mu_omega_int_points
            dL_dtheta[itheta] += .5 * (numpy.sum(dg1_X) - numpy.sum(dg2_X))
            dL_dtheta[itheta] += .5 * numpy.dot(
                - dg1_int_points - dg2_int_points,
                self.lmbda_q2) / self.num_integration_points
            dL_dtheta[itheta] -= .5 * numpy.trace(self.Ks_inv.dot(
                dKs[:, :, itheta])) # (correct)
            dL_dtheta[itheta] -= .5 * numpy.trace(dKs_inv.dot(Sigma_s_mugmug))
                       # (correct)
            dL_dtheta[itheta] += numpy.sum(self.mu_g_s.dot(dKs_inv))*self.gp_mu
            dL_dtheta[itheta] -= .5*numpy.sum(dKs_inv)*self.gp_mu**2

        return dL_dtheta

    def calculate_lower_bound(self):
        """ Calculates the variational lower bound for current posterior.

        :return: Variational lower bound.
        :rtype: float
        """
        Sigma_s_mugmug = self.Sigma_g_s + numpy.outer(self.mu_g_s, self.mu_g_s)
        f_int_points = .5*(- self.mu_g_int_points -
                           self.mu_g2_int_points*self.mu_omega_int_points) -\
                           numpy.log(2)
        log_density_int_points = numpy.log(self.base_measure.evaluate_density(
            self.integration_points))
        integrand = f_int_points + log_density_int_points - \
                    self.log_lmbda_q2 - numpy.log(numpy.cosh(
                    .5*self.c_int_points)) \
                    + self.log_lmbda_q1 + \
                    .5*self.c_int_points**2*self.mu_omega_int_points + 1.
        f_X = .5 * (self.mu_g_X - self.mu_g2_X * self.mu_omega_X) - \
                    numpy.log(2)
        summand = f_X + self.log_lmbda_q1 - numpy.log(numpy.cosh(
            .5*self.c_X)) + .5*self.c_X**2*self.mu_omega_X + \
                  numpy.log(self.base_measure.evaluate_density(self.X))
        L = integrand.dot(self.lmbda_q2)/self.num_integration_points
        L += numpy.sum(summand)
        L -= .5*numpy.trace(self.Ks_inv.dot(Sigma_s_mugmug))
        L += numpy.sum(self.Ks_inv.dot(self.mu_g_s))*self.gp_mu
        L -= .5*numpy.sum(self.Ks_inv)*self.gp_mu**2
        L -= .5*self.logdet_Ks
        L += .5*self.logdet_Sigma_g_s + .5*self.num_inducing_points
        L +=  - self.log_lmbda_q1
        L += - self.alpha_q1*self.log_lmbda_q1 + gammaln(self.alpha_q1)

        return L

    def calculate_hyperparam_derivative_mu(self):
        """ Calculates the derivate of variational lower bound wrt GP prior
        mean.

        :return: Derivative
        :rtype: float
        """
        dg_dmu_int_points = 1. - numpy.sum(self.kappa_int_points, axis=0)
        dg_dmu_X = 1. - numpy.sum(self.kappa_X, axis=0)
        dg2_dmu_int_points = 2.*self.mu_g_int_points*dg_dmu_int_points
        dg2_dmu_X = 2. * self.mu_g_X * dg_dmu_X
        dL_dmu = .5*numpy.sum(dg_dmu_X)
        dL_dmu -= .5 * numpy.sum(dg_dmu_int_points*self.lmbda_q2) \
                  / self.num_integration_points
        dL_dmu -= .5*numpy.sum(dg2_dmu_X *self.mu_omega_X)
        dL_dmu -= .5 * numpy.sum(dg2_dmu_int_points * self.mu_omega_int_points *
                                 self.lmbda_q2) / self.num_integration_points
        dL_dmu += numpy.sum(self.mu_g_s.dot(self.Ks_inv))
        dL_dmu -= numpy.sum(self.Ks_inv)*self.gp_mu

        return dL_dmu

    def update_hyperparameters(self):
        """ ADAM step for kernel hyperparameters and GP prior mean.

        :return:
        """
        dL_dtheta = self.calculate_hyperparam_derivative()
        dL_dmu = self.calculate_hyperparam_derivative_mu()
        logtheta1, logtheta2 = numpy.log(self.cov_params[0]), \
                               numpy.log(self.cov_params[1])
        dL_dlogtheta1 = dL_dtheta[0] * numpy.exp(logtheta1)
        dL_dlogtheta2 = dL_dtheta[1:] * numpy.exp(logtheta2)
        self.m_hyper_adam[0] = self.beta1_adam*self.m_hyper_adam[0] + \
                               (1. - self.beta1_adam)*dL_dmu
        self.v_hyper_adam[0] = self.beta2_adam*self.v_hyper_adam[0] + \
                               (1. - self.beta2_adam)*dL_dmu**2
        self.m_hyper_adam[1] = self.beta1_adam * self.m_hyper_adam[1] + \
                               (1. - self.beta1_adam) * dL_dlogtheta1
        self.v_hyper_adam[1] = self.beta2_adam * self.v_hyper_adam[1] + \
                               (1. - self.beta2_adam) * dL_dlogtheta1 ** 2
        self.m_hyper_adam[2:] = self.beta1_adam * self.m_hyper_adam[2:] + \
                               (1. - self.beta1_adam) * dL_dlogtheta2
        self.v_hyper_adam[2:] = self.beta2_adam * self.v_hyper_adam[2:] + \
                               (1. - self.beta2_adam) * dL_dlogtheta2 ** 2
        m_hat = self.m_hyper_adam/(1. - self.beta1_adam)
        v_hat = self.v_hyper_adam / (1. - self.beta2_adam)
        self.gp_mu += self.epsilon*m_hat[0]/(numpy.sqrt(v_hat[0]) +
                                             self.epsilon_adam)
        logtheta1 += self.epsilon*m_hat[1]/(numpy.sqrt(v_hat[1]) +
                                             self.epsilon_adam)
        logtheta2 += self.epsilon * m_hat[2:] / (numpy.sqrt(v_hat[2:]) +
                                                self.epsilon_adam)
        self.cov_params[0] = numpy.exp(logtheta1)
        self.cov_params[1] = numpy.exp(logtheta2)
        self.update_kernels()
        self.update_predictive_posterior()

    def update_predictive_posterior(self, only_int_points=False):
        """ Updates the function g (mean & variance) at each point (observed
        and points for monte carlo integral)

        :param only_int_points: If True it only updates the integration points
        (Default=False)
        :type only_int_points: bool
        """

        if not only_int_points:
            mu_g_X, var_g_X = self.predictive_posterior_GP(
                self.X, points='X')
            self.mu_g_X = mu_g_X
            self.mu_g2_X = var_g_X + mu_g_X ** 2
        mu_g_int_points, var_g_int_points = self.predictive_posterior_GP(
            self.integration_points, points='int_points')
        self.mu_g_int_points = mu_g_int_points
        self.mu_g2_int_points = var_g_int_points + mu_g_int_points ** 2

    def predictive_density_function(self, X_grid, dx):
        """ Calculates the mean predictive density via numerical integration.

        :param X_grid: The regular grid on the space, on which density is
        evaluated.
        :type X_grid: numpy.ndarray [num_grid_points x D]
        :param dx: Area of bins.
        :type dx: float

        :return: Sample mean density at grid points
        :rtype: numpy.ndarray [num_grid_points]
        """

        num_preds = X_grid.shape[0]
        mu_pred, var_pred = self.predictive_posterior_GP(X_grid)
        base_measure_pred = self.base_measure.evaluate_density(X_grid)


        lmbda_pred = numpy.empty(num_preds)
        for ipred in range(num_preds):
            mu, std = mu_pred[ipred], numpy.sqrt(var_pred[ipred])
            func = lambda g_pred: 1. / (1. + numpy.exp(-g_pred)) * \
                                  numpy.exp(-.5*(g_pred - mu)**2 / std**2) / \
                                  numpy.sqrt(2.*numpy.pi*std**2)
            a, b = mu - 10.*std, mu + 10.*std
            lmbda_pred[ipred] = base_measure_pred[
                                    ipred]\
                                * quadrature(func, a, b, maxiter=100,
                                                              )[0]
        lmbda_pred /= numpy.sum(lmbda_pred, axis=0)[None] * dx
        return lmbda_pred

    def predictive_log_likelihood(self, X_test, num_samples=int(2e3),
                                  return_integral_stats=False):
        """ Given test set, log test likelihood is sampled.

        :param X_test: Observations of test set.
        :type X_test: numpy.ndarray [num_of_points x D]
        :param num_samples: Number of samples. Default=2000.
        :type num_samples: int
        :param return_integral_stats: If True returns also mean and variance
        of each sampled importance sampling. (For checking, whether
        importance sampling was good).
        :type return_integral_stats: bool

        :return: Returns the log mean likelihood and if indicated mean and
        variance of importance sampling
        :rtype: float or list
        """

        num_test_points = X_test.shape[0]
        num_samples = int(num_samples)
        log_base_measure = numpy.sum(numpy.log(
            self.base_measure.evaluate_density(X_test)))
        test_log_likelihood = numpy.empty(num_samples)
        X = numpy.vstack([self.integration_points, X_test])
        K = self.cov_func(X,X)
        K += self.noise*numpy.eye(K.shape[0])
        kx = self.cov_func(X,self.induced_points)
        kappa = kx.dot(self.Ks_inv)
        Sigma_post = K - kappa.dot(kx.T - self.Sigma_g_s.dot(kappa.T))
        mu_post = self.gp_mu + kappa.dot(self.mu_g_s - self.gp_mu)
        L_post = numpy.linalg.cholesky(Sigma_post)

        num_points = X.shape[0]
        mean_integral = numpy.empty(num_samples)
        var_integral = numpy.empty(num_samples)

        for isample in range(num_samples):
            if isample % 100 == 0:
                print('%d of %d Iterations' %(isample, num_samples))

            rand_nums = numpy.random.randn(num_points)
            g_sample = mu_post + L_post.dot(rand_nums)
            integral = 1. / (
                1. + numpy.exp(-g_sample[:self.num_integration_points]))
            mean_integral[isample] = numpy.mean(integral)
            var_integral[isample] =  (numpy.mean(integral**2) - numpy.mean(
                integral)**2)/self.num_integration_points
            log_denominator = num_test_points * numpy.log(numpy.mean(integral))
            log_nominator = log_base_measure - \
                            numpy.sum(numpy.log(1. + numpy.exp(
                                -g_sample[self.num_integration_points:])))
            test_log_likelihood[isample] = log_nominator - log_denominator

        mean_test_log_likelihood = numpy.log(numpy.mean(numpy.exp(
            test_log_likelihood - numpy.amax(test_log_likelihood)))) + \
                                   numpy.amax(test_log_likelihood)

        if return_integral_stats:
            return mean_test_log_likelihood, mean_integral, var_integral
        else:
            return mean_test_log_likelihood