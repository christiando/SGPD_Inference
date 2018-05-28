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
from scipy.linalg import cholesky
import time

def sample_gp(x, cov_params, noise=1e-5):
    """ Samples a GP.

    :param x: Points, where the GP is sampled at.
    :type x: numpy.ndarray [num_points x dims]
    :param cov_params: Parameters of covariance function.
    :type cov_params: list
    :param noise: Small noise term on covariance diagonal. Default=1e-4
    :type noise: float

    :return: Function values of GP a the points.
    :rtype:  numpy.ndarray [num_points]
    """
    num_points, D = x.shape
    K = cov_func(x, x, cov_params)
    K += noise * numpy.eye(K.shape[0])
    L = numpy.linalg.cholesky(K)
    rand_nums = numpy.random.randn(num_points)
    g = numpy.dot(L.T, rand_nums)

    return g

def sample_from_cond_GP(xprime, g_cond, K_inv, X_cond, cov_params, mu_g,
                        noise=1e-5):
    """ Draws GP conditioned on an GP.

    :param xprime: Poinst at which it should be evaluated.
    :type xprime: numpy.ndarray [num_of_points x D]
    :param g_cond: GP values at conditional locations.
    :type g_cond: numpy.ndarray [num_of_points]
    :param K_inv: Inverse covariance function of GP at conditional locations.
    :type K_inv: numpy.ndarray [num_of_points x num_of_points]
    :param X_cond: Locations of conditional GP values.
    :type X_cond: numpy.ndarray [num_of_points x D]
    :param cov_params: Parameters of covariance kernel. List with first entry
    being the scalar factor and second a D dimensional array with length scales.
    :type cov_params: list
    :param mu_g: GP prior mean.
    :type mu_g: float
    :param noise: Small noise term on covariance diagonal. Default=1e-4
    :type noise: float
    :return: GP value at new locations
    :rtype: numpy.ndarray [num_of_points]
    """

    k = cov_func(X_cond, xprime, cov_params)
    mu = mu_g + k.T.dot(K_inv.dot(g_cond - mu_g)).T
    kprimeprime = cov_func(xprime, xprime, cov_params)
    kprimeprime += noise*numpy.eye(kprimeprime.shape[0])
    Sigma = (kprimeprime - k.T.dot(K_inv.dot(k)))
    L = cholesky(Sigma)

    gprime = mu + numpy.dot(L.T, numpy.random.randn(xprime.shape[0]))
    return gprime

def cov_func(x, x_prime, cov_params):
    """ Computes the covariance functions (squared exponential) between x
    and x_prime.

    :param x: Contains coordinates for points of x
    :type x: numpy.ndarray [num_points x D]
    :param x_prime: Contains coordinates for points of x_prime
    :type x_prime: numpy.ndarray [num_points_prime x D]
    :param cov_params: Parameters of covariance kernel. List with first entry
    being the scalar factor and second a D dimensional array with length scales.
    :type cov_params: list


    :return: Kernel matrix.
    :rtype: numpy.ndarray [num_points x num_points_prime]
    """
    theta_1, theta_2 = cov_params[0], cov_params[1]
    dx = numpy.subtract(x[:, None], x_prime[None])
    h = .5 * dx ** 2 / (theta_2[None,None]) ** 2
    K = theta_1 * numpy.exp(- numpy.sum(h, axis=2))
    return K

def sample_density(num_of_samples, cov_params,
                   base_measure, mu_g=0, grid_points=50,
                   S_borders=None, noise=1e-5):
    """ Samples density observation with the algorithm of [Murray et al, 2009].

    :param num_of_samples: How many sampled shall be generated.
    :type num_of_samples: int
    :param cov_params: Parameters of covariance kernel. List with first entry
    being the scalar factor and second a D dimensional array with length scales.
    :type cov_params: list
    :param base_measure: Base measure of density.
    :type BaseMeasure:
    :param mu_g: GP prior mean. Default=0.
    :type mu_g: float
    :param grid_points: Points per dimension, where density should be
    evaluated (for visualisation purpose). Default=50.
    :type grid_points: int
    :param S_borders: Limits of domain on which density should be evaluated.
    Default=None.
    :type S_borders: numpy.ndarray [D x 2]
    :param noise: Small noise term on covariance diagonal. Default=1e-4
    :type noise: float

    :return: Samples, locations that where proposed, GP at proposed
    locations, locations of grid, GP at grid.
    :rtype: list
    """

    D = len(cov_params[1])
    num_of_accepted_samples = 0
    X_samples = numpy.empty((0,D))
    X_prev = numpy.empty((0, D))
    g_prev = numpy.empty((0,1))
    x_proposed = base_measure.sample_density(1)
    var = cov_func(x_proposed, x_proposed, cov_params)
    g_new = mu_g + numpy.sqrt(var)*numpy.random.randn(1)
    rand_num = numpy.random.rand(1)
    if rand_num < 1./(1 + numpy.exp(-g_new)):
        num_of_accepted_samples += 1
        X_samples = numpy.concatenate([X_samples, x_proposed])
    X_prev = numpy.concatenate([X_prev, x_proposed])
    g_prev = numpy.concatenate([g_prev, g_new])
    K = cov_func(X_prev, X_prev, cov_params)
    K += noise * numpy.eye(K.shape[0])
    L = cholesky(K)
    L_inv = numpy.linalg.solve(L, numpy.eye(L.shape[0]))
    K_inv = L_inv.dot(L_inv.T)
    t = time.perf_counter()
    while num_of_accepted_samples < num_of_samples:
        t_new = time.perf_counter()

        if t_new - t > 10:
            t = t_new
            print('%d of %d points sampled.' %(num_of_accepted_samples,
                                               num_of_samples))
        x_proposed = base_measure.sample_density(1)
        g_new = sample_from_cond_GP(x_proposed, g_prev, K_inv, X_prev,
                                    cov_params, mu_g)
        rand_num = numpy.random.rand(1)
        if rand_num < 1. / (1. + numpy.exp(-g_new)):
            num_of_accepted_samples += 1
            X_samples = numpy.concatenate([X_samples, x_proposed])
        X_prev = numpy.concatenate([X_prev, x_proposed])
        g_prev = numpy.concatenate([g_prev, g_new])
        K = cov_func(X_prev, X_prev, cov_params)
        K += noise * numpy.eye(K.shape[0])
        L = cholesky(K)
        L_inv = numpy.linalg.solve(L, numpy.eye(L.shape[0]))
        K_inv = L_inv.dot(L_inv.T)


    if grid_points > 0 and S_borders is not None:
        X_grid = numpy.empty([grid_points, D])
        for di in range(D):
            X_grid[:, di] = numpy.linspace(S_borders[di, 0], S_borders[di, 1],
                                           grid_points)
        X_mesh = numpy.meshgrid(*X_grid.T.tolist())
        X_mesh = numpy.array(X_mesh).reshape([D, -1]).T
        g_grid = sample_from_cond_GP(X_mesh,g_prev,K_inv,X_prev,cov_params,
                                     mu_g)
    else:
        g_grid = None
        X_grid = None

    return X_samples, X_prev, g_prev, X_grid, g_grid