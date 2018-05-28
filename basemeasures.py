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
from scipy.special import gamma, digamma
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

class BaseMeasure():
    """ Class for base measure of GSPD model.
    """
    def __init__(self, D, type='normal', params=None):
        """ Initialises base measure.

        :param D: Dimension of data space
        :type D: int
        :param type: Type of base measure. So far implemented uniform (
        'uniform'), normal ('normal'), laplace ('laplace'), student t (
        'standard_t'), and Gaussian mixture model ('gmm'). Default is 'normal'.
        Default='uniform'.
        :type type: str
        :param params: Parameters for base measure. Depends on type. If None
        standard parameters are used. Default is None.
        :type params: ???
        """

        self.D = D
        self.params = params
        self.type = type

        if type is 'uniform':
            self._init_uniform()
        elif type is 'normal':
            self._init_normal()
        elif type is 'laplace':
            self._init_laplace()
        elif type is 'standard_t':
            self._init_standard_t()
        elif type is 'gmm':
            self._init_gmm()

    def _init_uniform(self):
        """ Initialises uniform base measure on unit square. Has no parameters.
        """

        if self.params is None:
            self.params = numpy.empty([self.D, 2])
            self.params[:,0] = 0.
            self.params[:,1] = 1.

        self.width = self.params[:,1] - self.params[:,0]
        self.vol = numpy.prod(self.width)
        self.evaluate_density = lambda x: 1./self.vol*numpy.ones(x.shape[0])
        self.sample_density = lambda N: \
            self.width[None]*numpy.random.rand(N, self.D) + self.params[:,0][
                                                           None,:]
        self.integral = 1.
        self.max = 1./self.vol
        self.log_derivative = lambda x: numpy.zeros(x.shape)

    def _init_normal(self):
        """ Initialises normal base measure with zero mean diagonal covariance
        matrix. params is a D-dimensional numpy.ndarray with the variances. If
        params=None, it will be a standard normal distribution. Parameters
        can be optimised.
        """

        if self.params is None:
            self.params = numpy.ones(self.D)

        self.sigmas = self.params
        self.std = numpy.sqrt(self.sigmas)

        self.Z = numpy.sqrt((2.*numpy.pi)**self.D)*numpy.prod(self.std)
        self.evaluate_density = lambda x:numpy.exp(-.5*
        numpy.sum(x ** 2 / self.sigmas[None], axis=1))/self.Z

        self.sample_density = lambda N: self.std[None]*numpy.random.randn(
            N, self.D)
        self.log_derivative = lambda x: -.5/self.sigmas + .5*x**2/self.sigmas[
            None]**2

    def _init_laplace(self):
        """ Initialises Laplace base measure with zero mean. params is a
        D-dimensional numpy.ndarray with the scales. If params=None scales
        will be one. Parameters can be optimised.
        """

        if self.params is None:
            self.params = numpy.ones(self.D)

        self.loc = numpy.zeros(self.D)
        self.scale = self.params
        self.sample_density = lambda N: numpy.random.laplace(self.loc,
                                                             self.scale,
                                                             (N, self.D))
        self.Z = 2.*self.scale
        self.evaluate_density = lambda x: numpy.prod(
            numpy.exp(-numpy.absolute(x - self.loc[None])/self.scale[None])
            / self.Z[None], axis=1)
        self.log_derivative = lambda x: - 1./self.scale + \
                                        numpy.absolute(x)/self.scale[None]**2

    def _init_standard_t(self):
        """ Initialises Student-t base measure with zero mean. params is a
        D-dimensional numpy.ndarray with the degrees of freedom. If
        params=None degrees of freedom will be 2. Parameters can be optimised.
        """

        if self.params is None:
            self.params = 2.*numpy.ones(self.D)

        self.df = self.params
        self.sample_density = lambda N: numpy.random.standard_t(self.df,
                                                                (N, self.D))
        self.Z = numpy.sqrt(self.df*numpy.pi)*gamma(
            .5*self.df)/gamma(.5*(self.df + 1.))
        self.evaluate_density = lambda x: numpy.prod((1. + x**2/self.df[
            None])**(-(self.df[None] + 1.)/2.)/self.Z[None], axis=1)
        self.log_derivative = lambda x: .5*(digamma(.5*(self.df + 1.)) -
                                        1./self.df - digamma(.5*self.df) -
                                        numpy.log(1. + x**2/self.df) +
                                        (self.df + 1.)/(1. +
                                                x**2/self.df)*x**2/self.df**2)

    def _init_gmm(self):
        """ Initialises Gaussian mixture model base measure. params must be
        the train data or an already trained GMM base measure. If it is data
        number of components is optimised by grid search. Parameters cannot
        be optimised.
        """
        self.min_nc = 2
        self.max_nc = 50
        if isinstance(self.params, numpy.ndarray):
            X = self.params
            nc_range = numpy.arange(self.min_nc, self.max_nc)

            grid = GridSearchCV(
                GaussianMixture(covariance_type='full', reg_covar=1e-5,
                                max_iter=100, n_init=5),
                {'n_components': nc_range},
                cv=5)
            grid.fit(X)
            optimal_nc = grid.best_params_['n_components']
            self.gmm = GaussianMixture(n_components=optimal_nc,
                                     covariance_type='full',
                                 reg_covar=1e-5, max_iter=100, n_init=50)
            self.gmm.fit(X)
            self.sample_density = lambda N: self.gmm.sample(N)[0]
            self.evaluate_density = lambda x: numpy.exp(self.gmm.score_samples(x))
        else:
            self.gmm = self.params
            self.sample_density = lambda N: self.gmm.sample(N)[0]
            self.evaluate_density = lambda x: numpy.exp(
                self.gmm.score_samples(x))
