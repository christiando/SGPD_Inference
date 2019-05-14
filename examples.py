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
from .datageneration import sample_density
from .variational import VI_SGPD
from .basemeasures import BaseMeasure
from .sampler import GS_SGPD
from matplotlib import pyplot


def example_1D():
    D = 1
    numpy.random.seed(1)
    grid_points = 200
    num_samples = 150
    S_borders = numpy.array([[-4.,4.]])
    dx = numpy.prod((S_borders[:,1] - S_borders[:,0])/float(grid_points))

    base_measure = BaseMeasure(D, type='normal')
    cov_params = [5., numpy.array([.5])]
    X_samples, X_prev, g_prev, X_grid, g_grid = sample_density(num_samples,
                                                        cov_params,
                                                        base_measure,
                                                        grid_points=grid_points,
                                                        S_borders=S_borders)
    X = X_grid
    base_measure_grid = base_measure.evaluate_density(X)
    density = base_measure_grid/(1. + numpy.exp(-g_grid))
    density = density / numpy.sum(density) / dx

    X_train = X_samples[:100]
    X_test = X_samples[100:]
    print('Data sampled - Start VB')
    vb = VI_SGPD(X_train, cov_params, base_measure, num_inducing_points=50 * D,
                 num_integration_points=int(1e3*D),
                 update_hyperparams=0,
                 update_basemeasure=0)
    vb.run()
    # log_test_likelihood_vb = vb.predictive_log_likelihood(X_test)
    # print(log_test_likelihood_vb)
    density_pred_vb = vb.predictive_density_function(X, dx)
    print('VB finished - Start sampler')
    gibbs = GS_SGPD(X_train, cov_params, base_measure, burnin=int(1e3),
                    max_iterations=int(2e3),
                    num_integration_points=int(1e3 * D),
                    update_hyperparams=0,
                    update_basemeasure=0)
    gibbs.run()
    print('Sampler finished - Sample predictive density')
    # log_test_likelihood_gibbs = gibbs.predictive_log_likelihood(
    #     X_test)
    # print(log_test_likelihood_gibbs)
    density_pred_gibbs = gibbs.predictive_posterior(X, dx)

    fig = pyplot.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.plot(X,density[0], 'k', linewidth=2, label='Ground truth')
    ax.plot(X, density_pred_vb, 'C0', linewidth=2, label='Variational')
    ax.plot(X, density_pred_gibbs, 'C3', linewidth=2, linestyle='--',
            label='Gibbs')
    ax.vlines(X_train, -.05,0)
    ax.set_xlabel('$x$')
    ax.set_ylabel('Density $\\rho(x)$')
    ax.legend(frameon=0)
    fig.show()

def example_2D():

    D = 2
    numpy.random.seed(2)
    grid_points = 50
    num_samples = 200
    S_borders = numpy.array([[-3., 4.],
                             [-4., 4.5]])
    dx = numpy.prod((S_borders[:, 1] - S_borders[:, 0]) / float(grid_points))
    base_measure = BaseMeasure(D, type='normal')
    cov_params = [5., numpy.array([1., 1.])]
    X_samples, X_prev, g_prev, X_grid, g_grid = sample_density(num_samples,
                                                        cov_params,
                                                        base_measure,
                                                        grid_points=grid_points,
                                                        S_borders=S_borders)

    X_mesh = numpy.meshgrid(X_grid[:, 0], X_grid[:, 1])
    X = numpy.vstack([X_mesh[0].flatten(), X_mesh[1].flatten()]).T
    base_measure_grid = base_measure.evaluate_density(X)
    base_measure_grid = base_measure_grid.reshape((grid_points, grid_points))
    g_reshaped = g_grid.reshape((grid_points, grid_points))
    density = base_measure_grid / (1. + numpy.exp(-g_reshaped))
    density = density / numpy.sum(density) / dx

    X_train = X_samples[:200]
    # X_test = X_samples[200:]
    print('Data sampled - Start VB')

    vb = VI_SGPD(X_train, cov_params, base_measure,
                 num_inducing_points=50 * D,
                 num_integration_points=int(1e3 * D),
                 update_hyperparams=0,
                 update_basemeasure=0)
    vb.run()
    # log_test_likelihood_vb = vb.predictive_log_likelihood(X_test)
    density_pred_vb = vb.predictive_density_function(X, dx)
    density_pred_vb = density_pred_vb.reshape((grid_points,grid_points))
    print('VB finished - Start sampler')
    gibbs = GS_SGPD(X_train, cov_params, base_measure, burnin=int(1e2),
                    max_iterations=int(1e3),
                    num_integration_points=int(5e2 * D),
                    update_hyperparams=0,
                    update_basemeasure=0)
    gibbs.run()
    print('Sampler finished - Sample predictive density')
    # log_test_likelihood_gibbs = gibbs.predictive_log_likelihood(X_test)
    density_pred_gibbs = gibbs.predictive_posterior(X, dx)
    density_pred_gibbs = density_pred_gibbs.reshape((grid_points, grid_points))
    print('Gibbs done!')

    max_value = numpy.amax([numpy.amax(density), numpy.amax(density_pred_gibbs),
                            numpy.amax(density_pred_vb)])
    fig = pyplot.figure('2D Experiment', figsize=(6, 2))
    ax1 = fig.add_axes([.02, .1, .3, .7])
    ax1.contour(X_grid[:, 0], X_grid[:, 1], density, vmin=0,
                vmax=max_value)
    ax1.scatter(X_samples[:200, 0], X_samples[:200, 1], color='C1', s=5.)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_title('True')
    ax2 = fig.add_axes([.35, .1, .3, .7])
    ax2.contour(X_grid[:, 0], X_grid[:, 1], density_pred_vb, vmin=0,
                vmax=max_value)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title('VB')
    ax3 = fig.add_axes([.68, .1, .3, .7])
    ax3.contour(X_grid[:, 0], X_grid[:, 1], density_pred_gibbs, vmin=0,
                vmax=max_value)
    ax3.set_title('Gibbs')
    ax3.set_yticks([])
    ax3.set_xticks([])
    fig.show()
