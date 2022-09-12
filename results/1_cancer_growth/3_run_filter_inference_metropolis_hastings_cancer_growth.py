import os

import chi
import numpy as np
import pandas as pd
import pints

from exponential_growth_model import ExponentialGrowthModel


class StochasticFilterLogPosterior(chi.LogPosterior):
    """
    Implements the stochastic version of filter inference.
    """
    def __init__(
            self, population_filter, times, predictive_model, log_prior,
            n_samples):
        self._filter = population_filter
        self._times = times
        self._predictive_model = predictive_model
        self._log_prior = log_prior
        self._n_samples = n_samples
        self._n_parameters = self._predictive_model.n_parameters()

    def __call__(self, parameters):
        score = self._log_prior(parameters)
        if np.isinf(score):
            return -np.inf

        # Simulate measurements
        measurements = self._predictive_model.sample(
            parameters=parameters,
            times=self._times,
            n_samples=self._n_samples,
            return_df=False
        )

        # Estimate log-likelihood
        measurements = np.moveaxis(
            measurements, source=(0, 1, 2), destination=(1, 2, 0))
        score += self._filter.compute_log_likelihood(measurements)

        return score

    def get_id(self, *args, **kwargs):
        """
        Returns the id of the log-posterior. If no id is set, ``None`` is
        returned.
        """
        return [None] * self._n_parameters

    def get_parameter_names(self, *args, **kwargs):
        """
        Returns the names of the model parameters. By default the parameters
        are enumerated and assigned with the names 'Param #'.
        """
        # Get parameter names
        names = self._predictive_model.get_parameter_names()

        return names

    def n_parameters(self, *args, **kwargs):
        """
        Returns the number of parameters of the posterior.
        """
        return self._n_parameters


class MetropolisHastingsMCMC(pints.MetropolisRandomWalkMCMC):
    """
    A simple wrapper of pints.MetropolisRandomWalkMCMC that enables setting
    of the Covariance matrix with set_hyper_parameters.
    """
    def __init__(self, x0, sigma=None):
        super(MetropolisHastingsMCMC, self).__init__(x0, sigma)
        self._n_hyper_params = 1

    def n_hyper_parameters(self):
        return self._n_hyper_params

    def set_hyper_parameters(self, x):
        if len(x) != self._n_hyper_params:
            raise ValueError(
                'Setting hyperparameters failed. The provided number of '
                'hyperparameters does not match the number of hyperparameters '
                'of the model.')
        self._sigma0 = x[0]


def define_log_posterior(n_ids_per_t):
    # Import data
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df = pd.read_csv(
        directory + '/data/1_cancer_growth_data_%d.csv' % n_ids_per_t)

    # Reshape data to numpy array of shape (n_ids_per_t, n_output, n_times)
    output = measurements_df.Observable.unique()[0]
    n_outputs = 1
    times = np.sort(measurements_df.Time.unique())
    measurements = np.empty(shape=(n_ids_per_t, n_outputs, len(times)))
    measurements_df = measurements_df[measurements_df.Observable == output]
    for idt, time in enumerate(times):
        mask = measurements_df.Time == time
        measurements[:, 0, idt] = measurements_df[mask].Value.values

    # Define log-posterior
    mechanistic_model = ExponentialGrowthModel()
    error_model = chi.GaussianErrorModel()
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(
            n_dim=2, dim_names=['Initial count', 'Growth rate']),
        chi.PooledModel(n_dim=1, dim_names=['Sigma'])
    ])
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    n_samples = 100
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 3),        # Mean initial condition
        pints.GaussianLogPrior(5, 3),        # Mean exponential growth
        pints.LogNormalLogPrior(2, 1),       # Std. initial condition
        pints.LogNormalLogPrior(0.5, 1),     # Std. exponential growth
        pints.GaussianLogPrior(0.8, 0.1))    # Sigma
    population_filter = chi.GaussianFilter(measurements)
    log_posterior = StochasticFilterLogPosterior(
        population_filter, times, predictive_model, log_prior,
        n_samples=n_samples)

    return log_posterior


def run_inference(log_posterior, filename):
    seed = 3
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(MetropolisHastingsMCMC)

    # Initialise sampler at data-generating parameters
    controller._initial_params[0] = np.array([
        10,    # Mean initial condition
        2,     # Mean exponential growth
        1,     # Std. initial condition
        0.5,   # Std. exponential growth
        0.8]   # Sigma
        )

    n_iterations = 65000
    covariance_matrix = np.diag([0.03, 0.02, 0.03, 0.01, 0.02])
    posterior_samples = controller.run(
        n_iterations=n_iterations, hyperparameters=[covariance_matrix],
        log_to_screen=True)

    # Save results
    warmup = 0
    thinning = 1
    posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning)).to_netcdf(filename)


if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__))
    lp = define_log_posterior(15)
    filename = \
        directory + \
        '/posteriors/' + \
        '99_filter_inference_metropolis_hastings_cancer_growth_100_ids.nc'
    run_inference(lp, filename)
