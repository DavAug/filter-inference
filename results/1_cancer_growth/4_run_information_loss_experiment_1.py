import os

import chi
import numpy as np
import pandas as pd
import pints

from exponential_growth_model import ExponentialGrowthModel


def define_log_posterior(n_sim):
    # Import data
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df = pd.read_csv(
        directory + '/data/1_cancer_growth_data_405.csv')

    # Reshape data to numpy array of shape (n_ids_per_t, n_output, n_times)
    output = measurements_df.Observable.unique()[0]
    n_outputs = 1
    n_ids_per_t = 405
    times = np.sort(measurements_df.Time.unique())
    measurements = np.empty(shape=(n_ids_per_t, n_outputs, len(times)))
    measurements_df = measurements_df[measurements_df.Observable == output]
    for idt, time in enumerate(times):
        mask = measurements_df.Time == time
        measurements[:, 0, idt] = measurements_df[mask].Value.values

    # Define log-posterior
    mechanistic_model = ExponentialGrowthModel()
    population_filter = chi.GaussianFilter(measurements)
    population_model = chi.GaussianModel(
        n_dim=2, dim_names=['Initial count', 'Growth rate'], centered=False)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 3),        # Mean initial condition
        pints.GaussianLogPrior(5, 3),        # Mean exponential growth
        pints.LogNormalLogPrior(2, 1),       # Std. initial condition
        pints.LogNormalLogPrior(0.5, 1),     # Std. exponential growth
        pints.GaussianLogPrior(0.8, 0.1))    # Sigma
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter, times, mechanistic_model, population_model,
        log_prior, n_samples=n_sim)

    return log_posterior


def run_inference(log_posterior, filename):
    seed = 3
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(pints.NoUTurnMCMC)
    n_iterations = 1500
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Save results
    warmup = 500
    thinning = 1
    posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning)).to_netcdf(filename)


if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__))
    for idn, n in enumerate([3, 10, 100, 500]):
        lp = define_log_posterior(n)
        filename = \
            directory + \
            '/posteriors/%d_filter_inference_cancer_growth_%d_ids.nc' \
            % (idn+6, n)
        run_inference(lp, filename)
