import os

import chi
import numpy as np
import pandas as pd
import pints

from exponential_growth_model import ExponentialGrowthModel


def define_log_posterior(n_ids_per_t, f):
    # Import data
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df = pd.read_csv(
        directory + '/data/2_bimodal_cancer_growth_data_%d.csv' % n_ids_per_t)

    # Reshape data to numpy array of shape (n_ids_per_t, n_output, n_times)
    output = 'Tumour volume'
    n_outputs = 1
    times = np.sort(measurements_df.Time.dropna().unique())
    measurements = np.empty(shape=(n_ids_per_t, n_outputs, len(times)))
    temp = measurements_df[measurements_df.Observable == output]
    for idt, time in enumerate(times):
        mask = temp.Time == time
        measurements[:, 0, idt] = temp[mask].Value.values

    # Select filter
    if f == 'gaussian':
        f = chi.GaussianFilter
    elif f == 'lognormal':
        f = chi.LogNormalFilter
    elif f == 'gaussian_mixture':
        f = chi.GaussianMixtureFilter
    elif f == 'gaussian_kde':
        f = chi.GaussianKDEFilter
    else:
        f = chi.LogNormalKDEFilter

    # Define log-posterior
    covariates = np.zeros((100, 1))
    covariates[50:] = 1
    mechanistic_model = ExponentialGrowthModel()
    population_filter = f(measurements)
    growth_rate_model = chi.CovariatePopulationModel(
        chi.GaussianModel(centered=False),
        chi.LinearCovariateModel(cov_names=['Aggressive variant']),
        dim_names=['Growth rate'])
    growth_rate_model.set_population_parameters([(0, 0)])
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(
            n_dim=1, dim_names=['Initial count'], centered=False),
        growth_rate_model])
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 3),        # Mean initial condition
        pints.LogNormalLogPrior(2, 1),       # Std. initial condition
        pints.GaussianLogPrior(5, 3),        # Mean exponential growth (slow)
        pints.LogNormalLogPrior(0.5, 1),     # Std. exponential growth
        pints.GaussianLogPrior(5, 3),        # Mean shift growth (fast)
        pints.GaussianLogPrior(0.8, 0.1))    # Sigma
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter, times, mechanistic_model, population_model,
        log_prior, covariates=covariates)

    # Create a second log-posterior to sample valid initial values from
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 1),        # Mean initial condition
        pints.LogNormalLogPrior(0, 0.1),       # Std. initial condition
        pints.GaussianLogPrior(5, 1),        # Mean exponential growth (slow)
        pints.LogNormalLogPrior(0, 0.1),     # Std. exponential growth
        pints.GaussianLogPrior(5, 1),        # Mean shift growth (fast)
        pints.GaussianLogPrior(0.8, 0.1))    # Sigma
    temp = chi.PopulationFilterLogPosterior(
        population_filter, times, mechanistic_model, population_model,
        log_prior, covariates=covariates)

    return log_posterior, temp


def run_inference(log_posterior, filename, temp):
    seed = 6
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(pints.NoUTurnMCMC)
    n_iterations = 1500

    # Make sure initial values contain only positive simulated measurements
    controller._initial_params = temp.sample_initial_parameters(seed=seed)

    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Save results
    warmup = 500
    thinning = 1
    posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning)).to_netcdf(filename)


if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__))
    filters = [
        'gaussian', 'lognormal', 'gaussian_mixture', 'gaussian_kde',
        'lognormal_kde']
    for idn, n in enumerate([20, 500]):
        for idf, f in enumerate(filters):
            lp, temp = define_log_posterior(n, f)
            if lp is None:
                continue
            filename = \
                directory + '/posteriors/' \
                '%d_filter_inference_bimodal_cancer_growth_%s_filter_%d.nc' \
                % (idn*5+idf+12, f, n)
            run_inference(lp, filename, temp)
