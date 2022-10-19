import os

import chi
import numpy as np
import pints

from exponential_growth_model import ExponentialGrowthModel


def generate_data():
    # Define data-generating model
    mechanistic_model = ExponentialGrowthModel()
    error_model = chi.GaussianErrorModel()
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(n_dim=2, dim_names=['Initial count', 'Growth rate']),
        chi.PooledModel(dim_names=['Sigma'])])
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Data-generating parameters
    population_parameters = [
        10,    # Mean initial condition
        2,     # Mean exponential growth
        1,     # Std. initial condition
        0.5,   # Std. exponential growth
        0.8]   # Sigma

    # Simulate measurements
    seed = 145
    n_ids = 60000
    times = np.linspace(0, 0.6, num=6)
    dense_measurements = predictive_model.sample(
        population_parameters, times, n_samples=n_ids, seed=seed,
        return_df=False)

    # Keep only one measurement per individual
    n_times = len(times)
    n_observables = 1
    n_ids_per_t = 10000
    measurements = np.empty(shape=(n_ids_per_t, n_observables, n_times))
    for idt in range(n_times):
        start_ids = idt * n_ids_per_t
        end_ids = (idt + 1) * n_ids_per_t
        measurements[:, 0, idt] = dense_measurements[0, idt, start_ids:end_ids]

    return measurements, times


def define_log_posterior(measurements, times):
    # Define log-posterior
    n_ids = np.prod(measurements.shape)
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
        log_prior, n_samples=10000)

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
        draw=slice(warmup, n_iterations, thinning))[[
            'Mean Initial volume',
            'Std. Initial volume',
            'Mean Growth rate',
            'Std. Growth rate',
            'Sigma Tumour volume'
        ]].to_netcdf(filename)


if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__))
    m, t = generate_data()
    lp = define_log_posterior(m, t)
    filename = \
        directory + \
        '/posteriors' + \
        '/S2_filter_inference_cancer_growth_N_equal_S_10000.nc'
    run_inference(lp, filename)
