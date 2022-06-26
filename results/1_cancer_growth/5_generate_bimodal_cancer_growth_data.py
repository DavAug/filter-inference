import os

import chi
import numpy as np
import pandas as pd

from exponential_growth_model import ExponentialGrowthModel


def generate_data(n_ids_per_t):
    # Define data-generating model
    mechanistic_model = ExponentialGrowthModel()
    error_model = chi.GaussianErrorModel()
    growth_rate_model = chi.CovariatePopulationModel(
        chi.GaussianModel(),
        chi.LinearCovariateModel(cov_names=['Shift aggressive variant']),
        dim_names=['Growth rate'])
    growth_rate_model.set_population_parameters([(0, 0)])
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(n_dim=1, dim_names=['Initial count']),
        growth_rate_model,
        chi.PooledModel(dim_names=['Sigma'])])
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Data-generating parameters
    population_parameters = [
        10,    # Mean initial condition
        1,     # Std. initial condition
        2,     # Mean exponential growth (slow variant)
        0.5,   # Std. exponential growth
        2,     # Growth increase (fast variant)
        0.8]   # Sigma

    # Simulate measurements
    seed = n_ids_per_t
    n_ids = 5000
    times = np.linspace(0, 0.6, num=6)
    covariates = [0, 1]
    dense_measurements = []
    for idc, cov in enumerate(covariates):
        dense_measurements += [predictive_model.sample(
            population_parameters, times, n_samples=n_ids, seed=seed,
            return_df=False, covariates=[cov])]

    # Keep only one measurement per individual
    n_times = len(times)
    n_observables = 1
    measurements = np.empty(shape=(n_ids_per_t*2, n_observables, n_times))
    for idm, meas in enumerate(dense_measurements):
        for idt in range(n_times):
            start_ids = idt * n_ids_per_t
            end_ids = (idt + 1) * n_ids_per_t
            measurements[idm*n_ids_per_t:(idm+1)*n_ids_per_t, 0, idt] = \
                meas[0, idt, start_ids:end_ids]

    # Format data as dataframe
    n_times = len(times)
    ids = np.arange(1, len(measurements.flatten()) + 1)
    measurements_df = pd.DataFrame({
        'Observable': 'Tumour volume',
        'Value': measurements.flatten(),
        'ID': ids,
        'Time': np.broadcast_to(
            times[np.newaxis, :], (2*n_ids_per_t, n_times)).flatten(),
    })
    covariates = np.zeros(shape=(n_ids_per_t*2, n_observables, n_times))
    covariates[n_ids_per_t:] = 1
    measurements_df = pd.concat([measurements_df, pd.DataFrame({
        'Observable': 'Aggressive variant',
        'Value': covariates.flatten(),
        'ID': ids})
    ])

    # Export data to file
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df.to_csv(
        directory + '/data/2_bimodal_cancer_growth_data_%d.csv'
        % (2 * n_ids_per_t),
        index=False)


if __name__ == '__main__':
    for n in [10, 250]:
        generate_data(n)
