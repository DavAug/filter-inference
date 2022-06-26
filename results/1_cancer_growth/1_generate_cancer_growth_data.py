import os

import chi
import numpy as np
import pandas as pd

from exponential_growth_model import ExponentialGrowthModel


def generate_data(n_ids_per_t):
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
    seed = n_ids_per_t - 13
    n_ids = 5000
    times = np.linspace(0, 0.6, num=6)
    dense_measurements = predictive_model.sample(
        population_parameters, times, n_samples=n_ids, seed=seed,
        return_df=False)

    # Keep only one measurement per individual
    n_times = len(times)
    n_observables = 1
    measurements = np.empty(shape=(n_ids_per_t, n_observables, n_times))
    for idt in range(n_times):
        start_ids = idt * n_ids_per_t
        end_ids = (idt + 1) * n_ids_per_t
        measurements[:, 0, idt] = dense_measurements[0, idt, start_ids:end_ids]

    # Format data as dataframe
    n_times = len(times)
    measurements_df = pd.DataFrame({
        'Observable': 'Tumour volume',
        'Value': measurements.flatten(),
        'ID': np.arange(1, len(measurements.flatten()) + 1),
        'Time': np.broadcast_to(
            times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
    })

    # Export data to file
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df.to_csv(
        directory + '/data/1_cancer_growth_data_%d.csv' % n_ids_per_t,
        index=False)


if __name__ == '__main__':
    for n in [15, 45, 135, 405]:
        generate_data(n)
