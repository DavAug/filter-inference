import os

import chi
import numpy as np
import pandas as pd

from growth_factor_model import GrowthFactorModel


def define_data_generating_model():
    # Define mechanistic model
    mechanistic_model = GrowthFactorModel()

    # Define error model
    error_models = [
        chi.LogNormalErrorModel(),  # Active ligand concentration Amount 1
        chi.LogNormalErrorModel(),  # Inactive ligand concentration Amount 1
        chi.LogNormalErrorModel(),  # Active ligand concentration Amount 1
        chi.LogNormalErrorModel()]  # Inactive ligand concentration Amount 1

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate']),
        chi.PooledModel(n_dim=3, dim_names=[
            'Deactivation rate', 'Deg. rate (act.)', 'Deg. rate (inact.)']),
        chi.GaussianModel(dim_names=['Production rate']),
        chi.PooledModel(n_dim=4, dim_names=[
            'Sigma act. ligand conc. Amount 1',
            'Sigma inact. ligand conc. Amount 1',
            'Sigma act. ligand conc. Amount 2',
            'Sigma inact. ligand conc. Amount 2'])])

    # Compose data-generating model
    predictive_model = chi.PredictiveModel(mechanistic_model, error_models)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Define model paramters
    parameters = [
        1.7,    # Mean activation rate
        0.05,   # Std. activation rate
        8,      # deactivation rate
        0.015,  # degradation rate (active)
        0.25,   # degradation rate (inactive)
        1.7,    # Mean production rate
        0.05,   # Std. production rate
        0.05,  # Sigma act. ligand conc. Amount 1
        0.05,  # Sigma inact. ligand conc. Amount 1
        0.05,  # Sigma act. ligand conc. Amount 2
        0.05]  # Sigma inact. ligand conc. Amount 2

    return predictive_model, parameters


def generate_data(predictive_model, parameters):
    # Simulate measurements
    seed = 2
    n_ids = 2400
    times = np.array([1, 5, 10, 15, 20, 25])
    dense_measurements = predictive_model.sample(
        parameters, times, n_samples=n_ids, seed=seed, return_df=False)

    # Keep only one measurement per individual
    n_ids_per_t = 100
    n_times = len(times)
    n_observables = 4
    measurements = np.empty(shape=(n_ids_per_t, n_observables, n_times))
    for ido in range(n_observables):
        offset = ido * n_times * n_ids_per_t
        for idt in range(n_times):
            start_ids = offset + idt * n_ids_per_t
            end_ids = offset + (idt + 1) * n_ids_per_t
            measurements[:, ido, idt] = dense_measurements[
                ido, idt, start_ids:end_ids]

    # Format data as dataframe
    n_times = len(times)
    ids = np.arange(1, n_ids_per_t * n_times * n_observables + 1)
    measurements_df = pd.DataFrame({
        'Observable': 'Act. receptor conc. Amount 1',
        'Value': measurements[:, 0].flatten(),
        'ID': ids[:n_times * n_ids_per_t],
        'Time': np.broadcast_to(
            times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
    })
    measurements_df = pd.concat([
        measurements_df,
        pd.DataFrame({
            'Observable': 'Inact. receptor conc. Amount 1',
            'Value': measurements[:, 1].flatten(),
            'ID': ids[n_times * n_ids_per_t:2*(n_times * n_ids_per_t)],
            'Time': np.broadcast_to(
                times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
        })
    ])
    measurements_df = pd.concat([
        measurements_df,
        pd.DataFrame({
            'Observable': 'Act. receptor conc. Amount 2',
            'Value': measurements[:, 2].flatten(),
            'ID': ids[2*(n_times * n_ids_per_t):3*(n_times * n_ids_per_t)],
            'Time': np.broadcast_to(
                times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
        })
    ])
    measurements_df = pd.concat([
        measurements_df,
        pd.DataFrame({
            'Observable': 'Inact. receptor conc. Amount 2',
            'Value': measurements[:, 3].flatten(),
            'ID': ids[3*(n_times * n_ids_per_t):],
            'Time': np.broadcast_to(
                times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
        })
    ])

    return measurements_df


if __name__ == '__main__':
    m, p = define_data_generating_model()
    data = generate_data(m, p)

    # Export data to file
    directory = os.path.dirname(os.path.abspath(__file__))
    data.to_csv(directory + '/data/1_egf_pathway_data.csv', index=False)
