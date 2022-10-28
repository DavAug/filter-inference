import os
import timeit

import chi
import numpy as np
import pandas as pd
import pints
from tqdm import tqdm

from growth_factor_model import GrowthFactorModel


def define_egf_model():
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

    # Define model paramters
    parameters = [
        1.7,    # Mean activation rate
        0.05,   # Std. activation rate
        8,      # deactivation rate
        0.015,  # degradation rate (active)
        0.25,   # degradation rate (inactive)
        1.7,    # Mean production rate
        0.05,   # Std. production rate
        0.05,   # Sigma act. ligand conc. Amount 1
        0.05,   # Sigma inact. ligand conc. Amount 1
        0.05,   # Sigma act. ligand conc. Amount 2
        0.05]   # Sigma inact. ligand conc. Amount 2

    return mechanistic_model, error_models, population_model, parameters


def generate_measurements(n_ids_per_t, predictive_model, parameters, seed):
    # Simulate dense measurements
    n_observables = 4
    times = np.array([1, 5, 10, 15, 20, 25])
    n_times = len(times)
    n_ids = n_ids_per_t * n_times * n_observables
    dense_measurements = predictive_model.sample(
        parameters, times, n_samples=n_ids, seed=seed, return_df=False)

    # Keep only one measurement per individual
    measurements = np.empty(shape=(n_ids_per_t, n_observables, n_times))
    for ido in range(n_observables):
        offset = ido * n_times * n_ids_per_t
        for idt in range(n_times):
            start_ids = offset + idt * n_ids_per_t
            end_ids = offset + (idt + 1) * n_ids_per_t
            measurements[:, ido, idt] = dense_measurements[
                ido, idt, start_ids:end_ids]

    return measurements, times


def define_filter_log_posterior(
        measurements, times, mechanistic_model, population_model, n_samples):
    population_filter = chi.GaussianFilter(measurements)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(2, 0.5),       # Mean activation rate
        pints.LogNormalLogPrior(-2, 0.5),     # Std. activation rate
        pints.GaussianLogPrior(10, 2),        # deactivation rate
        pints.GaussianLogPrior(0.02, 0.005),  # degradation rate (active)
        pints.GaussianLogPrior(0.3, 0.05),    # degradation rate (inactive)
        pints.GaussianLogPrior(2, 0.5),       # Mean production rate
        pints.LogNormalLogPrior(-2, 0.5),     # Std. production rate
        pints.LogNormalLogPrior(-2, 0.5),     # Sigma
        pints.LogNormalLogPrior(-2, 0.5),     # Sigma
        pints.LogNormalLogPrior(-2, 0.5),     # Sigma
        pints.LogNormalLogPrior(-2, 0.5))     # Sigma
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter=population_filter, times=times,
        mechanistic_model=mechanistic_model, population_model=population_model,
        log_prior=log_prior, error_on_log_scale=True, n_samples=n_samples)

    return log_posterior


def estimate_evaluation_time(log_posterior):
    test_parameters = log_posterior.sample_initial_parameters()[0]
    # Evaluate once, so sensitivities are switched on
    log_posterior.evaluateS1(test_parameters)

    number = 1
    repeats = 10
    run_time = timeit.repeat(
        'logp(p)',
        globals=dict(logp=log_posterior, p=test_parameters),
        number=number, repeat=repeats)

    return np.min(run_time) / number


if __name__ == '__main__':
    n_ids_per_t = [1, 2, 3, 4, 5, 6]
    mm, em, pm, p = define_egf_model()
    predictive_model = chi.PredictiveModel(mm, em)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, pm)

    # Estimate evaluation time of log-posteriors
    filter_costs = [[], [], []]
    n_samples = [50, 100, 150]
    for seed, n_ids in enumerate(tqdm(n_ids_per_t)):
        meas, t = generate_measurements(n_ids, predictive_model, p, seed)

        reduced_pm = chi.ComposedPopulationModel([
            chi.GaussianModel(dim_names=['Activation rate'], centered=False),
            chi.PooledModel(n_dim=3, dim_names=[
                'Deactivation rate', 'Deg. rate (act.)',
                'Deg. rate (inact.)']),
            chi.GaussianModel(dim_names=['Production rate'], centered=False)])
        for ids, s in enumerate(n_samples):
            logp = define_filter_log_posterior(meas, t, mm, reduced_pm, s)
            filter_costs[ids] += [estimate_evaluation_time(logp)]

    # Save results to csv
    directory = os.path.dirname(os.path.abspath(__file__))
    n_ids = np.array(n_ids_per_t) * 6 * 4
    nlme_costs = [np.nan] * len(n_ids)
    df = pd.DataFrame({
        'Number of measured individuals': n_ids,
        'Type': 'NLME',
        'Cost in sec': nlme_costs})
    for ids, s in enumerate(n_samples):
        df = pd.concat([df, pd.DataFrame({
            'Number of measured individuals': n_ids,
            'Type': 'Filter %d' % s,
            'Cost in sec': filter_costs[ids]})])
    df.to_csv(directory + '/supplementary_scaling_with_measurements.csv')
