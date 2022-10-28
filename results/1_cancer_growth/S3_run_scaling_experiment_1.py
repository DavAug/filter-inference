import os
import timeit

import chi
import numpy as np
import pandas as pd
import pints
from tqdm import tqdm

from exponential_growth_model import ExponentialGrowthModel


def define_model():
    # Define mechanistic model
    mechanistic_model = ExponentialGrowthModel()

    # Define error model
    error_model = chi.GaussianErrorModel()

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(n_dim=2, dim_names=['Initial count', 'Growth rate']),
        chi.PooledModel(dim_names=['Sigma'])])

    # Define model paramters
    parameters = [
        10,    # Mean initial condition
        2,     # Mean exponential growth
        1,     # Std. initial condition
        0.5,   # Std. exponential growth
        0.8]   # Sigma

    return mechanistic_model, error_model, population_model, parameters


def generate_measurements(n_ids_per_t, predictive_model, parameters, seed):
    # Simulate dense measurements
    n_times = 6
    n_ids = n_ids_per_t * n_times
    times = np.linspace(0, 0.6, num=n_times)
    dense_measurements = predictive_model.sample(
        parameters, times, n_samples=n_ids, seed=seed, return_df=False)

    # Keep only one measurement per individual
    n_observables = 1
    measurements = np.empty(shape=(n_ids_per_t, n_observables, n_times))
    for idt in range(n_times):
        start_ids = idt * n_ids_per_t
        end_ids = (idt + 1) * n_ids_per_t
        measurements[:, 0, idt] = dense_measurements[0, idt, start_ids:end_ids]

    return measurements, times


def define_nlme_log_posterior(
        measurements, times, mechanistic_model, error_model, population_model):
    # Format data as dataframe
    n_ids_per_t, n_observables, n_times = measurements.shape
    ids = np.arange(1, n_ids_per_t * n_times * n_observables + 1)
    n_times = len(times)
    measurements_df = pd.DataFrame({
        'Observable': 'Count',
        'Value': measurements.flatten(),
        'ID': ids,
        'Time': np.broadcast_to(
            times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
    })

    # Define log-posterior
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 3),        # Mean initial condition
        pints.GaussianLogPrior(5, 3),        # Mean exponential growth
        pints.LogNormalLogPrior(-0.1, 1),    # Std. initial condition
        pints.LogNormalLogPrior(-1, 1),      # Std. exponential growth
        pints.LogNormalLogPrior(-1, 1))      # Sigma
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.set_population_model(population_model)
    problem.set_data(measurements_df)
    problem.set_log_prior(log_prior)

    return problem.get_log_posterior()


def define_filter_log_posterior(
        measurements, times, mechanistic_model, population_model, n_samples):
    population_filter = chi.GaussianFilter(measurements)
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 3),        # Mean initial condition
        pints.GaussianLogPrior(5, 3),        # Mean exponential growth
        pints.LogNormalLogPrior(-0.1, 1),    # Std. initial condition
        pints.LogNormalLogPrior(-1, 1),      # Std. exponential growth
        pints.LogNormalLogPrior(-1, 1))      # Sigma
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter=population_filter, times=times,
        mechanistic_model=mechanistic_model, population_model=population_model,
        log_prior=log_prior, error_on_log_scale=False, n_samples=n_samples)

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
    n_ids_per_t = [4, 8, 12, 16, 20, 24]
    mm, em, pm, p = define_model()
    predictive_model = chi.PredictiveModel(mm, em)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, pm)

    # Estimate evaluation time of log-posteriors
    nlme_costs = []
    filter_costs = [[], [], []]
    n_samples = [50, 100, 150]
    for seed, n_ids in enumerate(tqdm(n_ids_per_t)):
        meas, t = generate_measurements(n_ids, predictive_model, p, seed)
        logp = define_nlme_log_posterior(meas, t, mm, em, pm)
        nlme_costs += [estimate_evaluation_time(logp)]

        reduced_pm = chi.GaussianModel(
            n_dim=2, dim_names=['Initial count', 'Growth rate'],
            centered=False)
        for ids, s in enumerate(n_samples):
            logp = define_filter_log_posterior(meas, t, mm, reduced_pm, s)
            filter_costs[ids] += [estimate_evaluation_time(logp)]

    # Save results to csv
    directory = os.path.dirname(os.path.abspath(__file__))
    n_ids = np.array(n_ids_per_t) * 6
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
