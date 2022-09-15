import os

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


def run_inference(log_posterior, tofile):
    # Run inference
    seed = 4
    n_chains = 1
    n_iterations = 1500
    initial_params = log_posterior.sample_initial_parameters(
        n_samples=n_chains, seed=seed)
    controller = pints.MCMCController(
        log_pdf=log_posterior, chains=n_chains, x0=initial_params,
        method=pints.NoUTurnMCMC)
    controller.set_max_iterations(n_iterations)
    controller.set_log_to_file(tofile, csv=True)
    controller.set_chain_storage(False)
    controller.set_parallel(True)
    controller.run()


if __name__ == '__main__':
    n_ids_per_t = [4, 8, 12, 16, 20, 24]
    mm, em, pm, p = define_model()
    predictive_model = chi.PredictiveModel(mm, em)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, pm)

    # Estimate evaluation time of log-posteriors
    directory = os.path.dirname(os.path.abspath(__file__))
    for seed, n_ids in enumerate(tqdm(n_ids_per_t)):
        meas, t = generate_measurements(n_ids, predictive_model, p, seed)
        logp = define_nlme_log_posterior(meas, t, mm, em, pm)
        filename = directory + '/nlme_number_of_evaluations_%d.csv' % n_ids
        run_inference(logp, filename)

        reduced_pm = chi.GaussianModel(
            n_dim=2, dim_names=['Initial count', 'Growth rate'],
            centered=False)
        logp = define_filter_log_posterior(meas, t, mm, reduced_pm, 100)
        filename = directory + '/filter_number_of_evaluations_%d.csv' % n_ids
        run_inference(logp, filename)

    # Collect data from temporary files into one csv file
    warmup = 500
    iter_per_log = 20
    warmup_index = 500 // 20
    n_iterations = 1500
    n_evals = [[], []]
    for idf, f in enumerate(
            ['/nlme_number_of_evaluations', '/filter_number_of_evaluations']):
        for n_ids in n_ids_per_t:
            # Load data
            data = pd.read_csv(
                directory + f + '_%d.csv' % n_ids)
            # Get final number of evaluations and final run time
            # (final valid entry is determined by first NaN entry)
            final_index = data.isna().any(axis=1)[warmup_index:].argmax()
            final_iter = n_iterations
            if final_index > 0:
                # Final index is smaller than n_iterations
                final_iter = data.iloc[warmup_index+final_index]['Iter.']
            mask = data['Iter.'] == final_iter
            e = data[mask]['Eval.'].values[0]
            # Subtract warm up
            mask = data['Iter.'] == warmup
            e -= data[mask]['Eval.'].values[0]
            # Estimate number of evaluations and run time for 1000 iterations
            # of a single chain
            e = e / (final_iter - warmup) * n_iterations
            # Append to container
            n_evals[idf].append(e)

    # Save results to file
    n_ids = np.array(n_ids_per_t) * 6
    df = pd.DataFrame({
        'Number of measured individuals': n_ids,
        'Type': 'NLME',
        'Number of evaluations': n_evals[0]})
    df = pd.concat([df, pd.DataFrame({
        'Number of measured individuals': n_ids,
        'Type': 'Filter 100',
        'Cost in sec': n_evals[1]})])
    df.to_csv(directory + '/number_of_evaluations.csv')

    # Remove temporary files
    for idf, f in enumerate(
            ['/nlme_number_of_evaluations', '/filter_number_of_evaluations']):
        for n_ids in n_ids_per_t:
            os.remove(directory + f + '_%d.csv' % n_ids)
