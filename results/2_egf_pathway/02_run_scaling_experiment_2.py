import os

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


def define_nlme_log_posterior(
        measurements, times, mechanistic_model, error_model, population_model):
    # Format data as dataframe
    n_ids_per_t, n_observables, n_times = measurements.shape
    ids = np.arange(1, n_ids_per_t * n_times * n_observables + 1)
    measurements_df = pd.DataFrame({
        'Observable': 'central.receptor_active_amount ligand conc. 2',
        'Value': measurements[:, 0].flatten(),
        'ID': ids[:n_times * n_ids_per_t],
        'Time': np.broadcast_to(
            times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
    })
    measurements_df = pd.concat([
        measurements_df,
        pd.DataFrame({
            'Observable': 'central.receptor_inactive_amount ligand conc. 2',
            'Value': measurements[:, 1].flatten(),
            'ID': ids[n_times * n_ids_per_t:2*(n_times * n_ids_per_t)],
            'Time': np.broadcast_to(
                times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
        })
    ])
    measurements_df = pd.concat([
        measurements_df,
        pd.DataFrame({
            'Observable': 'central.receptor_active_amount ligand conc. 10',
            'Value': measurements[:, 2].flatten(),
            'ID': ids[2*(n_times * n_ids_per_t):3*(n_times * n_ids_per_t)],
            'Time': np.broadcast_to(
                times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
        })
    ])
    measurements_df = pd.concat([
        measurements_df,
        pd.DataFrame({
            'Observable': 'central.receptor_inactive_amount ligand conc. 10',
            'Value': measurements[:, 3].flatten(),
            'ID': ids[3*(n_times * n_ids_per_t):],
            'Time': np.broadcast_to(
                times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
        })
    ])

    # Define log-posterior
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
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.set_population_model(population_model)
    problem.set_data(measurements_df)
    problem.set_log_prior(log_prior)

    return problem.get_log_posterior()


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
    n_ids_per_t = [1, 2, 3, 4, 5, 6]
    mm, em, pm, p = define_egf_model()
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

        reduced_pm = chi.ComposedPopulationModel([
            chi.GaussianModel(dim_names=['Activation rate'], centered=False),
            chi.PooledModel(n_dim=3, dim_names=[
                'Deactivation rate', 'Deg. rate (act.)',
                'Deg. rate (inact.)']),
            chi.GaussianModel(dim_names=['Production rate'], centered=False)])
        logp = define_filter_log_posterior(meas, t, mm, reduced_pm, 100)
        filename = directory + '/filter_number_of_evaluations_%d.csv' % n_ids
        run_inference(logp, filename)

    # Collect data from temporary files into one csv file
    warmup = 500
    n_iterations = 1500
    n_evals = [[], []]
    for idf, f in enumerate(
            ['/nlme_number_of_evaluations', '/filter_number_of_evaluations']):
        for n_ids in n_ids_per_t:
            # Load data
            data = pd.read_csv(
                directory + f + '_%d.csv' % n_ids)

            # Get number of evaluations post warmup
            e_warmup = data[data['Iter.'] == warmup]['Eval.'].values[0]
            e_total = data[data['Iter.'] == n_iterations]['Eval.'].values[0]
            n_evals[idf].append(e_total - e_warmup)

    # Save results to file
    n_ids = np.array(n_ids_per_t) * 6
    df = pd.DataFrame({
        'Number of measured individuals': n_ids,
        'Type': 'NLME',
        'Number of evaluations': n_evals[0]})
    df = pd.concat([df, pd.DataFrame({
        'Number of measured individuals': n_ids,
        'Type': 'Filter 100',
        'Number of evaluations': n_evals[1]})])
    df.to_csv(directory + '/number_of_evaluations.csv')

    # Remove temporary files
    for idf, f in enumerate(
            ['/nlme_number_of_evaluations', '/filter_number_of_evaluations']):
        for n_ids in n_ids_per_t:
            os.remove(directory + f + '_%d.csv' % n_ids)
