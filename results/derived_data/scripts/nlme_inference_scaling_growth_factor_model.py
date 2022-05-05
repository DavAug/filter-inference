import os

import chi
import numpy as np
import pandas as pd
import pints


def define_data_generating_model():
    # Define mechanistic model
    directory = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))))
    mechanistic_model = chi.SBMLModel(
        directory + '/models/dixit_growth_factor_model.xml')
    mechanistic_model.set_outputs([
        'central.receptor_active_concentration'])
    mechanistic_model.set_output_names({
        'central.receptor_active_concentration': 'Concentration (act.)'})
    mechanistic_model = chi.ReducedMechanisticModel(mechanistic_model)
    mechanistic_model.fix_parameters({
        'central.receptor_active_amount': 0,
        'central.receptor_inactive_amount': 0,
        'central.ligand_amount': 2,
        'central.size': 1
    })

    # Define error model
    error_models = chi.LogNormalErrorModel()

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate']),
        chi.PooledModel(n_dim=3, dim_names=[
            'Deactivation rate', 'Deg. rate (act.)', 'Deg. rate (inact.)']),
        chi.GaussianModel(dim_names=['Production rate']),
        chi.PooledModel(n_dim=2, dim_names=['Sigma inact.', 'Sigma act.'])])
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
        0.05]   # Sigma act.

    return mechanistic_model, error_models, predictive_model, parameters


def generate_measurements(n_ids_per_t, predictive_model, parameters):
    # Simulate dense measurements
    seed = 2
    n_times = 6
    n_ids = n_ids_per_t * n_times
    times = np.array([1, 5, 10, 15, 20])
    dense_measurements = predictive_model.sample(
        parameters, times, n_samples=n_ids, seed=seed, return_df=False)

    # Keep only one measurement per individual
    n_observables = 1
    measurements = np.empty(shape=(n_ids_per_t, n_observables, n_times))
    for idt in range(n_times):
        start_ids = idt * n_ids_per_t
        end_ids = (idt + 1) * n_ids_per_t
        measurements[:, 0, idt] = dense_measurements[0, idt, start_ids:end_ids]

    # Format data as dataframe
    ids = np.arange(1, n_ids + 1)
    n_times = len(times)
    measurements_df = pd.DataFrame({
        'Observable': 'Concentration',
        'Value': measurements.flatten(),
        'ID': ids,
        'Time': np.broadcast_to(
            times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
    })

    return measurements_df


def define_log_posterior(
        measurements, mechanistic_model, error_model, params, sigma):
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate'], centered=False),
        # chi.PooledModel(n_dim=3, dim_names=[
        #     'Deactivation rate', 'Deg. rate (act.)', 'Deg. rate (inact.)']),
        chi.GaussianModel(dim_names=['Production rate'], centered=False)])
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(2, 0.5),       # Mean activation rate
        pints.LogNormalLogPrior(-2, 0.5),     # Std. activation rate
        # pints.GaussianLogPrior(10, 2),        # deactivation rate
        # pints.GaussianLogPrior(0.02, 0.005),  # degradation rate (active)
        # pints.GaussianLogPrior(0.3, 0.05),    # degradation rate (inactive)
        pints.GaussianLogPrior(2, 0.5),       # Mean production rate
        pints.LogNormalLogPrior(-2, 0.5))     # Std. production rate
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.fix_parameters({
        'myokit.deactivation_rate': params[0],
        'myokit.degradation_rate_active_receptor': params[1],
        'myokit.degradation_rate_inactive_receptor': params[2],
        'Concentration (act.) Sigma log': sigma})
    problem.set_population_model(population_model)
    problem.set_data(measurements)
    problem.set_log_prior(log_prior)
    log_posterior = problem.get_log_posterior()

    return log_posterior


def run_inference(log_posterior, tofile):
    # Run inference
    seed = 3
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
    controller.set_parallel(False)
    controller.run()


if __name__ == '__main__':
    mm, em, pm, p = define_data_generating_model()
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for n_ids in [10, 20, 30, 40, 50]:
        meas = generate_measurements(n_ids, pm, p)

        logp = define_log_posterior(meas, mm, em, p[2:5], p[-1])
        tofile = \
            directory + '/posteriors/growth_factor_model_2_free_params_' \
            + str(int(n_ids)) + '.csv'
        run_inference(logp, tofile)
