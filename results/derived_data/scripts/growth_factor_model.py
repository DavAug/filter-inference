import os

import chi
import numpy as np
import pints


def define_data_generating_model():
    # Define mechanistic model
    directory = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    mechanistic_model = chi.SBMLModel(
        directory + '/models/dixit_growth_factor_model.xml')
    mechanistic_model = chi.ReducedMechanisticModel(mechanistic_model)
    mechanistic_model.set_outputs([
        'central.receptor_active_concentration',
        'central.receptor_inactive_concentration'])
    mechanistic_model.fix_parameters({
        'central.receptor_active_amount': 0,
        'central.receptor_inactive_amount': 0,
        'central.ligand_amount': 2,
        'central.size': 1
    })

    # Define error model
    error_models = [
        chi.LogNormalErrorModel(),  # active receptor conc.
        chi.LogNormalErrorModel()]  # inactive receptor conc.

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate']),
        chi.PooledModel(n_dim=3, dim_names=[
            'Deactivation rate', 'Deg. rate (act.)', 'Deg. rate (inact.)']),
        chi.GaussianModel(dim_names=['Production rate']),
        chi.PooledModel(dim_names=['Sigma act.']),
        chi.PooledModel(dim_names=['Sigma inact.'])])
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
        0.05,   # Sigma act.
        0.05]   # Sigma inact.

    return mechanistic_model, predictive_model, parameters


def generate_measurements(predictive_model, parameters):
    # Simulate measurements
    seed = 2
    n_ids = 3000
    times = np.array([5, 10, 15])
    dense_measurements = predictive_model.sample(
        parameters, times, n_samples=n_ids, seed=seed, return_df=False)

    # Keep only one measurement per individual
    n_ids = 1000
    n_times = len(times)
    n_observables = 2
    measurements = np.empty(shape=(n_ids, n_observables, n_times))
    for idt in range(n_times):
        start_ids = idt * n_ids
        end_ids = (idt + 1) * n_ids
        measurements[:, 0, idt] = dense_measurements[0, idt, start_ids:end_ids]
        measurements[:, 1, idt] = dense_measurements[1, idt, start_ids:end_ids]

    return measurements, times


def define_log_posterior(measurements, times, mechanistic_model, sigma):
    # Define population filter log-posterior
    population_filter = chi.GaussianFilter(measurements)
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate'], centered=False),
        chi.PooledModel(n_dim=3, dim_names=[
            'Deactivation rate', 'Deg. rate (act.)', 'Deg. rate (inact.)']),
        chi.GaussianModel(dim_names=['Production rate'], centered=False)])
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(2, 0.5),       # Mean activation rate
        pints.LogNormalLogPrior(-2, 0.5),     # Std. activation rate
        pints.GaussianLogPrior(10, 2),        # deactivation rate
        pints.GaussianLogPrior(0.02, 0.005),  # degradation rate (active)
        pints.GaussianLogPrior(0.3, 0.05),    # degradation rate (inactive)
        pints.GaussianLogPrior(2, 0.5),       # Mean production rate
        pints.LogNormalLogPrior(-2, 0.5))     # Std. production rate
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter, times, mechanistic_model, population_model,
        log_prior, sigma=sigma)

    return log_posterior


def run_inference(log_posterior):
    # Run inference
    seed = 2
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(pints.NoUTurnMCMC)
    n_iterations = 100
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Save samples
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    posterior_samples.to_netcdf(
        directory + '/posteriors/growth_factor_model.nc')


if __name__ == '__main__':
    mm, pm, p = define_data_generating_model()
    meas, times = generate_measurements(pm, p)
    logp = define_log_posterior(meas, times, mm, p[-2:])
    run_inference(logp)
