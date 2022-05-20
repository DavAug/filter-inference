import os
import pickle
import timeit

import chi
import numpy as np
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
        'central.size': 1,
        'myokit.degradation_rate_active_receptor': 0.015,
        'myokit.degradation_rate_inactive_receptor': 0.25
    })

    # Define error model
    error_model = chi.LogNormalErrorModel()

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate']),
        chi.PooledModel(n_dim=1, dim_names=['Deactivation rate']),
        chi.GaussianModel(dim_names=['Production rate']),
        chi.PooledModel(n_dim=1, dim_names=['Sigma act.'])])
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Define model paramters
    parameters = [
        1.7,    # Mean activation rate
        0.05,   # Std. activation rate
        8,      # deactivation rate
        1.7,    # Mean production rate
        0.05,   # Std. production rate
        0.05]   # Sigma act.

    return mechanistic_model, error_model, predictive_model, parameters


def generate_measurements(n_ids_per_t, predictive_model, parameters):
    # Simulate dense measurements
    seed = 2
    times = np.array([1, 5, 10, 15, 20, 25])
    n_times = len(times)
    n_ids = n_ids_per_t * n_times
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


def define_log_posterior(
        measurements, times, mechanistic_model, sigma):
    population_filter = chi.GaussianFilter(observations=measurements)
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate'], centered=False),
        chi.PooledModel(n_dim=1),
        chi.GaussianModel(dim_names=['Production rate'], centered=False)])
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(2, 0.5),       # Mean activation rate
        pints.LogNormalLogPrior(-2, 0.5),     # Std. activation rate
        pints.GaussianLogPrior(10, 2),        # deactivation rate
        pints.GaussianLogPrior(2, 0.5),       # Mean production rate
        pints.LogNormalLogPrior(-2, 0.5))     # Std. production rate
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter=population_filter, times=times,
        mechanistic_model=mechanistic_model, population_model=population_model,
        log_prior=log_prior, sigma=sigma, error_on_log_scale=True)

    return log_posterior


def estimate_evaluation_time(log_posterior):
    n = log_posterior.n_parameters()
    test_parameters = np.ones(n)
    test_parameters[n//2:] += 0.1
    # Evaluate once, so sensitivities are switched on
    log_posterior.evaluateS1(test_parameters)

    number = 10
    repeats = 10
    run_time = timeit.repeat(
        'logp.evaluateS1(p)',
        globals=dict(logp=log_posterior, p=test_parameters),
        number=number, repeat=repeats)

    return np.min(run_time) / number


def run_inference(log_posterior, tofile):
    # Run inference
    seed = 3
    n_chains = 3
    n_iterations = 1000
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
    n_ids_per_t = [15, 20, 25, 30, 35, 40, 45, 50, 55]
    mm, em, pm, p = define_data_generating_model()
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Estimate evaluation time of log-posterior
    times = []
    for n_ids in n_ids_per_t:
        print('Estimating evaluation time for n_per_t=%d' % n_ids)
        meas, t = generate_measurements(n_ids, pm, p)
        logp = define_log_posterior(meas, t, mm, p[-1])
        times += [estimate_evaluation_time(logp)]
    tofile = \
        directory + '/posteriors/growth_factor_model_3_fixed_params_pfi_' \
        'eval_time.p'
    pickle.dump([n_ids_per_t, times], open(tofile, 'wb'))

    # Estimate number of evaluations for inference
    for n_ids in n_ids_per_t:
        meas, t = generate_measurements(n_ids, pm, p)
        logp = define_log_posterior(meas, t, mm, p[-1])
        tofile = \
            directory + '/posteriors/growth_factor_model_3_fixed_params_pfi_' \
            + str(int(n_ids)) + '.csv'
        run_inference(logp, tofile)
