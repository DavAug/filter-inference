import os

import chi
import numpy as np
import pints


def define_data_generating_model():
    # Define mechanistic model
    directory = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    mechanistic_model = chi.PKPDModel(
        directory + '/models/chaves_tnf_network_model.xml')
    mechanistic_model.set_administration(
        compartment='central', amount_var='tumour_necrosis_factor_amount')
    mechanistic_model.set_dosing_regimen(dose=2 * 100, duration=2, start=0)
    mechanistic_model.set_outputs(['myokit.caspase_3'])
    mechanistic_model = chi.ReducedMechanisticModel(mechanistic_model)
    mechanistic_model.fix_parameters({
        'central.tumour_necrosis_factor_amount': 1,
        'myokit.caspase_3': 0,
        'myokit.caspase_8': 0,
        'myokit.nf_kappaB': 0.26,
        'myokit.nf_kappaB_inhibitor': 0.625,
        'central.size': 1,
        'myokit.activation_rate_c3': 1,
        'myokit.activation_rate_c8_c3': 0.5,
        'myokit.activation_rate_c8_tnf': 0.5,
        'myokit.activation_rate_inf_nf': 0.5,
        'myokit.activation_rate_inf_tnf': 0.5,
        'myokit.activation_rate_nf': 1,
        'myokit.deactivation_rate_c3': 1,
        'myokit.deactivation_rate_c8': 1,
        'myokit.deactivation_rate_inf': 1,
        'myokit.deactivation_rate_nf': 1,
        'myokit.elimination_rate': 100,
        'myokit.half_maximal_effect_concentration_c3_c8': 0.2,
        'myokit.half_maximal_effect_concentration_c8_c3': 0.2,
        'myokit.half_maximal_effect_concentration_c8_tnf': 0.6,
        'myokit.half_maximal_inhibitory_concentration_c8_nf': 0.5,
        'myokit.half_maximal_inhibitory_concentration_inf_tnf': 0.4,
        'myokit.half_maximal_inhibitory_concentration_nf_c3': 0.7,
        'myokit.half_maximal_inhibitory_concentration_nf_inf': 0.4
    })

    # Define error model
    error_model = chi.LogNormalErrorModel()

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(n_dim=2, dim_names=[
            'EC50 C3 NF kappa B', 'EC50 NF inhibitor NF kappa B']),
        chi.PooledModel(n_dim=1, dim_names=['Sigma'])
    ])
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Define model paramters
    parameters = np.array([
        0.2,    # Mean EC50 C3 NF kappa B (non-apoptotic cells)
        0.005,  # Std. EC50 C3 NF kappa B
        0.05,   # Mean shift EC50 C3 NF kappa B (apoptotic cells)
        0.5,    # Mean EC50 NF inhibitor NF kappa B
        0.1,    # Std. EC50 NF inhibitor NF kappa B
        0.05])  # Sigma

    return mechanistic_model, predictive_model, parameters


def generate_measurements(predictive_model, parameters):
    # Simulate measurements
    # (2 subpopulations: 1. non-apoptotic cells; 2. apoptotic cells)
    seed = 2
    n_ids = 5000
    times = [1, 3.5, 7, 10.5, 14]
    dense_measurements = np.empty((1, len(times), n_ids))
    mask = np.array([0, 1, 3, 4, 5])
    p = parameters[mask]
    dense_measurements[:, :, :n_ids//2] = predictive_model.sample(
        p, times, n_samples=n_ids//2, seed=seed, return_df=False)
    p[0] += parameters[2]
    dense_measurements[:, :, n_ids//2:] = predictive_model.sample(
        p, times, n_samples=n_ids//2, seed=seed, return_df=False)

    # Keep only one measurement per individual (shuffle measurements first)
    order = np.arange(n_ids)
    rng = np.random.default_rng(seed+1)
    rng.shuffle(order)
    dense_measurements = dense_measurements[:, :, order]
    n_ids = 1000
    n_times = len(times)
    n_observables = 1
    measurements = np.empty(shape=(n_ids, n_observables, n_times))
    for idt in range(n_times):
        start_ids = idt * n_ids
        end_ids = (idt + 1) * n_ids
        measurements[:, 0, idt] = dense_measurements[0, idt, start_ids:end_ids]

    return measurements, times


def define_log_posterior(measurements, times, mechanistic_model, sigma):
    # Define population filter log-posterior
    n_samples = 100
    population_filter = chi.GaussianFilter(measurements)
    population_model = chi.ComposedPopulationModel([
        chi.HeterogeneousModel(dim_names=['EC50 C3 NF kappa B']),
        chi.GaussianModel(
            dim_names=['EC50 NF inhibitor NF kappa B'], centered=False)
    ])
    log_prior = pints.ComposedLogPrior(*[
        pints.GaussianLogPrior(0.22, 0.5)       # EC50 C3 NF kappa B
        ] * n_samples + [
        pints.GaussianLogPrior(0.6, 0.1),       # Mean EC50 NF inhibitor NF kB
        pints.LogNormalLogPrior(-2, 0.2)        # Std. EC50 NF inhibitor NF kB
    ])
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter, times, mechanistic_model, population_model,
        log_prior, sigma=sigma, n_samples=n_samples)

    return log_posterior


def run_inference(log_posterior):
    # Run inference
    seed = 2
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(pints.NoUTurnMCMC)
    n_iterations = 1500
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Save samples
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    posterior_samples.to_netcdf(
        directory + '/posteriors/tnf_network_model_maximum_entropy_gf.nc')


if __name__ == '__main__':
    mm, pm, p = define_data_generating_model()
    meas, times = generate_measurements(pm, p)
    logp = define_log_posterior(meas, times, mm, p[-1])
    run_inference(logp)
