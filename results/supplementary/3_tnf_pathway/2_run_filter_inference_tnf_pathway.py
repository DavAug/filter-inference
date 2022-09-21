import os

import chi
import numpy as np
import pandas as pd
import pints


def define_log_posterior():
    # Import data
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df = pd.read_csv(
        directory + '/data/1_tnf_pathway_data.csv')

    # Define mechanistic model
    directory = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
        'myokit.elimination_rate': 100
    })

    # Define population model
    ec50_c3_nf_model = chi.CovariatePopulationModel(
        chi.GaussianModel(centered=False),
        chi.LinearCovariateModel(cov_names=['Shift apoptotic']),
        dim_names=['EC50 C3 NF kappa B'])
    ec50_c3_nf_model.set_population_parameters([(0, 0)])
    population_model = chi.ComposedPopulationModel([
        chi.PooledModel(
            n_dim=3, dim_names=['EC50 C3 C8', 'EC50 C8 C3', 'EC50 C8 TNF']),
        ec50_c3_nf_model,
        chi.PooledModel(dim_names=['EC50 C8 NF kappa B']),
        chi.GaussianModel(
            dim_names=['EC50 NF inhibitor NF kappa B'], centered=False),
        chi.PooledModel(n_dim=3, dim_names=[
            'EC50 NF inhibitor TNF',
            'EC50 NF kappa B C3',
            'EC50 NF kappa B NF inhibitor'])
    ])

    # Reshape data to numpy array of shape (n_ids_per_t, n_output, n_times)
    n_ids_per_t = 500
    output = 'Caspase 3 concentration'
    n_outputs = 1
    times = np.sort(measurements_df.dropna().Time.unique())
    measurements = np.empty(shape=(n_ids_per_t, n_outputs, len(times)))
    temp = measurements_df[measurements_df.Observable == output]
    for idt, time in enumerate(times):
        mask = temp.Time == time
        measurements[:, 0, idt] = temp[mask].Value.values

    # Define log-posterior
    n_samples = 100
    covariates = np.zeros((n_samples, 1))
    covariates[n_samples//2:] = 1
    population_filter = chi.GaussianMixtureFilter(measurements, n_kernels=2)
    log_prior = pints.ComposedLogPrior(*[
        pints.GaussianLogPrior(0.3, 0.07),  # EC50 C3 C8
        pints.GaussianLogPrior(0.3, 0.07),  # EC50 C8 C3
        pints.GaussianLogPrior(0.5, 0.1),   # EC50 C8 TNF
        pints.GaussianLogPrior(0.4, 0.07),  # Mean EC50 C3 NF kappa B no ap.
        pints.GaussianLogPrior(0.01, 2E-3),  # Std. EC50 C3 NF kappa B
        pints.GaussianLogPrior(0.1, 0.02),  # Mean shift EC50 C3 NF kappa B ap.
        pints.GaussianLogPrior(0.5, 0.1),   # EC50 C8 NF kappa B
        pints.GaussianLogPrior(0.6, 0.1),   # Mean EC50 NF inhibitor NF kappa B
        pints.GaussianLogPrior(0.1, 2E-2),  # Std. EC50 NF inhibitor NF kappa B
        pints.GaussianLogPrior(0.5, 0.1),   # EC50 NF inhibitor TNF
        pints.GaussianLogPrior(0.6, 0.1),   # EC50 NF kappa B C3
        pints.GaussianLogPrior(0.5, 0.1)    # EC50 NF kappa B NF inhibitor
    ])
    log_posterior = chi.PopulationFilterLogPosterior(
        population_filter, times, mechanistic_model, population_model,
        log_prior, sigma=[0.05], error_on_log_scale=True,
        covariates=covariates, n_samples=n_samples)

    return log_posterior


def run_inference(log_posterior, filename):
    seed = 3
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(True)
    controller.set_sampler(pints.NoUTurnMCMC)
    n_iterations = 1500
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Save results
    warmup = 500
    thinning = 1
    posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning)).to_netcdf(filename)


if __name__ == '__main__':
    directory = os.path.dirname(os.path.abspath(__file__))
    lp = define_log_posterior()
    filename = \
        directory + \
        '/posteriors/filter_inference_tnf_pathway.nc'
    run_inference(lp, filename)
