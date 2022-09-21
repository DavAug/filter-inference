import os

import chi
import numpy as np
import pandas as pd
import pints

from growth_factor_model import GrowthFactorModel


def define_log_posterior():
    # Import data
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df = pd.read_csv(
        directory + '/data/1_egf_pathway_data.csv')

    # Define model
    mechanistic_model = GrowthFactorModel()
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate'], centered=False),
        chi.PooledModel(n_dim=3, dim_names=[
            'Deactivation rate', 'Deg. rate (act.)', 'Deg. rate (inact.)']),
        chi.GaussianModel(dim_names=['Production rate'], centered=False)])

    # Reshape data to numpy array of shape (n_ids_per_t, n_output, n_times)
    n_ids_per_t = 100
    outputs = [
        'Act. receptor conc. Amount 1', 'Inact. receptor conc. Amount 1',
        'Act. receptor conc. Amount 2', 'Inact. receptor conc. Amount 2']
    n_outputs = 4
    times = np.sort(measurements_df.Time.unique())
    measurements = np.empty(shape=(n_ids_per_t, n_outputs, len(times)))
    for ido, output in enumerate(outputs):
        temp = measurements_df[measurements_df.Observable == output]
        for idt, time in enumerate(times):
            mask = temp.Time == time
            measurements[:, ido, idt] = temp[mask].Value.values

    # Define log-posterior
    population_filter = chi.GaussianFilter(measurements)
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
        log_prior, sigma=[0.05, 0.05, 0.05, 0.05], error_on_log_scale=True)

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
        '/posteriors/filter_inference_egf_pathway.nc'
    run_inference(lp, filename)
