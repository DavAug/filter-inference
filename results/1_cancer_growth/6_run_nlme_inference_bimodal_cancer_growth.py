import os

import chi
import pandas as pd
import pints

from exponential_growth_model import ExponentialGrowthModel


def define_log_posterior():
    # Import data
    directory = os.path.dirname(os.path.abspath(__file__))
    measurements_df = pd.read_csv(
        directory + '/data/2_bimodal_cancer_growth_data_20.csv')

    # Define hierarchical log-posterior
    mechanistic_model = ExponentialGrowthModel()
    error_model = chi.GaussianErrorModel()
    growth_rate_model = chi.CovariatePopulationModel(
        chi.GaussianModel(centered=False),
        chi.LinearCovariateModel(cov_names=['Aggressive variant']),
        dim_names=['Growth rate'])
    growth_rate_model.set_population_parameters([(0, 0)])
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(
            n_dim=1, dim_names=['Initial count'], centered=False),
        growth_rate_model,
        chi.PooledModel(dim_names=['Sigma'])])
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 3),        # Mean initial condition
        pints.LogNormalLogPrior(2, 1),       # Std. initial condition
        pints.GaussianLogPrior(5, 3),        # Mean exponential growth (slow)
        pints.LogNormalLogPrior(0.5, 1),     # Std. exponential growth
        pints.GaussianLogPrior(5, 3),        # Mean shift growth (fast)
        pints.GaussianLogPrior(0.8, 0.1))    # Sigma
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.set_population_model(population_model)
    problem.set_data(measurements_df)
    problem.set_log_prior(log_prior)

    return problem.get_log_posterior()


def run_inference(log_posterior):
    seed = 2
    controller = chi.SamplingController(log_posterior, seed=seed)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(pints.NoUTurnMCMC)
    n_iterations = 1500
    posterior_samples = controller.run(
        n_iterations=n_iterations, log_to_screen=True)

    # Save results
    warmup = 500
    thinning = 1
    directory = os.path.dirname(os.path.abspath(__file__))
    posterior_samples.sel(
        draw=slice(warmup, n_iterations, thinning)
    ).to_netcdf(
        directory +
        '/posteriors/11_nlme_inference_bimodal_cancer_growth.nc'
    )


if __name__ == '__main__':
    lp = define_log_posterior()
    run_inference(lp)
