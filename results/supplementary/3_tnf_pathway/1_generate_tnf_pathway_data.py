import os

import chi
import numpy as np
import pandas as pd


def define_data_generating_model():
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

    # Define error model
    error_model = chi.LogNormalErrorModel()

    # Define population model
    ec50_c3_nf_model = chi.CovariatePopulationModel(
        chi.GaussianModel(),
        chi.LinearCovariateModel(cov_names=['Shift apoptotic']),
        dim_names=['EC50 C3 NF kappa B'])
    ec50_c3_nf_model.set_population_parameters([(0, 0)])
    population_model = chi.ComposedPopulationModel([
        chi.PooledModel(
            n_dim=3, dim_names=['EC50 C3 C8', 'EC50 C8 C3', 'EC50 C8 TNF']),
        ec50_c3_nf_model,
        chi.PooledModel(dim_names=['EC50 C8 NF kappa B']),
        chi.GaussianModel(dim_names=['EC50 NF inhibitor NF kappa B']),
        chi.PooledModel(n_dim=4, dim_names=[
            'EC50 NF inhibitor TNF',
            'EC50 NF kappa B C3',
            'EC50 NF kappa B NF inhibitor',
            'Sigma'])
    ])

    # Compose data-generating model
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Define model paramters
    parameters = np.array([
        0.2,    # EC50 C3 C8
        0.2,    # EC50 C8 C3
        0.6,    # EC50 C8 TNF
        0.2,    # Mean EC50 C3 NF kappa B (non-apoptotic cells)
        0.005,  # Std. EC50 C3 NF kappa B
        0.05,   # Mean shift EC50 C3 NF kappa B (apoptotic cells)
        0.5,    # EC50 C8 NF kappa B
        0.5,    # Mean EC50 NF inhibitor NF kappa B
        0.1,    # Std. EC50 NF inhibitor NF kappa B
        0.4,    # EC50 NF inhibitor TNF
        0.7,    # EC50 NF kappa B C3
        0.4,    # EC50 NF kappa B NF inhibitor
        0.05    # Sigma
    ])
    return predictive_model, parameters


def generate_data(predictive_model, parameters):
    # Simulate measurements
    seed = 2
    n_ids = 5000
    times = np.array([1, 3.5, 7, 10.5, 14])
    covariates = [0, 1]
    dense_measurements = []
    for idc, cov in enumerate(covariates):
        dense_measurements += [predictive_model.sample(
            parameters, times, n_samples=n_ids, seed=seed+idc,
            return_df=False, covariates=[cov])]

    # Keep only one measurement per individual
    n_ids_per_t = 250
    n_times = len(times)
    n_observables = 1
    measurements = np.empty(shape=(n_ids_per_t*2, n_observables, n_times))
    for idm, meas in enumerate(dense_measurements):
        for idt in range(n_times):
            start_ids = idt * n_ids_per_t
            end_ids = (idt + 1) * n_ids_per_t
            measurements[idm*n_ids_per_t:(idm+1)*n_ids_per_t, 0, idt] = \
                meas[0, idt, start_ids:end_ids]

    # Format data as dataframe
    n_times = len(times)
    ids = np.arange(1, len(measurements.flatten()) + 1)
    measurements_df = pd.DataFrame({
        'Observable': 'Caspase 3 concentration',
        'Value': measurements.flatten(),
        'ID': ids,
        'Time': np.broadcast_to(
            times[np.newaxis, :], (2*n_ids_per_t, n_times)).flatten(),
    })
    covariates = np.zeros(shape=(n_ids_per_t*2, n_observables, n_times))
    covariates[n_ids_per_t:] = 1
    measurements_df = pd.concat([measurements_df, pd.DataFrame({
        'Observable': 'Apoptotic',
        'Value': covariates.flatten(),
        'ID': ids})
    ])

    return measurements_df


if __name__ == '__main__':
    m, p = define_data_generating_model()
    data = generate_data(m, p)

    # Export data to file
    directory = os.path.dirname(os.path.abspath(__file__))
    data.to_csv(directory + '/data/1_tnf_pathway_data.csv', index=False)
