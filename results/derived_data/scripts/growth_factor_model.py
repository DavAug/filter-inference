import os

import chi
import numpy as np
import pints


class GrowthFactorModel(chi.MechanisticModel):
    """
    A model that simulates inactive and active receptor concentrations in
    the presence of 2 distinct ligand concentrations.
    """
    def __init__(self, ligand_concs=[2, 10]):
        super(GrowthFactorModel, self).__init__()
        conc1, conc2 = ligand_concs
        self._ligand_concs = ligand_concs

        # Define models
        directory = os.path.dirname(os.getcwd())
        model1 = chi.SBMLModel(
            directory + '/models/dixit_growth_factor_model.xml')
        model1 = chi.ReducedMechanisticModel(model1)
        model1.fix_parameters({
            'central.receptor_active_amount': 0,
            'central.receptor_inactive_amount': 0,
            'central.ligand_amount': 2,
            'central.size': conc1
        })
        model2 = chi.SBMLModel(
            directory + '/models/dixit_growth_factor_model.xml')
        model2 = chi.ReducedMechanisticModel(model2)
        model2.fix_parameters({
            'central.receptor_active_amount': 0,
            'central.receptor_inactive_amount': 0,
            'central.ligand_amount': 2,
            'central.size': conc2
        })

        self._model1 = model1
        self._model2 = model2

    def copy(self):
        return GrowthFactorModel(self._ligand_concs)

    def enable_sensitivities(self, enabled):
        self._model1.enable_sensitivities(enabled)
        self._model2.enable_sensitivities(enabled)

    def has_sensitivities(self):
        return self._model1.has_sensitivities()

    def n_outputs(self):
        return 4

    def n_parameters(self):
        return self._model1.n_parameters()

    def outputs(self):
        outputs = self._model1.outputs()
        names = [
            n + ' ligand conc. ' + str(int(self._ligand_concs[0]))
            for n in outputs]
        names += [
            n + ' ligand conc. ' + str(int(self._ligand_concs[1]))
            for n in outputs]
        return names

    def parameters(self):
        return self._model1.parameters()

    def simulate(self, parameters, times):
        # Simulate model for two concentrations
        s1 = self._model1.simulate(parameters, times)
        s2 = self._model2.simulate(parameters, times)

        # Collect results
        n_times = len(times)
        sim = np.empty(shape=(4, n_times))
        if not self.has_sensitivities():
            sim[:2] = s1
            sim[2:] = s2
            return sim

        # Collect results and sensitivities
        dsim = np.empty(shape=(n_times, 4, self.n_parameters()))
        sim[:2] = s1[0]
        sim[2:] = s2[0]
        dsim[:, :2] = s1[1]
        dsim[:, 2:] = s2[1]
        return sim, dsim


def define_data_generating_model():
    # Define mechanistic model
    mechanistic_model = GrowthFactorModel()

    # Define error model
    error_models = [
        chi.LogNormalErrorModel(), chi.LogNormalErrorModel(),
        chi.LogNormalErrorModel(), chi.LogNormalErrorModel()]

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(dim_names=['Activation rate']),
        chi.PooledModel(n_dim=3, dim_names=[
            'Deactivation rate', 'Deg. rate (act.)', 'Deg. rate (inact.)']),
        chi.GaussianModel(dim_names=['Production rate']),
        chi.PooledModel(n_dim=4, dim_names=[
            'Sigma act. ligand conc. 2',
            'Sigma inact. ligand conc. 2',
            'Sigma act. ligand conc. 10',
            'Sigma inact. ligand conc. 10'])])
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
        0.05,  # Sigma act. ligand conc. 2
        0.05,  # Sigma inact. ligand conc. 2
        0.05,  # Sigma act. ligand conc. 10
        0.05]  # Sigma inact. ligand conc. 10

    return mechanistic_model, predictive_model, parameters


def generate_measurements(predictive_model, parameters):
    # Simulate measurements
    seed = 2
    n_ids = 5000
    times = np.array([1, 5, 10, 15, 20, 25])
    dense_measurements = predictive_model.sample(
        parameters, times, n_samples=n_ids, seed=seed, return_df=False)

    # Keep only one measurement per individual
    n_ids = 100
    n_times = len(times)
    n_observables = 4
    measurements = np.empty(shape=(n_ids, n_observables, n_times))
    for idt in range(n_times):
        start_ids = idt * n_ids
        end_ids = (idt + 1) * n_ids
        measurements[:, 0, idt] = dense_measurements[0, idt, start_ids:end_ids]
        measurements[:, 1, idt] = dense_measurements[1, idt, start_ids:end_ids]
        measurements[:, 2, idt] = dense_measurements[2, idt, start_ids:end_ids]
        measurements[:, 3, idt] = dense_measurements[3, idt, start_ids:end_ids]

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
        log_prior, sigma=sigma, error_on_log_scale=True)

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
        directory + '/posteriors/growth_factor_model_2_regimens.nc')


if __name__ == '__main__':
    mm, pm, p = define_data_generating_model()
    meas, times = generate_measurements(pm, p)
    logp = define_log_posterior(meas, times, mm, p[-4:])
    run_inference(logp)
