import os
import pickle
import timeit

import chi
import numpy as np
import pandas as pd
import pints


class ToyExponentialGrowthModel(chi.MechanisticModel):
    """
    A toy exponential growth model.
    """
    def __init__(self):
        super(ToyExponentialGrowthModel, self).__init__()

        self._has_sensitivities = False

    def enable_sensitivities(self, enabled, parameter_names=None):
        r"""
        Enables the computation of the model output sensitivities to the model
        parameters if set to ``True``.

        The sensitivities of the model outputs are defined as the partial
        derviatives of the ouputs :math:`\bar{y}` with respect to the model
        parameters :math:`\psi`

        .. math:
            \frac{\del \bar{y}}{\del \psi}.

        :param enabled: A boolean flag which enables (``True``) / disables
            (``False``) the computation of sensitivities.
        :type enabled: bool
        """
        self._has_sensitivities = bool(enabled)

    def has_sensitivities(self):
        """
        Returns a boolean indicating whether sensitivities have been enabled.
        """
        return self._has_sensitivities

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return 1

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return 2

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return ['Count']

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        return ['Initial count', 'Growth rate']

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs (and optionally
        the sensitivites) for the specified parameters and times.

        The model outputs are returned as a 2 dimensional NumPy array of shape
        ``(n_outputs, n_times)``. If sensitivities are enabled, a tuple is
        returned with the NumPy array of the model outputs and a NumPy array of
        the sensitivities of shape ``(n_times, n_outputs, n_parameters)``.

        :param parameters: An array-like object with values for the model
            parameters.
        :type parameters: list, numpy.ndarray
        :param times: An array-like object with time points at which the output
            values are returned.
        :type times: list, numpy.ndarray

        :rtype: np.ndarray of shape (n_outputs, n_times) or
            (n_times, n_outputs, n_parameters)
        """
        y0, growth_rate = parameters
        times = np.asarray(times)

        # Solve model
        y = y0 * np.exp(growth_rate * times)

        if not self._has_sensitivities:
            return y[np.newaxis, :]

        sensitivities = np.empty(shape=(len(times), 1, 2))
        sensitivities[:, 0, 0] = np.exp(growth_rate * times)
        sensitivities[:, 0, 1] = times * y

        return y[np.newaxis, :], sensitivities


def define_data_generating_model():
    # Define mechanistic model
    mechanistic_model = ToyExponentialGrowthModel()

    # Define error model
    error_model = chi.GaussianErrorModel()

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(n_dim=2, dim_names=['Initial count', 'Growth rate']),
        chi.PooledModel(dim_names=['Sigma'])])
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Define model paramters
    parameters = [
        10,    # Mean initial condition
        2,     # Mean exponential growth
        1,     # Std. initial condition
        0.5,   # Std. exponential growth
        0.8]   # Sigma

    return mechanistic_model, error_model, predictive_model, parameters


def generate_measurements(n_ids_per_t, predictive_model, parameters, seed):
    # Simulate dense measurements
    n_times = 6
    n_ids = n_ids_per_t * n_times
    times = np.linspace(0, 0.6, num=n_times)
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
        'Observable': 'Count',
        'Value': measurements.flatten(),
        'ID': ids,
        'Time': np.broadcast_to(
            times[np.newaxis, :], (n_ids_per_t, n_times)).flatten()
    })

    return measurements_df


def define_log_posterior(measurements, mechanistic_model, error_model):
    population_model = chi.ComposedPopulationModel([
        chi.GaussianModel(
            n_dim=2, dim_names=['Initial count', 'Growth rate'],
            centered=False),
        chi.PooledModel(dim_names=['Sigma'])])
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(9, 3),        # Mean initial condition
        pints.GaussianLogPrior(5, 3),        # Mean exponential growth
        pints.LogNormalLogPrior(-0.1, 1),    # Std. initial condition
        pints.LogNormalLogPrior(-1, 1),      # Std. exponential growth
        pints.LogNormalLogPrior(-1, 1))      # Sigma
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.set_population_model(population_model)
    problem.set_data(measurements)
    problem.set_log_prior(log_prior)
    log_posterior = problem.get_log_posterior()

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
    seed = 4
    n_chains = 1
    n_iterations = 1000
    controller = chi.SamplingController(log_posterior, seed)
    controller.set_parallel_evaluation(False)
    controller.set_n_runs(n_chains)
    samples = controller.run(n_iterations, log_to_screen=True)
    samples.to_netcdf(tofile)


if __name__ == '__main__':
    n_ids_per_t = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    mm, em, pm, p = define_data_generating_model()
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Estimate evaluation time of log-posterior
    times = []
    for seed, n_ids in enumerate(n_ids_per_t):
        meas = generate_measurements(n_ids, pm, p, seed)
        logp = define_log_posterior(meas, mm, em)
        times += [estimate_evaluation_time(logp)]
    tofile = \
        directory + '/posteriors/hierarchical_exponential_growth_model_' \
        'eval_time.p'
    pickle.dump([n_ids_per_t, times], open(tofile, 'wb'))

    # Estimate number of evaluations for inference
    for seed, n_ids in enumerate(n_ids_per_t):
        meas = generate_measurements(n_ids, pm, p, seed)
        logp = define_log_posterior(meas, mm, em)
        tofile = \
            directory + '/posteriors/hierarchical_exponential_growth_model_' \
            + str(int(n_ids)) + '_samples.nc'
        run_inference(logp, tofile)
