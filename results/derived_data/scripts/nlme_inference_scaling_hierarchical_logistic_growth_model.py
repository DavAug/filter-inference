import os
import pickle
import timeit

import chi
import numpy as np
import pandas as pd
import pints


class ToyLogisticGrowthModel(chi.MechanisticModel):
    """
    A toy logistic growth model.
    """
    def __init__(self):
        super(ToyLogisticGrowthModel, self).__init__()

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
        return 3

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return ['Count']

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        return ['Initial count', 'Growth rate', 'Capacity']

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
        y0, growth_rate, capacity = parameters
        times = np.asarray(times)

        # Solve model
        y = \
            capacity * y0 \
            / (y0 + (capacity - y0) * np.exp(-growth_rate * times))

        if not self._has_sensitivities:
            return y[np.newaxis, :]

        sensitivities = np.empty(shape=(len(times), 1, 3))
        sensitivities[:, 0, 0] = \
            capacity \
            / (y0 + (capacity - y0) * np.exp(-growth_rate * times)) \
            - (capacity * y0) \
            / (y0 + (capacity - y0) * np.exp(-growth_rate * times)) ** 2 \
            * (1 - np.exp(-growth_rate * times))
        sensitivities[:, 0, 1] = \
            (capacity - y0) * y * times * np.exp(-growth_rate * times) \
            / (y0 + (capacity - y0) * np.exp(-growth_rate * times))
        sensitivities[:, 0, 2] = \
            y0 / (y0 + (capacity - y0) * np.exp(-growth_rate * times)) \
            - (capacity * y0) * np.exp(-growth_rate * times) \
            / (y0 + (capacity - y0) * np.exp(-growth_rate * times)) ** 2

        return y[np.newaxis, :], sensitivities


def define_data_generating_model():
    # Define mechanistic model
    mechanistic_model = ToyLogisticGrowthModel()

    # Define error model
    error_model = chi.LogNormalErrorModel()

    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.LogNormalModel(
            n_dim=2, dim_names=['Initial count', 'Growth rate']),
        chi.GaussianModel(dim_names=['Capacity']),
        chi.PooledModel(dim_names=['Sigma'])])
    predictive_model = chi.PredictiveModel(mechanistic_model, error_model)
    predictive_model = chi.PopulationPredictiveModel(
        predictive_model, population_model)

    # Define model paramters
    parameters = [
        1.8,   # Log mean initial condition
        2.3,   # Log mean exponential growth
        0.4,   # Log std. initial condition
        0.5,   # Log std. exponential growth
        110,   # Mean capacity
        5,     # Std. capacity
        0.08]  # Sigma

    return (
        mechanistic_model, error_model, predictive_model, population_model,
        parameters)


def generate_measurements(n_ids_per_t, predictive_model, parameters):
    # Simulate dense measurements
    seed = 2
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


def define_log_posterior(measurements, mechanistic_model, error_model, sigma):
    population_model = chi.ComposedPopulationModel([
        chi.LogNormalModel(
            n_dim=2, dim_names=['Initial count', 'Growth rate'],
            centered=False),
        chi.GaussianModel(dim_names=['Capacity'], centered=False)])
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(1, 3),        # Log mean initial condition
        pints.GaussianLogPrior(3, 3),        # Log mean exponential growth
        pints.LogNormalLogPrior(-0.1, 0.5),  # Log std. initial condition
        pints.LogNormalLogPrior(-0.1, 0.5),  # Log std. exponential growth
        pints.GaussianLogPrior(100, 10),     # Mean capacity
        pints.LogNormalLogPrior(1.5, 0.5))   # Std. capacity
    problem = chi.ProblemModellingController(mechanistic_model, error_model)
    problem.fix_parameters({'Sigma log': sigma})
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
    n_ids_per_t = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    mm, em, pm, pop_m, p = define_data_generating_model()
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Estimate evaluation time of log-posterior
    times = []
    for n_ids in n_ids_per_t:
        meas = generate_measurements(n_ids, pm, p)
        logp = define_log_posterior(meas, mm, em, p[-1])
        times += [estimate_evaluation_time(logp)]
    tofile = \
        directory + '/posteriors/hierarchical_logistic_growth_model_' \
        'eval_time.p'
    pickle.dump([n_ids_per_t, times], open(tofile, 'wb'))

    # Estimate number of evaluations for inference
    for n_ids in n_ids_per_t:
        meas = generate_measurements(n_ids, pm, p)
        logp = define_log_posterior(meas, mm, em, p[-1])
        tofile = \
            directory + '/posteriors/hierarchical_logistic_growth_model_' \
            + str(int(n_ids)) + '.csv'
        run_inference(logp, tofile)
