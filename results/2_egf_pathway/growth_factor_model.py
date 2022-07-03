import os

import chi
import numpy as np


class GrowthFactorModel(chi.MechanisticModel):
    """
    A model that simulates inactive and active receptor concentrations in
    the presence of 2 distinct ligand concentrations.
    """
    def __init__(self, ligand_concs=[2, 10]):
        super(GrowthFactorModel, self).__init__()
        ligand_concs = np.array(ligand_concs)
        if np.any(ligand_concs) < 0:
            raise ValueError(
                'ligand_concs is invalid. Ligand concentrations have to be '
                'greater or equal to zero.'
            )
        if len(ligand_concs) != 2:
            raise ValueError(
                'ligand_concs is invalid. The ligand concentrations need to '
                'be an array-like object of shape (2,).')
        conc1, conc2 = ligand_concs
        self._ligand_concs = ligand_concs

        # Define models
        directory = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model1 = chi.SBMLModel(
            directory + '/models/dixit_growth_factor_model.xml')
        model1 = chi.ReducedMechanisticModel(model1)
        model1.fix_parameters({
            'central.receptor_active_amount': 0,
            'central.receptor_inactive_amount': 0,
            'central.ligand_amount': conc1,
            'central.size': 1
        })
        model2 = chi.SBMLModel(
            directory + '/models/dixit_growth_factor_model.xml')
        model2 = chi.ReducedMechanisticModel(model2)
        model2.fix_parameters({
            'central.receptor_active_amount': 0,
            'central.receptor_inactive_amount': 0,
            'central.ligand_amount': conc2,
            'central.size': 1
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
