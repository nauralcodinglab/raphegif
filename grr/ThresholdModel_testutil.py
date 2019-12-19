import unittest

import numpy as np

from grr.ThresholdModel import constructMedianModel
from grr.ModelStimulus import ModelStimulus
from ezephys import stimtools


class TestSimulateDoesNotCrash_VectorInput(object):

    def setUp(self):
        self.dt = 0.1
        self.V0 = -70.
        self.spkTimes = [50., 700.]

        self.stimulusDuration = 1e3  # Duration in ms.
        self.vectorStimulus = stimtools.SinStimulus(
            0.05, 0.05, 10, self.stimulusDuration, self.dt
        )

        # Assign a model to `mod` attribute, eg
        # self.mod = iGIF_NP(self.dt)

    def test_simulateDeterministic(self):
        """Test whether simulateDeterministic raises an error for valid vector input."""
        self.mod.simulateDeterministic_forceSpikes(
            self.vectorStimulus.command,
            self.V0,
            self.spkTimes
        )

    def test_simulate(self):
        """Test whether simulate() raises an error for valid input."""
        self.mod.simulate(self.vectorStimulus.command, self.V0)


class TestSimulateDoesNotCrash_ModelStimInput(object):

    def setUp(self):
        self.dt = 0.1
        self.V0 = -70.
        self.spkTimes = [50., 700.]

        self.stimulusDuration = 1e3  # Duration in ms.
        self.vectorStimulus = stimtools.SinStimulus(
            0.05, 0.05, 10, self.stimulusDuration, self.dt
        )

        self.modelStimulus = ModelStimulus(self.dt)

        # Assign a model to `mod` attribute, eg
        # self.mod = iGIF_NP(self.dt)

    def test_coerceInputToModelStimulus(self):
        I = self.mod._coerceInputToModelStimulus(self.modelStimulus)
        self.assertTrue(
            isinstance(I, ModelStimulus),
            'Expected result of _coerceInputToModelStimulus to be instance of '
            'type ModelStimulus, got {} instead.'.format(type(I))
        )
        self.assertEqual(I.timesteps, self.modelStimulus.timesteps)

        I = self.mod._coerceInputToModelStimulus(self.vectorStimulus.command)
        self.assertTrue(
            isinstance(I, ModelStimulus),
            'Expected result of _coerceInputToModelStimulus to be instance of '
            'type ModelStimulus, got {} instead.'.format(type(I))
        )
        self.assertEqual(I.timesteps, self.vectorStimulus.no_timesteps)

    def test_simulateDeterministic_currentInput(self):
        """Test whether simulateDeterministic raises an error for valid current only input."""
        self.modelStimulus.appendCurrents(self.vectorStimulus.command)
        self.mod.simulateDeterministic_forceSpikes(
            self.modelStimulus,
            self.V0,
            self.spkTimes
        )

    def test_simulate_currentInput(self):
        """Ensure no error raised for valid current only input."""
        self.modelStimulus.appendCurrents(self.vectorStimulus.command)
        self.mod.simulate(
            self.modelStimulus,
            self.V0
        )

    def test_simulateDeterministic_conductanceInput(self):
        """Ensure no error raised for valid conductance only input."""
        self.modelStimulus.appendConductances(self.vectorStimulus.command, [-70.])
        self.mod.simulateDeterministic_forceSpikes(
            self.modelStimulus,
            self.V0,
            self.spkTimes
        )

    def test_simulate_conductanceInput(self):
        """Ensure no error raised for valid conductance only input."""
        self.modelStimulus.appendConductances(self.vectorStimulus.command, [-70.])
        self.mod.simulate(
            self.modelStimulus,
            self.V0
        )


class TestSimulate_StimulusResponse(unittest.TestCase):

    rtol = 1e-5
    atol = 1e-8

    def setUp(self):
        self.dt = 0.1
        self.V0 = -70.
        self.spkTimes = []

        self.stimulusDuration = 1e3  # Duration in ms.
        self.vectorStimulus = stimtools.SinStimulus(
            0., 0.01, 10, self.stimulusDuration, self.dt
        )
        self.modelStimulus = ModelStimulus(self.dt)

        # Assign a model to `mod` attribute, eg
        # self.mod = iGIF_NP(self.dt)
        # self.mod.El = self.V0

    def test_stableWithZeroCurrent(self):
        self.modelStimulus.appendCurrents(
            np.zeros_like(self.vectorStimulus.command)
        )
        t, V, _ = self.mod.simulateDeterministic_forceSpikes(
            self.modelStimulus, self.V0, self.spkTimes
        )
        self.assertTrue(
            np.allclose(self.V0, V, self.rtol, self.atol),
            'Voltage drifts with zero current input.'
        )

    def test_currentInputChangesVoltage(self):
        self.modelStimulus.appendCurrents(self.vectorStimulus.command)
        t, V, _ = self.mod.simulateDeterministic_forceSpikes(
            self.modelStimulus, self.V0, self.spkTimes
        )
        self.assertFalse(
            np.allclose(self.V0, V, self.rtol, self.atol),
            'Current input does not affect subthreshold voltage.'
        )

    def test_stableWithZeroConductance(self):
        self.modelStimulus.appendConductances(
            np.zeros_like(self.vectorStimulus.command), [self.V0 - 10.]
        )
        t, V, _ = self.mod.simulateDeterministic_forceSpikes(
            self.modelStimulus, self.V0, self.spkTimes
        )
        self.assertTrue(
            np.allclose(self.V0, V, self.rtol, self.atol),
            'Voltage drifts with zero conductance input.'
        )

    def test_stableWithConductanceAtReversal(self):
        self.modelStimulus.appendConductances(
            self.vectorStimulus.command, [self.V0]
        )
        t, V, _ = self.mod.simulateDeterministic_forceSpikes(
            self.modelStimulus, self.V0, self.spkTimes
        )
        self.assertTrue(
            np.allclose(self.V0, V, self.rtol, self.atol),
            'Voltage drifts with conductance at reversal.'
        )

    def test_conductanceInputChangesVoltage(self):
        self.modelStimulus.appendConductances(
            self.vectorStimulus.command, [self.V0 - 10.]
        )
        t, V, _ = self.mod.simulateDeterministic_forceSpikes(
            self.modelStimulus, self.V0, self.spkTimes
        )
        self.assertFalse(
            np.allclose(self.V0, V, self.rtol, self.atol),
            'Conductance input does not affect voltage.'
        )


class TestConstructMedianModel(object):
    def setUp(self):
        self._setModelType()
        self._instantiateRandomModels()
        self._instantiateExpectedModel()

    def _setModelType(self):
        raise NotImplementedError('Must be implemented by subclasses.')
        # self.modelType = <someModelType>

    def _instantiateRandomModels(self):
        noOfModels = 10
        self.randomCoefficients = {
            paramName: np.random.normal(size=(noOfModels))
            for paramName in self.modelType.scalarParameters
        }
        self.models = [self.modelType() for i in range(noOfModels)]

        for paramName in self.modelType.scalarParameters:
            for i, mod in enumerate(self.models):
                setattr(mod, paramName, self.randomCoefficients[paramName][i])

    def _instantiateExpectedModel(self):
        self.expectedModel = self.modelType()
        for paramName in self.randomCoefficients:
            setattr(
                self.expectedModel,
                paramName,
                np.median(self.randomCoefficients[paramName]),
            )

    def testSetsCorrectScalarParameters(self):
        medianModel = constructMedianModel(self.modelType, self.models)
        self._assertEqualScalarParameters(medianModel, self.expectedModel)

    @staticmethod
    def _assertEqualScalarParameters(mod1, mod2):
        assert mod1.scalarParameters == mod2.scalarParameters
        for paramName in mod1.scalarParameters:
            assert getattr(mod1, paramName, np.nan) == getattr(
                mod2, paramName, np.nan
            )


if __name__ == '__main__':
    print(
        'Not intended to be run as a test suite. ThresholdModel test suites '
        'should import and subclass these tests.'
    )
