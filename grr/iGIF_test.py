import unittest

from grr.iGIF import iGIF_Na, iGIF_NP
from grr.ModelStimulus import ModelStimulus
from ezephys import stimtools


class TestiGIF_NPSimulateDoesNotCrash_VectorInput(unittest.TestCase):

    def setUp(self):
        self.dt = 0.1
        self.mod = iGIF_NP(self.dt)
        self.V0 = -70.
        self.spkTimes = [50., 700.]

        self.stimulusDuration = 1e3  # Duration in ms.
        self.vectorStimulus = stimtools.SinStimulus(
            0.05, 0.05, 10, self.stimulusDuration, self.dt
        )

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


class TestiGIF_NPSimulateDoesNotCrash_ModelStimInput(unittest.TestCase):

    def setUp(self):
        self.dt = 0.1
        self.mod = iGIF_NP(self.dt)
        self.V0 = -70.
        self.spkTimes = [50., 700.]

        self.stimulusDuration = 1e3  # Duration in ms.
        self.vectorStimulus = stimtools.SinStimulus(
            0.05, 0.05, 10, self.stimulusDuration, self.dt
        )

        self.modelStimulus = ModelStimulus(self.dt)

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


class TestiGIF_NaSimulateDoesNotCrash_VectorInput(unittest.TestCase):

    def setUp(self):
        self.dt = 0.1
        self.mod = iGIF_Na(self.dt)
        self.V0 = -70.
        self.spkTimes = [50., 700.]

        self.stimulusDuration = 1e3  # Duration in ms.
        self.vectorStimulus = stimtools.SinStimulus(
            0.05, 0.05, 10, self.stimulusDuration, self.dt
        )

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


class TestiGIF_NaSimulateDoesNotCrash_ModelStimInput(unittest.TestCase):

    def setUp(self):
        self.dt = 0.1
        self.mod = iGIF_Na(self.dt)
        self.V0 = -70.
        self.spkTimes = [50., 700.]

        self.stimulusDuration = 1e3  # Duration in ms.
        self.vectorStimulus = stimtools.SinStimulus(
            0.05, 0.05, 10, self.stimulusDuration, self.dt
        )

        self.modelStimulus = ModelStimulus(self.dt)

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


if __name__ == '__main__':
    unittest.main()
