import unittest

import numpy as np
import numpy.testing as npt

from grr.GainEstimator import GainEstimator

class GainEstimatorAttributesTests(unittest.TestCase):
    def setUp(self):
        self.gainEstimator = GainEstimator(dt=0.1)

        self.expectedNumReps = 3
        self.expectedNumSweeps = 6
        self.expectedNumTimeSteps = 7

        # Mock out measuredResponse and inputAmplitudes.
        self.gainEstimator.measuredResponse = np.zeros((self.expectedNumReps, self.expectedNumSweeps, self.expectedNumTimeSteps))
        self.gainEstimator.inputAmplitudes = np.zeros(self.expectedNumSweeps)

    def testRepsProperty(self):
        self.assertEqual(self.gainEstimator.no_reps, self.expectedNumReps)

    def testSweepsProperty(self):
        self.assertEqual(self.gainEstimator.no_sweeps, self.expectedNumSweeps)


class GainEstimatorFitTests(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.gainEstimator = GainEstimator(self.dt)

        self.expectedGain = 5.
        self.expectedIntercept = 3.
        self.expectedGainUncertainty = 2.
        self.expectedInterceptUncertainty = 1.5

        self.numSweeps = 10
        self.numReps = int(1e3)
        self.numTimeSteps = 100
        self.stimulusStartInd = 50
        self.baselineInterval = (0, 5.)
        self.stimulusInterval = (6., 8)

        self.inputAmplitudes = np.linspace(1., 10., self.numSweeps)

    def _drawSampleResponses(self):
        """Draw a random sample of f/I curves from a known distribution.

        Independently sample gains and intercepts from normal distributions
        with parameters specified by instance attributes and use these to draw
        linear f/I curves.

        """
        randomGains = np.random.normal(self.expectedGain, self.expectedGainUncertainty, self.numReps)
        randomIntercepts = np.random.normal(self.expectedIntercept, self.expectedInterceptUncertainty, self.numReps)

        sampleResponses = np.zeros((self.numReps, self.numSweeps, self.numTimeSteps))
        for i in range(self.numReps):
            values = np.polyval((randomGains[i], randomIntercepts[i]), self.inputAmplitudes)
            sampleResponses[i, :, self.stimulusStartInd:] = values[:, np.newaxis]

        return sampleResponses

    def testExpectedGain(self):
        sampleResponses = self._drawSampleResponses()
        self.gainEstimator.fit(sampleResponses, self.inputAmplitudes, self.baselineInterval, self.stimulusInterval)

        npt.assert_allclose(
            self.gainEstimator.gain,
            np.broadcast_to(self.expectedGain, len(self.gainEstimator.gain)),
            atol=0, rtol=0.2,
            err_msg='Estimated gain is not correct.'
        )

    def testExpectedGainUncertainty(self):
        sampleResponses = self._drawSampleResponses()
        self.gainEstimator.fit(sampleResponses, self.inputAmplitudes, self.baselineInterval, self.stimulusInterval)

        npt.assert_allclose(
            self.gainEstimator.gainUncertainty,
            np.broadcast_to(self.expectedGainUncertainty, len(self.gainEstimator.gainUncertainty)),
            atol=0, rtol=0.2,
            err_msg='Estimated uncertainty on gain is not correct.'
        )

    def testExpectedIntercept(self):
        sampleResponses = self._drawSampleResponses()
        self.gainEstimator.fit(sampleResponses, self.inputAmplitudes, self.baselineInterval, self.stimulusInterval)

        npt.assert_allclose(
            self.gainEstimator.intercept,
            np.broadcast_to(self.expectedIntercept, len(self.gainEstimator.intercept)),
            atol=0, rtol=0.2,
            err_msg='Estimated intercept is not correct.'
        )

    def testExpectedInterceptUncertainty(self):
        sampleResponses = self._drawSampleResponses()
        self.gainEstimator.fit(sampleResponses, self.inputAmplitudes, self.baselineInterval, self.stimulusInterval)

        npt.assert_allclose(
            self.gainEstimator.interceptUncertainty,
            np.broadcast_to(self.expectedInterceptUncertainty, len(self.gainEstimator.interceptUncertainty)),
            atol=0, rtol=0.2,
            err_msg='Estimated uncertainty on intercept is not correct.'
        )

if __name__ == '__main__':
    np.random.seed(42)
    unittest.main()

