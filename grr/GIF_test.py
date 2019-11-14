import unittest

import grr.ThresholdModel_testutil as tmt
from grr.GIF import GIF


class TestGIFSimulateDoesNotCrash_VectorInput(
    tmt.TestSimulateDoesNotCrash_VectorInput, unittest.TestCase
):
    def setUp(self):
        super(TestGIFSimulateDoesNotCrash_VectorInput, self).setUp()
        self.mod = GIF(self.dt)


class TestGIFSimulateDoesNotCrash_ModelStimInput(
    tmt.TestSimulateDoesNotCrash_ModelStimInput, unittest.TestCase
):
    def setUp(self):
        super(TestGIFSimulateDoesNotCrash_ModelStimInput, self).setUp()
        self.mod = GIF(self.dt)


class TestGIFSimulate_StimulusResponse(
    tmt.TestSimulate_StimulusResponse, unittest.TestCase
):
    def setUp(self):
        super(TestGIFSimulate_StimulusResponse, self).setUp()
        self.mod = GIF(self.dt)
        self.mod.El = self.V0


if __name__ == '__main__':
    unittest.main()
