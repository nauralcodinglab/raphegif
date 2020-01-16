import unittest

import grr.ThresholdModel_testutil as tmt
from grr.iGIF import iGIF_NP, iGIF_Na


# iGIF_NP tests.
class TestiGIF_NPSimulateDoesNotCrash_VectorInput(
    tmt.TestSimulateDoesNotCrash_VectorInput, unittest.TestCase
):
    def setUp(self):
        super(TestiGIF_NPSimulateDoesNotCrash_VectorInput, self).setUp()
        self.mod = iGIF_NP(self.dt)


class TestiGIF_NPSimulateDoesNotCrash_ModelStimInput(
    tmt.TestSimulateDoesNotCrash_ModelStimInput, unittest.TestCase
):
    def setUp(self):
        super(TestiGIF_NPSimulateDoesNotCrash_ModelStimInput, self).setUp()
        self.mod = iGIF_NP(self.dt)


class TestiGIF_NPSimulate_StimulusResponse(
    tmt.TestSimulate_StimulusResponse, unittest.TestCase
):
    def setUp(self):
        super(TestiGIF_NPSimulate_StimulusResponse, self).setUp()
        self.mod = iGIF_NP(self.dt)
        self.mod.El = self.V0


# iGIF_Na tests.
class TestiGIF_NaSimulateDoesNotCrash_VectorInput(
    tmt.TestSimulateDoesNotCrash_VectorInput, unittest.TestCase
):
    def setUp(self):
        super(TestiGIF_NaSimulateDoesNotCrash_VectorInput, self).setUp()
        self.mod = iGIF_Na(self.dt)


class TestiGIF_NaSimulateDoesNotCrash_ModelStimInput(
    tmt.TestSimulateDoesNotCrash_ModelStimInput, unittest.TestCase
):
    def setUp(self):
        super(TestiGIF_NaSimulateDoesNotCrash_ModelStimInput, self).setUp()
        self.mod = iGIF_Na(self.dt)


class TestiGIF_NaSimulate_StimulusResponse(
    tmt.TestSimulate_StimulusResponse, unittest.TestCase
):
    def setUp(self):
        super(TestiGIF_NaSimulate_StimulusResponse, self).setUp()
        self.mod = iGIF_Na(self.dt)
        self.mod.El = self.V0


if __name__ == '__main__':
    unittest.main()
