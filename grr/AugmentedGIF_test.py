import unittest

import grr.ThresholdModel_testutil as tmt
from grr.AugmentedGIF import AugmentedGIF


class TestAugmentedGIFSimulateDoesNotCrash_VectorInput(
    tmt.TestSimulateDoesNotCrash_VectorInput, unittest.TestCase
):
    def setUp(self):
        super(TestAugmentedGIFSimulateDoesNotCrash_VectorInput, self).setUp()
        self.mod = AugmentedGIF(self.dt)


class TestAugmentedGIFSimulateDoesNotCrash_ModelStimInput(
    tmt.TestSimulateDoesNotCrash_ModelStimInput, unittest.TestCase
):
    def setUp(self):
        super(
            TestAugmentedGIFSimulateDoesNotCrash_ModelStimInput, self
        ).setUp()
        self.mod = AugmentedGIF(self.dt)


class TestAugmentedGIFSimulate_StimulusResponse(
    tmt.TestSimulate_StimulusResponse, unittest.TestCase
):
    def setUp(self):
        super(TestAugmentedGIFSimulate_StimulusResponse, self).setUp()
        self.mod = AugmentedGIF(self.dt)
        self.mod.El = self.V0
        self.mod.E_K = self.V0


class TestConstructMedianModel_AugmentedGIF(
    tmt.TestConstructMedianModel, unittest.TestCase
):
    def _setModelType(self):
        self.modelType = AugmentedGIF


if __name__ == '__main__':
    unittest.main()
