import unittest

import numpy.testing as npt

from grr import Trace


class TestFilterTimesByROI(unittest.TestCase):
    def testIntegerTimes(self):
        unfilteredTimes = [1, 5, 7, 110, 111, 150]
        ROI = [5, 111]
        expectedFilteredTimes = [5, 7, 110]
        npt.assert_array_equal(
            expectedFilteredTimes, Trace.filterTimesByROI(unfilteredTimes, ROI)
        )

    def testFloatTimes(self):
        unfilteredTimes = [0.1, 0.5, 0.7, 1.1, 1.11, 1.5]
        ROI = [0.45, 1.109]
        expectedFilteredTimes = [0.5, 0.7, 1.1]
        npt.assert_array_equal(
            expectedFilteredTimes, Trace.filterTimesByROI(unfilteredTimes, ROI)
        )


if __name__ == '__main__':
    unittest.main()

