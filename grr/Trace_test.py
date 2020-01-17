import unittest

import numpy as np
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


class TestGetRisingEdges(unittest.TestCase):
    def testGetsOnlyRising(self):
        x = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        expected = [1, 3]
        actual = Trace.getRisingEdges(x, 0.5, 0)
        npt.assert_array_equal(expected, actual)

    def testDebounceInclusiveEdge(self):
        x = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        expected = [3]
        actual = Trace.getRisingEdges(x, 0.5, 2)
        npt.assert_array_equal(expected, actual)

    def testDebounce(self):
        x = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        expected = [3]
        actual = Trace.getRisingEdges(x, 0.5, 3)
        npt.assert_array_equal(expected, actual)


class TestDetectSpikes(unittest.TestCase):
    def testRowsAsTimeAxis(self):
        x = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]).T
        expected = [[0.3, 0.5], [0.3, 0.6]]
        actual = Trace.detectSpikes(x, 0.5, 0., 0, 0.1)
        npt.assert_array_almost_equal(expected, actual)

    def testColumnsAsTimeAxis(self):
        x = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
        expected = [[0.3, 0.5], [0.3, 0.6]]
        actual = Trace.detectSpikes(x, 0.5, 0., 1, 0.1)
        npt.assert_array_almost_equal(expected, actual)

    def testFlattenArrayBeforeDetection(self):
        x = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
        expected = [0.3, 0.5, 1.1, 1.4]
        actual = Trace.detectSpikes(x, 0.5, 0., -1, 0.1)
        npt.assert_array_almost_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()

