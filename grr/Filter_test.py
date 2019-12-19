import unittest

import numpy as np
import numpy.testing as npt

from grr.Filter import constructMedianFilter
import grr.Filter_Rect as frect
from grr.Filter_Exps import Filter_Exps


class TestConstructMedianFilter(object):
    def setUp(self):
        self._setFilterType()
        self._instantiateRandomFilters()
        self._instantiateExpectedFilter()

    def _setFilterType(self):
        raise NotImplementedError('Must be implemented by subclasses.')
        # self.filterType = <someFilterType>

    def _instantiateRandomFilters(self):
        """Instantiate random set of filters.

        Can be overridden by subclasses.

        """
        noOfFilters = 10
        self.filters = [self.filterType() for i in range(10)]
        noOfCoefficients = self.filters[0].getNbOfBasisFunctions()
        self.randomCoefficients = np.random.normal(
            size=(noOfFilters, noOfCoefficients)
        )
        for i, filt in enumerate(self.filters):
            filt.setFilter_Coefficients(self.randomCoefficients[i, :])

    def _instantiateExpectedFilter(self):
        """Instantiate `expectedFilter` attribute.

        Can be overridden by subclasses.

        """
        self.expectedFilter = self.filterType()
        self.expectedFilter.setFilter_Coefficients(
            np.median(self.randomCoefficients, axis=0)
        )

    def testConstruct(self):
        medianFilter = constructMedianFilter(self.filterType, self.filters)
        npt.assert_array_almost_equal(
            medianFilter.getCoefficients(),
            self.expectedFilter.getCoefficients(),
            err_msg='Filter coefficients not as expected.',
        )


class TestContructMedianFilter_FilterRectLogSpaced(
    TestConstructMedianFilter, unittest.TestCase
):
    def _setFilterType(self):
        self.filterType = frect.Filter_Rect_LogSpaced


class TestContructMedianFilter_FilterRectLinSpaced(
    TestConstructMedianFilter, unittest.TestCase
):
    def _setFilterType(self):
        self.filterType = frect.Filter_Rect_LinSpaced


class TestContructMedianFilter_FilterRectArbitrarilySpaced(
    TestConstructMedianFilter, unittest.TestCase
):
    def _setFilterType(self):
        self.filterType = frect.Filter_Rect_ArbitrarilySpaced


class TestContructMedianFilter_FilterExps(
    TestConstructMedianFilter, unittest.TestCase
):
    def _setFilterType(self):
        self.filterType = Filter_Exps

    def _instantiateRandomFilters(self):
        """Instantiate random set of filters."""
        noOfFilters = 10
        self.filters = [self.filterType() for i in range(10)]
        noOfCoefficients = 5
        self.randomCoefficients = np.random.normal(
            size=(noOfFilters, noOfCoefficients)
        )
        for i, filt in enumerate(self.filters):
            filt.setFilter_Timescales(
                np.linspace(5.0, 100.0, noOfCoefficients)
            )
            filt.setFilter_Coefficients(self.randomCoefficients[i, :])

    def _instantiateExpectedFilter(self):
        """Instantiate `expectedFilter` attribute."""
        self.expectedFilter = self.filterType()
        self.expectedFilter.setFilter_Timescales(
            np.linspace(5.0, 100.0, self.randomCoefficients.shape[1])
        )
        self.expectedFilter.setFilter_Coefficients(
            np.median(self.randomCoefficients, axis=0)
        )


if __name__ == '__main__':
    unittest.main()
