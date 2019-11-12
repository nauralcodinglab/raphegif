"""Tests for input object to GIF-like classes.

@author : Emerson

"""

import unittest

import numpy as np

from grr import ModelStimulus


class TestModelStimulus_Append(unittest.TestCase):

    def setUp(self):
        self.dt = 0.5
        self.modStim = ModelStimulus.ModelStimulus(self.dt)

    def testAppendCurrents(self):
        try:
            self.modStim.appendCurrents(np.zeros((10, 50)))
        except Exception as e:
            self.fail('ModelStimulus.appendCurrents() raised {} unexpectedly: {}'.format(
                type(e), e.message
            ))

    def testAppendConductances(self):
        try:
            self.modStim.appendConductances(np.zeros((10, 50)), np.zeros(10))
        except Exception as e:
            self.fail('ModelStimulus.appendCurrents() raised {} unexpectedly: {}'.format(
                type(e), e.message
            ))

        with self.assertRaises(ValueError):
            self.modStim.appendConductances(np.zeros((10, 50)), np.zeros(9))


class TestModelStimulus_Currents(unittest.TestCase):

    def setUp(self):
        self.dt = 0.5
        self.modStim = ModelStimulus.ModelStimulus(self.dt)

    def test_netCurrent_emptyCurrent(self):
        """Test behaviour of getNetCurrentVector when there are no currents."""
        currentVec = self.modStim.getNetCurrentVector()
        self.assertEqual(len(currentVec), 0, 'Length of empty current vector is not zero.')
        self.assertEqual(currentVec.ndim, 1, 'Length of empty current vector is not one.')

        # Try getting currentVector with various data types.
        def checkDtype(dtype, expectedDtype):
            netCurrent = self.modStim.getNetCurrentVector(dtype)
            self.assertEqual(
                netCurrent.dtype, expectedDtype,
                'Got netCurrentVector of dtype {} with {} type specifier, '
                'expected {}'.format(netCurrent.dtype, dtype, expectedDtype)
            )
        for dtype, expectedDtype in zip(
            [np.float64, np.float32, 'double', 'float', 'single'],
            [np.float64, np.float32, np.float64, np.float64, np.float32]
        ):
            checkDtype(dtype, expectedDtype)


class TestInputArray_Constructor(unittest.TestCase):
    """Ensure that constructor raises errors for invalid input."""

    def test_initWithMatrix(self):
        numberOfInputVectors = 2
        dt = 0.7
        try:
            inputArr = ModelStimulus.InputArray(np.zeros((numberOfInputVectors, 10)), dt)
        except Exception as e:
            self.fail('Initializing InputArray with matrix raised {} unexpectedly: {}'.format(
            type(e), e.message
        ))
        self.assertEqual(
            inputArr.numberOfInputVectors, numberOfInputVectors,
            'Expected {} for InputArray.numberOfInputVectors, got {} instead.'.format(
                numberOfInputVectors, inputArr.numberOfInputVectors
            )
        )

    def test_initWithVector(self):
        dt = 0.7
        try:
            inputArr = ModelStimulus.InputArray(np.zeros(10), dt)
        except Exception as e:
            self.fail('Initializing InputArray with vector raised {} unexpectedly: {}'.format(
            type(e), e.message
        ))
        self.assertEqual(
            inputArr.numberOfInputVectors, 1,
            'Expected {} for InputArray.numberOfInputVectors, got {} instead.'.format(
                1, inputArr.numberOfInputVectors
            )
        )

    def test_initWithTooManyDimsRaisesError(self):
        dt = 0.7
        with self.assertRaises(ValueError):
            inputArr = ModelStimulus.InputArray(np.zeros((1, 2, 10)), dt)


class TestInputArray_Properties(unittest.TestCase):

    def test_timesteps(self):
        self.assertEqual(
            ModelStimulus.InputArray(np.zeros(20), 0.1).timesteps, 20,
            'InputArray timesteps incorrect for vector input.'
        )
        self.assertEqual(
            ModelStimulus.InputArray(np.zeros((2, 20)), 0.1).timesteps, 20,
            'InputArray timesteps incorrect for matrix input.'
        )

    def test_numberOfInputVectors(self):
        numInputVecs = ModelStimulus.InputArray([], 0.1).numberOfInputVectors
        self.assertEqual(
            numInputVecs, 0,
            'Got {} for numberOfInputVectors with empty InputArray, '
            'expected {}'.format(numInputVecs, 0)
        )

        numInputVecs = ModelStimulus.InputArray([0], 0.1).numberOfInputVectors
        self.assertEqual(
            numInputVecs, 1,
            'Got {} for numberOfInputVectors with vector InputArray, '
            'expected {}'.format(numInputVecs, 1)
        )

        numInputVecs = ModelStimulus.InputArray([[0], [0]], 0.1).numberOfInputVectors
        self.assertEqual(
            numInputVecs, 2,
            'Got {} for numberOfInputVectors with column vector InputArray, '
            'expected {}'.format(numInputVecs, 2)
        )

class TestInputArray_Append(unittest.TestCase):
    """Test appending to an empty InputArray."""

    def setUp(self):
        """Create an InputArray object for testing."""
        self.timesteps = 10
        self.numberOfInputVectors = 0
        self.dt = 0.1
        self.inputArr = ModelStimulus.InputArray([], self.dt)

    def test_appendNumpyMatrix(self):
        numberOfInputVectorsToAppend = 2

        try:
            self.inputArr.append(
                np.zeros((numberOfInputVectorsToAppend, self.timesteps))
            )
        except Exception as e:
            self.fail('InputArray.append() raised {} unexpectedly: {}'.format(
                type(e), e.message
            ))
        self.assertEqual(
            self.inputArr.numberOfInputVectors,
            self.numberOfInputVectors + numberOfInputVectorsToAppend,
            'got {} from `numberofinputvectors` property; expected {}'.format(
                self.inputArr.numberOfInputVectors,
                self.numberOfInputVectors + numberOfInputVectorsToAppend
            )
        )

    def test_appendNumpyVector(self):
        try:
            self.inputArr.append(np.zeros(self.timesteps))
        except Exception as e:
            self.fail('InputArray.append() raised {} unexpectedly: {}'.format(
                type(e), e.message
            ))
        self.assertEqual(
            self.inputArr.numberOfInputVectors,
            self.numberOfInputVectors + 1,
            'Got {} from `numberOfInputVectors` property; expected {}'.format(
                self.inputArr.numberOfInputVectors,
                self.numberOfInputVectors + 1
            )
        )

    def test_appendInputArray(self):
        try:
            self.inputArr.append(
                ModelStimulus.InputArray(np.zeros(self.timesteps), self.dt)
            )
        except Exception as e:
            self.fail('InputArray.append() raised {} unexpectedly: {}'.format(
                type(e), e.message
            ))

    def test_appendInvalidNumpyRaisesError(self):
        self.inputArr.append(np.zeros(self.timesteps))
        with self.assertRaises(ValueError):
            self.inputArr.append(np.zeros(self.timesteps + 1))
        with self.assertRaises(ValueError):
            self.inputArr.append(np.zeros(self.timesteps - 1))

    def test_appendInvalidInputArrayRaisesError(self):
        self.inputArr.append(np.zeros(self.timesteps))
        with self.assertRaises(ValueError):
            self.inputArr.append(
                ModelStimulus.InputArray(np.zeros(self.timesteps + 1), self.dt)
            )
        with self.assertRaises(ValueError):
            self.inputArr.append(
                ModelStimulus.InputArray(np.zeros(self.timesteps - 1), self.dt)
            )
        with self.assertRaises(ValueError):
            self.inputArr.append(
                ModelStimulus.InputArray(np.zeros(self.timesteps), self.dt + 0.1)
            )
        with self.assertRaises(ValueError):
            self.inputArr.append(
                ModelStimulus.InputArray(np.zeros(self.timesteps), self.dt - 0.1)
            )


if __name__ == '__main__':
    unittest.main()
