"""Input stimulus object for GIF-like models.

@author : Emerson Harkin

"""

import numpy as np


class ModelStimulus(object):
    """Container for input to GIF and related models."""

    def __init__(self, dt):
        self.dt = dt

        self._currentArray = InputArray([], dt)
        self._conductanceArray = InputArray([], dt)
        self._conductanceReversals = []

    @property
    def timesteps(self):
        """Duration of the stimulus in timesteps."""
        timesteps_ = [
            self._currentArray.timesteps,
            self._conductanceArray.timesteps
        ]
        # Check consistency.
        if min(timesteps_) > 0 and (max(timesteps_) != min(timesteps_)):
            raise RuntimeError(
                'Expected currents and conductances to have same number of '
                'timesteps; got {} and {} instead.'.format(
                    self._currentArray.timesteps,
                    self._conductanceArray.timesteps
                )
            )

        return max(timesteps_)

    @property
    def duration(self):
        """Duration of the stimulus in ms."""
        return self.timesteps * self.dt

    @property
    def numberOfCurrents(self):
        return self._currentArray.numberOfInputVectors

    @property
    def numberOfConductances(self):
        assert self._conductanceArray.numberOfInputVectors == len(self._conductanceReversals)
        return self._conductanceArray.numberOfInputVectors

    def getNetCurrentVector(self, dtype=np.float64):
        """Get vector of current to pass as model input.

        Friend function of GIF classes, not for public use.

        """
        return np.asarray(self._currentArray.sum()).astype(dtype)

    def getConductanceMatrix(self, dtype=np.float64):
        """Get conductance timeseries as a time by numberOfConductances matrix.

        Friend function of GIF classes, not for public use.

        """
        return np.asarray(self._conductanceArray.asTByNMatrix()).astype(dtype)

    def getConductanceReversals(self, dtype=np.float64):
        """Get conductance reversal potentials (mV).

        Friend function of GIF classes, not for public use.

        """
        return np.asarray(self._conductanceReversals).astype(dtype)

    def appendCurrents(self, currentArray):
        """Add an array of currents."""
        # Check that timestep is compatible.
        if hasattr(currentArray, 'dt') and not np.isclose(self.dt, currentArray.dt):
            raise ValueError(
                'currentArray.dt {} does not match ModelStimulus.dt {}'.format(
                    currentArray.dt, ModelStimulus.dt
                )
            )

        self._currentArray.append(currentArray)

    def appendConductances(self, conductanceArray, reversalPotentials):
        """Add an array of conductances.

        Arguments
        ---------
        conductanceArray : 1d or 2d array-like, or InputArray
        reversalPotentials : 1d array-like
            Reveral potential (mV) for each conductance.

        """
        if not issubclass(type(conductanceArray), InputArray):
            if hasattr(conductanceArray, 'dt'):
                conductanceArray = InputArray(conductanceArray, conductanceArray.dt)
            else:
                conductanceArray = InputArray(conductanceArray, self.dt)
        if len(np.atleast_1d(reversalPotentials)) != conductanceArray.numberOfInputVectors:
            raise ValueError(
                'Expected same number of reversalPotentials and '
                'conductanceVectors; got {} and {}'.format(
                    len(np.atleast_1d(reversalPotentials)),
                    conductanceArray.numberOfInputVectors
                )
            )

        self._conductanceArray.append(conductanceArray)
        self._conductanceReversals.append(np.atleast_1d(reversalPotentials))


class InputArray(object):
    """An array to be provided as input to a model."""

    def __init__(self, array, dt):
        """Initialize InputArray.

        Arguments
        ---------
        array : 1D or 2D array
        dt : float
            Timestep (ms).

        """
        arr_to_store = np.atleast_2d(array)
        if arr_to_store.ndim != 2:
            raise ValueError(
                'Input array should be a vector or matrix, '
                'not {}d array.'.format(arr_to_store.ndim)
            )

        self._array = arr_to_store
        self.dt = dt

    @property
    def timesteps(self):
        return self._array.shape[1]

    @property
    def duration(self):
        """Duration of input vectors in ms."""
        return self.timesteps * self.dt

    @property
    def numberOfInputVectors(self):
        if self._array.shape[0] == 0:
            numVecs = 0
        elif self._array.shape[0] == 1 and self.timesteps == 0:
            numVecs = 0
        else:
            numVecs = self._array.shape[0]
        return numVecs

    def sum(self):
        """Sum over InputVectors."""
        return self._array.sum(axis=0)

    def getInputVectors(self):
        """Return InputArray as a list of 1D InputArrays."""
        return [self.getInputVector(i) for i in range(self.numberOfInputVectors)]

    def getInputVector(self, idx):
        return InputArray(self._array[idx, :], self.dt)

    def asTByNMatrix(self):
        """Get input vectors as a matrix with time along rows."""
        return self._array.T

    def append(self, array):
        # Coerce to InputArray type.
        if not issubclass(type(array), InputArray):
            if hasattr(array, 'dt'):
                arr_to_append = InputArray(array, array.dt)
            else:
                arr_to_append = InputArray(array, self.dt)
        else:
            arr_to_append = array

        if self.timesteps == 0 and self.numberOfInputVectors < 2:
            self._array = arr_to_append._array
        elif self.timesteps == arr_to_append.timesteps and np.isclose(self.dt, arr_to_append.dt):
            self._array = np.concatenate([self._array, arr_to_append._array], axis=0)
        elif self.timesteps != arr_to_append.timesteps:
            raise ValueError(
                'Array to cannot be appended with {} timesteps. '
                'Expected {} timesteps.'.format(
                    arr_to_append.timesteps, self.timesteps
                )
            )
        elif not np.isclose(self.dt, arr_to_append.dt):
            raise ValueError(
                'Array dt {} not equal to instance dt {}'.format(
                    arr_to_append.dt, self.dt
                )
            )
        else:
            raise RuntimeError('Cannot get here.')
