import numpy as np

from .Tools import timeToIndex


class GainEstimator(object):
    """Estimate the gain of a process from measured input and output."""

    def __init__(self, dt=0.1):
        """Initialize GainEstimator.

        Arguments
        ---------
        dt : float, default 0.1
            Time step of measuredResponse signal (ms).

        """
        self.dt = dt

    def fit(
        self,
        measuredResponse,
        inputAmplitudes,
        baselineInterval,
        stimulusInterval,
    ):
        """Fit linear model of measuredResponse.

        Arguments
        ---------
        measuredResponse : 3D float array-like
        inputAmplitudes : 1D float array-like
        baselineInterval, stimulusInterval : pair of floats
            Time intervals in ms.

        Initializes the following attributes
        ------------------------------------
        gainByRep, interceptByRep: float 2D array
            Gain and intercept over time for each rep. Dimensionality is
            [rep, time].
        gain, intercept: float 1D array
            Mean gain and intercept over time.
        gainUncertainty, interceptUncertainty : float 1D array
            Standard deviation of estimated gain and intercept over time.

        """
        self.measuredResponse = measuredResponse
        self.inputAmplitudes = inputAmplitudes
        self._baselineIntervalTime = baselineInterval
        self._stimulusIntervalTime = stimulusInterval

        coeffs = self._getLinearFitCoeffs()

        # 2D arrays [cells, time]
        self.gainByRep = coeffs[:, 0, :]
        self.interceptByRep = coeffs[:, 0, :]

        repAxis = 0
        self.gain = coeffs[:, 0, :].mean(axis=repAxis)
        self.intercept = coeffs[:, 1, :].mean(axis=repAxis)
        self.gainUncertainty = coeffs[:, 0, :].std(axis=repAxis)
        self.interceptUncertainty = coeffs[:, 1, :].std(axis=repAxis)

    def _getLinearFitCoeffs(self):
        """Linear fit of measured response."""
        x = self.inputAmplitudes
        y = (
            self.measuredResponse[..., self.stimulusSlice]
            - self._getMeanMeasuredResponseBaselineBySweep()[:, np.newaxis]
        )
        no_timesteps = y.shape[2]

        # Pre-allocate arrays to store coefficients.
        # Implementation note: time is usually over the last axis in this
        # class. Here, time is over the first axis so that coeffs (second axis)
        # from a given fit are adjacent in memory.
        coeffsOverTime = np.empty((no_timesteps, 2, self.no_reps), order='C')
        for t in range(no_timesteps):
            coeffsOverTime[t, :, :] = np.polyfit(x, y[..., t].T, deg=1)

        # Return transpose of coeffsOverTime so that time is on last axis.
        return coeffsOverTime.T

    def _getMeanMeasuredResponseBaselineBySweep(self):
        averageBaselineBySweep = (
            self.measuredResponse[..., self.baselineSlice]
            .mean(axis=0)
            .mean(axis=1)
        )
        assert len(averageBaselineBySweep) == self.no_sweeps
        return averageBaselineBySweep

    @property
    def no_sweeps(self):
        """Number of sweeps at different input amplitudes."""
        assert self.measuredResponse.shape[1] == len(self.inputAmplitudes)
        return self.measuredResponse.shape[1]

    @property
    def no_reps(self):
        """Number of repetitions."""
        return self.measuredResponse.shape[0]

    @property
    def baselineInterval(self):
        return self._baselineIntervalTime

    @property
    def baselineSlice(self):
        """Baseline time interval as a slice object."""
        return slice(*timeToIndex(self.baselineInterval, self.dt))

    @property
    def stimulusInterval(self):
        return self._stimulusIntervalTime

    @property
    def stimulusSlice(self):
        """Stimulus time interval as a slice object."""
        return slice(*timeToIndex(self.stimulusInterval, self.dt))
