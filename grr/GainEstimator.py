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
        gain : float 1D array
        intercept : float 1D array
        gainUncertainty, interceptUncertainty : float 1D array
            Standard deviation of estimated gain and intercept over time.

        """
        self.measuredResponse = measuredResponse
        self.inputAmplitudes = inputAmplitudes
        self._baselineIntervalTime = baselineInterval
        self._stimulusIntervalTime = stimulusInterval

        coeffs, uncertainties = self._getLinearFitCoeffsAndUncertainties()

        self.gain = coeffs[0, :]
        self.intercept = coeffs[1, :]
        self.gainUncertainty = uncertainties[0, :]
        self.interceptUncertainty = uncertainties[1, :]

    def _getLinearFitCoeffsAndUncertainties(self):
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
        coeffsOverTime = np.empty((no_timesteps, 2), order='C')
        coeffUncertaintiesOverTime = np.empty_like(coeffsOverTime)
        for t in range(no_timesteps):
            coeffs = np.polyfit(x, y[..., t].T, deg=1)
            coeffsOverTime[t, :] = coeffs.mean(axis=1)
            coeffUncertaintiesOverTime[t, :] = coeffs.std(axis=1)

        # Return transpose of coeffsOverTime, coeffUncer... so that time is
        # on last axis.
        return coeffsOverTime.T, coeffUncertaintiesOverTime.T

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
