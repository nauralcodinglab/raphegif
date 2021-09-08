from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from .Tools import raiseExpectedGot


class FrequencyInputCurve(object):
    """A frequency vs. input curve fitted to data."""

    summary_metrics = [
        'f',
        'I',
        'CV',
        'rheobase',
        'freq_at_rheobase',
        'freq_at_50',
        'gain',
        'is_monotonic',
        'spearman_rho',
        'spearman_p',
    ]

    def __init__(self, is_monotonic_function):
        """Initialize FrequencyInputCurve.

        Arguments
        ---------
        is_monotonic_function: callable
            Function for classifying f/I curves as monotonically
            increasing or not. Must accept a list of spike frequencies
            and return a boolean value.

        """
        self.is_monotonic_function = is_monotonic_function

        for attr in self.summary_metrics:
            setattr(self, attr, None)

        self._rheobase_ind = None
        self._linear_fit_coeffs = None

    def get_metric(self, metric_name):
        """Get summary metric by name.

        Arguments
        ---------
        metric_name: str
            Must be one of the metrics listed in `summary_metrics` class
            attribute.

        Returns
        -------
        Value of metric or `None`.

        """
        return getattr(self, metric_name, None)

    def fit(self, spike_times, input_current, window, dt):
        """Fit an f/I curve to data.

        Arguments
        ---------
        spike_times: list of lists
            Spike times for each sweep in ms.
        input_current: 2D array
            Stimulus current array with dimensionality [time, sweep].
        window: pair of ints
            Time window to get spike frequency **in timesteps**.
        dt: float
            Time step length in ms.

        Effects
        -------
        Sets instance attributes listed in `summary_metrics` class attribute.

        """
        spks_in_window = self._get_spikes_in_window(spike_times, window, dt)

        # Compute and store summary metrics.
        ISIs = [np.diff(x) for x in spks_in_window]
        self.CV = [x.std() / x.mean() if len(x) > 0 else 0 for x in ISIs]

        self.f = np.array([len(x) for x in spks_in_window]) / (
            1e-3 * dt * (window[1] - window[0])
        )

        self.I = input_current[
            max(window[1] - 1000, window[0]) : (
                window[1] - 10
            )  # Offsets are to avoid boundary effects from stimulus edges.
        ].mean(axis=0)

        self._rheobase_ind = np.where(self.f > 1e-8)[0][0]
        self.rheobase = self.I[self._rheobase_ind]
        self.freq_at_rheobase = self.f[self._rheobase_ind]

        self.freq_at_50 = self._get_interpolated_f_at_I(50.0)

        self.is_monotonic = self.is_monotonic_function(self.f)
        self._linear_fit_coeffs = np.polyfit(
            self.I[self._rheobase_ind :], self.f[self._rheobase_ind :], 1
        )
        self.gain = self._linear_fit_coeffs[0]

        self.spearman_rho, self.spearman_p = spearmanr(
            self.I[self._rheobase_ind :], self.f[self._rheobase_ind :]
        )

    @staticmethod
    def _get_spikes_in_window(spike_times, window, dt):
        # Argument checks.
        if len(window) != 2:
            raiseExpectedGot('pair of values', 'argument `window`', window)
        elif int(window[0]) != window[0] or int(window[1]) != window[1]:
            raiseExpectedGot('integers', 'values of argument `window`', window)

        # Get spikes in window.
        spks_in_window = [
            np.asarray(x)[(x >= window[0] * dt) & (x < window[1] * dt)] * dt
            for x in spike_times
        ]

        return spks_in_window

    def _get_interpolated_f_at_I(self, input_):
        return np.interp(input_, self.I, self.f)

    def plot(self, fitted=True, fitted_pltargs={}, ax=None, **pltargs):
        """Make a plot of the frequency vs. input curve and linear fit.

        Arguments
        ---------
        fitted: boolean, default True
            Whether to plot the linear fit to the f/I curve.
        fitted_pltargs: dict
            Formatting parameters for linear fit to f/I curve. Overrides other
            pltargs if any are given. Only used if `fitted` is `True`.
        ax: matplotlib.Axes
            Axes on which to make the plot. Defaults to current axes if `None`.
        pltargs
            Additional parameters passed to `ax.plot()` as keyword arguments.

        """
        if ax is None:
            ax = plt.gca()

        ax.plot(self.I, self.f, **pltargs)
        if fitted:
            if 'label' in pltargs.keys():
                # `label` doesn't apply to fitted line, even if it isn't
                # overridden by `fitted_pltargs`, so we must remove it.
                pltargs.pop('label')
            pltargs.update(fitted_pltargs)  # Inherit/override `pltargs`.

            ax.plot(*self._get_fitted_curve(), **pltargs)

    def _get_fitted_curve(self):
        x_fitted = self.I[self._rheobase_ind :]
        y_fitted = np.polyval(self._linear_fit_coeffs, x_fitted)
        return (x_fitted, y_fitted)

    def copy(self):
        """Return deep copy of self."""
        return deepcopy(self)
