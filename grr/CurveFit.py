from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from .cell_class import subtract_baseline, subtract_leak
from .Tools import raiseExpectedGot, timeToIndex


def exponential_curve(p, t):
    """Three parameter exponential.

    I = (A + C) * exp (-t/tau) + C
    p = [A, C, tau]
    """
    A, C, tau = p
    return (A + C) * np.exp(-t / tau) + C


def fit_decay_curve(I, p0, dt=0.1):
    """Fit a monoexponential decay to a current `I`.

    Arguments
    ---------
    I : numeric 1D array
    p0 : 3 tuple
        Initial parameter guess: (peak, baseline_offset, time_constant).
    dt : float
        Time step.

    """
    t = np.arange(0, len(I) * dt, dt)[: len(I)]

    p = optimize.least_squares(
        compute_residuals,
        p0,
        kwargs={'func': exponential_curve, 'X': t, 'Y': I},
    )['x']

    no_pts = 500

    fitted_points = np.empty((2, no_pts))
    fitted_points[1, :] = np.linspace(t[0], t[-1], no_pts)
    fitted_points[0, :] = exponential_curve(p, fitted_points[1, :])

    return p, fitted_points


def sigmoid_curve(p, V):
    """Three parameter logit.

    p = [A, k, V0]
    y = A / ( 1 + exp(-k * (V - V0)) )
    """
    A, k, V0 = p
    return A / (1 + np.exp(-k * (V - V0)))


def compute_residuals(p, func, Y, X):
    """Compute residuals of a fitted curve.

    Inputs:
        p       -- vector of function parameters
        func    -- a callable function
        Y       -- real values
        X       -- vector of points on which to compute fitted values

    Returns:
        Array of residuals.
    """
    if len(Y) != len(X):
        raise ValueError('Y and X must be of the same length.')

    Y_hat = func(p, X)

    return Y - Y_hat


def fit_gating_curve(pdata, p0, max_norm=True):
    """Fit a sigmoid curve to `pdata`.

    Uses `compute_residuals` as the loss function to optimize `sigmoid_curve`

    Returns:

    - Tuple of parameters and corresponding curve.
    - Curve is stored as a [channel, sweep] np.ndarray; channels 0 and 1 should
      correspond to I and V, respectively.
    - Curve spans domain of data used for fitting.

    """
    X = pdata[1, :, :].flatten()

    if max_norm:
        y = _max_normalize(pdata[0, :, :]).flatten()
    else:
        y = pdata[0, :, :].flatten()

    p = optimize.least_squares(
        compute_residuals, p0, kwargs={'func': sigmoid_curve, 'X': X, 'Y': y}
    )['x']

    no_pts = 500

    fitted_points = np.empty((2, no_pts))
    x_min = pdata[1, :, :].mean(axis=1).min()
    x_max = pdata[1, :, :].mean(axis=1).max()
    fitted_points[1, :] = np.linspace(x_min, x_max, no_pts)
    fitted_points[0, :] = sigmoid_curve(p, fitted_points[1, :])

    return p, fitted_points


def _max_normalize(x, axis=0):
    return x / x.max(axis=axis)


def extract_gating_data(
    recordings,
    baseline_window,
    test_window,
    activation_peak_window,
    inactivation_peak_window,
    steady_state_window,
    window_unit='ms',
    current_channel=0,
    voltage_channel=1,
    potassium_equilibrium_potential=-101.0,
):
    """Process voltage step experiments for fitting gating curves.

    Intended to extract information about the activation and inactivation
    gates of I_A and the activation gate of a non-inactivating potassium
    current in 5-HT cells.

    Parameters
    ----------
    recordings
        List of `Recording`s of voltage step experiments.
    baseline_window: (float, float) or [(float, float)]
        Time window for extracting baseline. Voltage during this window should
        usually be -70mV.
    test_window: (float, float) or [(float, float)]
        Time window during test pulse to use for estimating membrane
        resistance.
    activation_peak_window: (float, float) or [(float, float)]
        Time window for estimating the voltage-dependence of I_A's activation
        gate based on peak current.
    inactivation_peak_window: (float, float) or [(float, float)]
        Time window for estimating the voltage-dependence of I_A's inactivation
        gate based on peak current.
    steady_state_window: (float, float) or [(float, float)]
        Time window for estimating the voltage-dependence of the activation
        gate of the non-inactivating potassium current found in 5-HT cells.
    window_unit: 'ms' or 'time_step'
        Whether `baseline_window`, `test_window`, etc are given in ms or time
        steps.
    current_channel, voltage_channel: int
        `Recording` channels corresponding to clamping current and command
        voltage, respectively.
    potassium_equilibrium_potential: float
        Reversal potential of potassium in gating experiments (mV).

    Returns
    -------
    dict

    """
    if not all([r.ndim == 3 for r in recordings]):
        raise ValueError('Expected all recordings to be 3D arrays.')
    if not all([r.shape[2] == recordings[0].shape[2] for r in recordings]):
        raise ValueError(
            'Expected all recordings to have the same number of sweeps.'
        )
    if not np.allclose([r.dt for r in recordings], recordings[0].dt):
        raise ValueError(
            'Expected all recordings to have the same sampling rate.'
        )
    num_sweeps = recordings[0].shape[2]
    num_recordings = len(recordings)
    dt = recordings[0].dt
    windows = {
        'baseline': _broadcast_time_window_to_slices(
            baseline_window, window_unit, dt, num_recordings
        ),
        'test': _broadcast_time_window_to_slices(
            test_window, window_unit, dt, num_recordings
        ),
        'activation_peak': _broadcast_time_window_to_slices(
            activation_peak_window, window_unit, dt, num_recordings
        ),
        'inactivation_peak': _broadcast_time_window_to_slices(
            inactivation_peak_window, window_unit, dt, num_recordings
        ),
        'steady_state': _broadcast_time_window_to_slices(
            steady_state_window, window_unit, dt, num_recordings
        ),
    }
    gating_data = {
        k: np.empty((2, num_sweeps, num_recordings))
        for k in ['activation_peak', 'inactivation_peak', 'steady_state']
    }

    for i, r_raw in enumerate(recordings):
        r = subtract_leak(
            subtract_baseline(r_raw, windows['baseline'][i], current_channel),
            windows['baseline'][i],
            windows['test'][i],
        )

        # Average current in each time window.
        for k in gating_data:
            gating_data[k][current_channel, :, i] = r[
                current_channel, windows[k][i], :
            ].mean(axis=0)

        # Average command voltage in appropriate time window.
        gating_data['activation_peak'][voltage_channel, :, i] = r[
            voltage_channel, windows['activation_peak'][i], :
        ].mean(axis=0)
        gating_data['steady_state'][voltage_channel, :, i] = r[
            voltage_channel, windows['steady_state'][i], :
        ].mean(axis=0)
        # Note: height of inactivation peak depends on voltage during
        # activation part of protocol.
        gating_data['inactivation_peak'][voltage_channel, :, i] = r[
            voltage_channel, windows['activation_peak'][i], :
        ].mean(axis=0)

    # Divide currents by driving force to convert to conductance.
    gating_data['activation_peak'][current_channel, :, :] /= (
        gating_data['activation_peak'][voltage_channel, :, :]
        - potassium_equilibrium_potential
    )
    gating_data['steady_state'][current_channel, :, :] /= (
        gating_data['steady_state'][voltage_channel, :, :]
        - potassium_equilibrium_potential
    )
    gating_data['inactivation_peak'][current_channel, :, :] /= (
        gating_data['inactivation_peak'][voltage_channel, -1, :]
        - potassium_equilibrium_potential
    )  # Since driving force is same for all sweeps.

    # Average out small differences in cmd between cells due to Rs comp
    for k in gating_data:
        gating_data[k][voltage_channel, ...] = gating_data[k][
            voltage_channel, ...
        ].mean(axis=1, keepdims=True)

    # Remove contribution of non-inactivating conductance to inactivation peak.
    gating_data['inactivation_peak'][current_channel, ...] -= gating_data[
        'steady_state'
    ][current_channel, ...]

    return gating_data


def _broadcast_time_window_to_slices(
    time_window, time_unit, dt, broadcast_length
):
    return [
        _time_window_to_slice(t, time_unit, dt)
        for t in _broadcast_tuple(time_window, broadcast_length)
    ]


def _time_window_to_slice(time_window, time_unit, dt):
    if time_unit == 'ms':
        result = slice(
            timeToIndex(time_window[0], dt), timeToIndex(time_window[1], dt)
        )
    elif time_unit in {'timestep', 'time_step'}:
        result = slice(time_window[0], time_window[1])
    else:
        raiseExpectedGot(
            "'ms' or 'time_step'", "argument 'time_unit'", time_unit
        )
    return result


def _broadcast_tuple(tuple_or_tuples, length):
    try:
        if not all(
            [len(t) == len(tuple_or_tuples[0]) for t in tuple_or_tuples]
        ):
            raise ValueError('Not all tuples have same length')
        if len(tuple_or_tuples) != length:
            raise ValueError(
                'Expected 1 or {} tuples, got {}'.format(
                    length, len(tuple_or_tuples)
                )
            )
        return tuple_or_tuples
    except TypeError:
        return [tuple_or_tuples for _ in range(length)]


def plot_linear_fit(x, y, ax=None, **pltargs):
    """Plot linear fit on a matplotlib axis."""
    if ax is None:
        ax = plt.gca()

    x_unique = np.unique(x)
    y_fitted = np.poly1d(np.polyfit(x, y, 1))(x_unique)

    ax.plot(x_unique, y_fitted, **pltargs)
