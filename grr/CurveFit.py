from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from .cell_class import max_normalize_channel
from .Tools import raiseExpectedGot


def exponential_curve(p, t):
    """Three parameter exponential.

    I = (A + C) * exp (-t/tau) + C
    p = [A, C, tau]
    """
    A = p[0]
    C = p[1]
    tau = p[2]

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
    if len(p) != 3:
        raiseExpectedGot('3-tuple', 'argument `p`', p)
    A = p[0]
    k = p[1]
    V0 = p[2]

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
        y = max_normalize(pdata[0, :, :]).flatten()
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


def plot_linear_fit(x, y, ax=None, **pltargs):
    """Plot linear fit on a matplotlib axis."""
    if ax is None:
        ax = plt.gca()

    x_unique = np.unique(x)
    y_fitted = np.poly1d(np.polyfit(x, y, 1))(x_unique)

    ax.plot(x_unique, y_fitted, **pltargs)

