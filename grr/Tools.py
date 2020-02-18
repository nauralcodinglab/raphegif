import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import weave


###########################################################
# Check integrity of objects.
###########################################################

def check_dict_fields(x, template, raise_error=True):
    """Check that fields in a dict match a template.

    Arguments
    ---------
    x : dict
        Dict to check.
    template : dict
        Dict to check against. Each field should either contain a dict (with
        additional fields) or `None`.
    raise_error : bool (default True)
        Raise a `KeyError` if any fields are missing.

    Returns
    -------
    Fields of `template` missing from `x`.

    """
    missing_fields = []
    for key in template:
        if key not in x:
            # Missing field base case.
            missing_fields.append(key)
        elif template[key] is None:
            # Successful base case.
            pass
        elif isinstance(template[key], dict):
            # Recursive case.
            try:
                missing_subfields = check_dict_fields(
                    x[key], template[key], raise_error=False
                )
            except TypeError as err:
                # Handle case that x[key] is not subscriptable.
                if 'not subscriptable' in err:
                    missing_subfields = template[key].keys()
                else:
                    raise
            if len(missing_subfields) > 0:
                missing_fields.extend(
                    [key + '/' + missing for missing in missing_subfields]
                )
        else:
            # Exception if wrong type in template.
            raise TypeError(
                'Expected type `None` or `dict` for template[`{key}`], '
                'got type `{type_}` instead.'.format(
                    key=key, type_=type(template[key])
                )
            )

    if len(missing_fields) > 0 and raise_error:
        raise KeyError('Missing fields {}'.format(missing_fields))

    return missing_fields


def validate_array_ndim(label, arr, dimensions):
    """Raise an exception if array has the wrong number of dimensions."""
    if dimensions == 0:
        raise ValueError('Arrays cannot have 0 dimensions.')
    elif np.ndim(arr) != dimensions:
        raise ValueError(
            'Expected {} to have {} dimensions, got {} instead.'.format(
                label, dimensions, np.ndim(arr)
            )
        )
    else:
        pass


def validate_matching_axis_lengths(arrs, axes_):
    """Raise an exception if array lengths are not identical along specified axes.

    Arguments
    ---------
    arrs : list of array-like
    axes_ : list-like of ints

    """
    for axis_ in axes_:
        axis_lengths = []
        for arr in arrs:
            axis_lengths.append(np.shape(arr)[axis_])
        if not all(axis_lengths[0] == np.array(axis_lengths)):
            raise ValueError(
                'Expected all arrays to have matching lengths along axis {}, '
                'got lengths {} instead.'.format(axis_, axis_lengths)
            )
        else:
            pass

def raiseExpectedGot(expected, for_, got):
    """Raise an error for an unexpected value.

    Raise a ValueError with the following message:
    `Expected `expected` for `for_`, got `got` instead.`

    """
    raise ValueError(
        'Expected {} for {}, got {} instead.'.format(expected, for_, got)
    )

def assertAllAlmostSame(values, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Raise a ValueError if not all values are almost the same."""
    # Raise a more helpful error message is non-equality is due to NaN values.
    if not equal_nan:
        if any(np.isnan(values)) and not all(np.isnan(values)):
            raise ValueError(
                "Expected all values to be close, but {} out of {} are NaN.".format(
                    sum(np.isnan(values)), np.asarray(values).size
                )
            )

    # Check non-NaN values.
    if not all(
        np.isclose(
            np.asarray(values).flatten()[0],
            values,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    ):
        raise ValueError(
            "Expected all values to be close, got values ranging from {} to {} (mean {}).".format(
                np.nanmin(values), np.nanmax(values), np.nanmean(values)
            )
        )


def assertHasAttributes(obj, requiredAttributes):
    """Raise TypeError if obj does not have all requiredAttributes.

    Arguments
    ---------
    obj : object instance
        Object to check for required attributes.
    requiredAttributes : list-like of strings
        Attributes obj must have.

    """
    for requiredAttribute in requiredAttributes:
        if not hasattr(obj, requiredAttribute):
            raise TypeError(
                '{} object is missing required '
                'attribute {}'.format(str(type(obj)), requiredAttribute)
            )


###########################################################
# Tools for plotting.
###########################################################

def removeAxis(ax, which_ax=['top', 'right']):

    for loc, spine in ax.spines.iteritems():
        if loc in which_ax:
            spine.set_color('none')  # don't draw spine

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def dashedBorder(ax=None, lw=0.5, color='gray'):
    if ax is None:
        ax = plt.gca()

    for side in ['right', 'left', 'top', 'bottom']:
        ax.spines[side].set_linestyle('--')
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_edgecolor(color)


###########################################################
# Reprint
###########################################################
def reprint(str):
    sys.stdout.write('%s\r' % (str))
    sys.stdout.flush()


class gagProcess(object):
    """Class to forcibly gag verbose methods.

    Temporarily redirects stdout to block print commands.

    Usage:

    with gagProcess:
        print 'Things that will not be printed.'
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


###########################################################
# Generate Ornstein-Uhlenbeck process
###########################################################

def generateOUprocess(T=10000.0, tau=3.0, mu=0.0, sigma=1.0, dt=0.1, random_seed=42):
    """
    Generate an Ornstein-Uhlenbeck (stationnary) process with:
    - mean mu
    - standard deviation sigma
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    T_ind = int(T/dt)

    if random_seed is not None:
        np.random.seed(random_seed)
    white_noise = np.random.randn(T_ind)
    white_noise = white_noise.astype("double")

    OU_process = np.zeros(T_ind)
    OU_process[0] = mu
    OU_process = OU_process.astype("double")

    code = """

            #include <math.h>

            int cT_ind    = int(T_ind);
            float cdt     = float(dt);
            float ctau    = float(tau);
            float cmu     = float(mu);
            float csigma  = float(sigma);

            float OU_k1 = cdt / ctau ;
            float OU_k2 = sqrt(2.0*cdt/ctau) ;

            for (int t=0; t < cT_ind-1; t++) {
                OU_process[t+1] = OU_process[t] + (cmu - OU_process[t])*OU_k1 +  csigma*OU_k2*white_noise[t] ;
            }

            """

    vars = ['T_ind', 'dt', 'tau', 'sigma', 'mu', 'OU_process', 'white_noise']
    v = weave.inline(code, vars)

    return OU_process


def generateOUprocess_sinSigma(f=1.0, T=10000.0, tau=3.0, mu=0.0, sigma=1.0, delta_sigma=0.5, dt=0.1):
    """
    Generate an Ornstein-Uhlenbeck process with time dependent standard deviation:
    - mean mu
    - sigma(t) = sigma*(1+delta_sigma*sin(2pift)), f in Hz
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    OU_process = generateOUprocess(T=T, tau=tau, mu=0.0, sigma=1.0, dt=dt)
    t = np.arange(len(OU_process))*dt

    sin_sigma = sigma*(1+delta_sigma*np.sin(2*np.pi*f*t*10**-3))

    I = OU_process*sin_sigma + mu

    return I


def generateOUprocess_sinMean(f=1.0, T=10000.0, tau=3.0, mu=0.2, delta_mu=0.5, sigma=1.0, dt=0.1):
    """
    Generate an Ornstein-Uhlenbeck process with time dependent mean:
    - sigma
    - mu(t) = mu*(1+delta_mu*sin(2pift)), f in Hz
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    OU_process = generateOUprocess(T=T, tau=tau, mu=0.0, sigma=sigma, dt=dt)
    t = np.arange(len(OU_process))*dt

    sin_mu = mu*(1+delta_mu*np.sin(2*np.pi*f*t*10**-3))

    I = OU_process + sin_mu

    return I


###########################################################
# Functin to convert spike times in spike indices
###########################################################
def timeToIndex(time_, dt):

    time_ = np.atleast_1d(time_)
    x_i = np.array([int(np.round(s/dt)) for s in time_])
    x_i = x_i.astype('int')

    return x_i


def timeToIntVec(x_t, T, dt):
    """Convert vector of timestamps to a vector of zeros and ones.

    Inputs:
        x_t -- Vector of timestamps
        T   -- Total length of integer vector output (time units)
        dt  -- Timestep width
    """

    x_i = timeToIndex(x_t, dt)
    intvec = np.zeros(int(T / dt), dtype=np.int8)
    intvec[x_i] = 1

    return intvec


###########################################################
# Functions to perform exponential fit
###########################################################

def multiExpEval(x, bs, taus):

    result = np.zeros(len(x))
    L = len(bs)

    for i in range(L):
        result = result + bs[i] * np.exp(-x/taus[i])

    return result


def multiExpResiduals(p, x, y, d):
    bs = p[0:d]
    taus = p[d:2*d]

    return (y - multiExpEval(x, bs, taus))


def fitMultiExpResiduals(bs, taus, x, y):
    x = np.array(x)
    y = np.array(y)
    d = len(bs)
    p0 = np.concatenate((bs, taus))
    plsq = leastsq(multiExpResiduals, p0, args=(x, y, d), maxfev=100000, ftol=0.00000001)
    p_opt = plsq[0]
    bs_opt = p_opt[0:d]
    taus_opt = p_opt[d:2*d]

    fitted_data = multiExpEval(x, bs_opt, taus_opt)

    ind = np.argsort(taus_opt)

    taus_opt = taus_opt[ind]
    bs_opt = bs_opt[ind]

    return (bs_opt, taus_opt, fitted_data)


def getIndicesByPercentile(x, percentiles):
    """Get indices based on percentile of x.

    Main use case is to get eighty-twenty interval for fitting exponentials.

    Arguments
    ---------
    x : 1D array like
    percentiles : list of floats, range 0. - 1.
        Percentiles of x for which to return indices.

    Returns
    -------
    List of indices of x closest to percentiles, in order of appearance in
    percentiles.

    Example usage
    -------------
    > getIndicesByPercentile(np.arange(10, 0, -1), [0.80, 0.20])
    [8, 2]

    """
    # Input checks.
    if np.ndim(x) > 1:
        raiseExpectedGot(
            '1D array-like',
            'argument `x`',
            '{}D array-like'.format(np.ndim(x)),
        )
    if any(
        np.logical_or(
            np.asarray(percentiles) > 1.0, np.asarray(percentiles) < 0.0
        )
    ):
        raiseExpectedGot(
            'values between 0.0 and 1.0 ',
            'argument `percentiles`',
            'values ranging from {:.1f} to {:.1f}'.format(
                np.min(percentiles), np.max(percentiles)
            ),
        )

    # Convert x to percentiles.
    x = np.array(x, copy=True).astype(np.float64)
    data_percentiles = x - x.min()
    data_percentiles /= data_percentiles.max()

    # Find inds with closest match to `percentiles` argument.
    output = []
    for pctile in percentiles:
        output.append(np.argmin(np.abs(data_percentiles - pctile)))
    return output


###########################################################
# Misc utilities for handling arrays.
###########################################################

def stripNan(x):
    """Return copy of x with NaN values removed."""
    if np.ndim(x) > 1:
        raiseExpectedGot('vector-like', '`arr`', '{}d array'.format(np.ndim(x)))
    return np.array(x, copy=True).flatten()[~np.isnan(np.asarray(x).flatten())]

