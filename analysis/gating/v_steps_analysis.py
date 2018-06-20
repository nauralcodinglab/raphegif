#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

import sys
sys.path.append('./analysis/gating/')

from cell_class import Cell, Recording


#%% HANDY FUNCTIONS

max_normalize = lambda cell_channel: cell_channel / cell_channel.max(axis = 0)

def subtract_baseline(cell, baseline_range, channel):

    """
    Subtracts baseline from the selected channel of a Cell-like np.ndarray.

    Inputs:

    cell: Cell-like
    --  [c, t, s] array where c is channels, t is time (in timesteps), and s is sweeps

    baseline_range: slice
    --  Baseline time slice in timesteps

    channel: int
    --  Index of channel from which to subtract baseline. Not guaranteed to work with multiple channels.

    Returns:

        Copy of cell with baseline subtracted from the relevant channel.
    """

    cell = cell.copy()

    cell[channel, :, :] -= cell[channel, baseline_range, :].mean(axis = 0)

    return cell


def subtract_leak(cell, baseline_range, test_range, V_channel = 1, I_channel = 0):

    """
    Subtracts leak conductance from the I channel of a Cell-like np.ndarray.

    Calculates leak conductance based on Rm, which is extracted from test pulse.
    Assumes test pulse is the same in each sweep.

    Inputs:

        cell: Cell-like
        --  [c, t, s] array where c is channels, t is time (in timesteps), and s is sweeps

        baseline_range: slice
        --  baseline time slice in timesteps

        test_range: slice
        --  test pulse time slice in timesteps

        V_channel: int

        I_channel: int

    Returns:

        Leak-subtracted array.
    """

    Vtest_step = (
    cell[V_channel, baseline_range, :].mean(axis = 0)
    - cell[V_channel, test_range, :].mean(axis = 0)
    ).mean()
    Itest_step = (
    cell[I_channel, baseline_range, :].mean(axis = 0)
    - cell[I_channel, test_range, :].mean(axis = 0)
    ).mean()

    Rm = Vtest_step/Itest_step

    I_leak = (cell[V_channel, :, :] - cell[V_channel, baseline_range, :].mean()) / Rm

    leak_subtracted = cell.copy()
    leak_subtracted[I_channel, :, :] -= I_leak

    return leak_subtracted


#%% LOAD DATA

FIGDATA_PATH = './figs/figdata/'
GATING_PATH = './data/gating/'

# Load V-steps files for sample pharma traces.
baseline = Cell().read_ABF('./figs/figdata/18411010.abf')[0]
TEA = Cell().read_ABF('./figs/figdata/18411013.abf')[0]
TEA_4AP = Cell().read_ABF('./figs/figdata/18411015.abf')[0]

# Load drug washin files
TEA_washin = Cell().read_ABF([FIGDATA_PATH + '18411020.abf',
                              FIGDATA_PATH + '18411012.abf',
                              FIGDATA_PATH + '18412002.abf'])
TEA_4AP_washin = Cell().read_ABF([FIGDATA_PATH + '18411022.abf',
                                  FIGDATA_PATH + '18411014.abf'])
XE_washin = Cell().read_ABF([GATING_PATH + '18619021.abf',
                             GATING_PATH + '18614035.abf'])

# Load gating data
gating = Cell().read_ABF([GATING_PATH + '18411002.abf',
                          GATING_PATH + '18411010.abf',
                          GATING_PATH + '18411017.abf',
                          GATING_PATH + '18411019.abf',
                          GATING_PATH + 'c0_inact_18201021.abf',
                          GATING_PATH + 'c1_inact_18201029.abf',
                          GATING_PATH + 'c2_inact_18201034.abf',
                          GATING_PATH + 'c3_inact_18201039.abf',
                          GATING_PATH + 'c4_inact_18213011.abf',
                          GATING_PATH + 'c5_inact_18213017.abf',
                          GATING_PATH + 'c6_inact_18213020.abf',
                          GATING_PATH + '18619018.abf',
                          GATING_PATH + '18614032.abf'])

beautiful_gating_1 = Cell().read_ABF([GATING_PATH + '18619018.abf',
                                      GATING_PATH + '18619019.abf',
                                      GATING_PATH + '18619020.abf'])
beautiful_gating_1 = Recording(np.array(beautiful_gating_1).mean(axis = 0))

beautiful_gating_2 = Cell().read_ABF([GATING_PATH + '18614032.abf',
                                      GATING_PATH + '18614033.abf',
                                      GATING_PATH + '18614034.abf'])
beautiful_gating_2 = Recording(np.array(beautiful_gating_2).mean(axis = 0))


t_gating_bl = Cell().read_ABF([GATING_PATH + '18619038.abf',
                               GATING_PATH + '18619039.abf'])
t_gating_bl = Recording(np.array(t_gating_bl).mean(axis = 0))

#t_gating_washon = Cell().read_ABF([GATING_PATH + '18619049.abf'])[0] # High and changing Ra

#%% INSPECT RECORDINGS

%matplotlib qt5
beautiful_gating_1.plot(downsample = 1)

beautiful_gating_2.plot(downsample = 1)

t_gating_washon.plot(downsample = 1)

t_gating_washon.fit_test_pulse((2240, 2248), (3250, 3300))

# Look at XE991 (M-current blocker) washin
XE_washin_tmp = XE_washin[1]
XE_washin_tmp = subtract_baseline(XE_washin_tmp, slice(1000, 2000), 0)
XE_washin_tmp = subtract_leak(XE_washin_tmp, slice(1000, 2000), slice(3000, 3400))

plt.figure()
plt.plot(XE_washin_tmp[0, 60000:61000, :].mean(axis = 0))
plt.show()

"""
M-current blocker doesn't seem to affect leak- and baseline-subtracted current at -30mV.
"""

#%% PROCESS PHARMACOLOGY DATA

# Example traces.
sweep_to_use        = 10
xrange              = slice(25000, 45000)
xrange_baseline     = slice(24500, 25200)
baseline_sweep      = baseline[0, xrange, sweep_to_use] - baseline[0, xrange_baseline, sweep_to_use].mean()
TEA_sweep           = TEA[0, xrange, sweep_to_use] - TEA[0, xrange_baseline, sweep_to_use].mean()
TEA_4AP_sweep       = TEA_4AP[0, xrange, sweep_to_use] - TEA_4AP[0, xrange_baseline, sweep_to_use].mean()
cmd_sweep           = baseline[1, xrange, sweep_to_use] + TEA[1, xrange, sweep_to_use] / 2.

# TEA washin.
xrange_baseline     = slice(1000, 2000)
xrange_testpulse    = slice(3000, 3500)
xrange_ss           = slice(50000, 51000)
TEA_washin_pdata    = np.empty((len(TEA_washin), TEA_washin[0].shape[1], 44))
TEA_washin_pdata[:, :] = np.NAN

for i, cell in enumerate(TEA_washin):

    # Estimate Rm from test pulse and use to compute leak current.

    TEA_washin_pdata[i, :, :cell.shape[2]] = subtract_leak(cell, xrange_baseline, xrange_testpulse)[0, :, :]
    TEA_washin_pdata[i, :, :] -= np.nanmean(TEA_washin_pdata[i, xrange_baseline, :], axis = 0)
    #TEA_washin_pdata[i, :, :] /= np.nanmean(TEA_washin_pdata[i, xrange_ss, :6])

TEA_washin_pdata = np.nanmean(TEA_washin_pdata[:, xrange_ss, :], axis = 1).T

# Remove data where there is no conductance left after subtracting leak.
# (TEA can't have an effect if there's nothing to block.)
TEA_washin_pdata = np.delete(
TEA_washin_pdata,
np.where(TEA_washin_pdata[:7, :].mean(axis = 0) <= 0)[0],
axis = 1
)

# 4AP washin
# Probably won't use...
xrange_baseline         = slice(1000, 2000)
xrange_testpulse        = slice(3000, 3500)
xrange_ss               = slice(21770, 21800)
TEA_4AP_washin_pdata    = np.empty((len(TEA_4AP_washin), TEA_4AP_washin[0].shape[1], 44))
TEA_4AP_washin_pdata[:, :] = np.NAN

for i, cell in enumerate(TEA_4AP_washin):
    TEA_4AP_washin_pdata[i, :, :cell.shape[2]] = subtract_leak(cell, xrange_baseline, xrange_testpulse)[0, :, :]
    TEA_4AP_washin_pdata[i, :, :] -= np.nanmean(TEA_4AP_washin_pdata[i, xrange_baseline, :], axis = 0)
    #TEA_4AP_washin_pdata[i, :, :] /= np.nanmean(TEA_4AP_washin_pdata[i, xrange_ss, :6])

TEA_4AP_washin_pdata = np.nanmean(TEA_4AP_washin_pdata[:, xrange_ss, :], axis = 1).T

#%% PROCESS RAW GATING DATA

# Define time intervals from which to grab data.
xrange_baseline     = slice(0, 2000)
xrange_test         = slice(3500, 4000)
xrange_peakact      = slice(26140, 26160)
xrange_ss           = slice(55000, 56000)
xrange_peakinact    = slice(56130, 56160)

# Format will be [channel, sweep, cell]
# Such that we can use plt.plot(pdata[0, :, :], pdata[1, :, :], '-') to plot I over V by cell.

shape_pdata = (2, gating[0].shape[2], len(gating))

peakact_pdata       = np.empty(shape_pdata)
ss_pdata            = np.empty(shape_pdata)
peakinact_pdata     = np.empty(shape_pdata)

for i, cell in enumerate(gating):

    cell = subtract_baseline(cell, xrange_baseline, 0)
    cell = subtract_leak(cell, xrange_baseline, xrange_test)

    # Average time windows to get leak-subtracted IA and KSlow currents
    peakact_pdata[:, :, i]      = cell[:, xrange_peakact, :].mean(axis = 1)
    ss_pdata[:, :, i]           = cell[:, xrange_ss, :].mean(axis = 1)

    # Get prepulse voltage for peakinact
    peakinact_pdata[0, :, i]    = cell[0, xrange_peakinact, :].mean(axis = 0)
    peakinact_pdata[1, :, i]    = cell[1, xrange_peakact, :].mean(axis = 0)

peakact_pdata[0, :, :]      /= peakact_pdata[1, :, :] - -101
ss_pdata[0, :, :]           /= ss_pdata[1, :, :] - -101
peakinact_pdata[0, :, :]    /= peakinact_pdata[1, :, :] - -101

# Average out small differences in cmd between cells due to Rs comp
peakact_pdata[1, :, :]      = peakact_pdata[1, :, :].mean(axis = 1, keepdims = True)
ss_pdata[1, :, :]           = ss_pdata[1, :, :].mean(axis = 1, keepdims = True)
peakinact_pdata[1, :, :]    = peakinact_pdata[1, :, :].mean(axis = 1, keepdims = True)

# Remove contribution of KSlow to apparent inactivation peak.
peakinact_pdata[0, :, :] -= ss_pdata[0, :, :]

# Pickle in case needed.

with open(FIGDATA_PATH + 'peakact_pdata.pyc', 'wb') as f:
    pickle.dump(peakact_pdata, f)

with open(FIGDATA_PATH + 'ss_pdata.pyc', 'wb') as f:
    pickle.dump(ss_pdata, f)

with open(FIGDATA_PATH + 'peakinact_pdata.pyc', 'wb') as f:
    pickle.dump(peakinact_pdata, f)


#%% FIG SIGMOID CURVES TO GATING DATA

def sigmoid_curve(p, V):

    """Three parameter logit.

    p = [A, k, V0]

    y = A / ( 1 + exp(-k * (V - V0)) )
    """

    if len(p) != 3:
        raise ValueError('p must be vector-like with len 3.')

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


def optimizer_wrapper(pdata, p0, max_norm = True):

    """
    Least-squares optimizer

    Uses `compute_residuals` as the loss function to optimize `sigmoid_curve`

    Returns:

    Tupple of parameters and corresponding curve.
    Curve is stored as a [channel, sweep] np.ndarray; channels 0 and 1 should correspond to I and V, respectively.
    Curve spans domain of data used for fitting.
    """

    X = pdata[1, :, :].flatten()

    if max_norm:
        y = max_normalize(pdata[0, :, :]).flatten()
    else:
        y = pdata[0, :, :].flatten()

    p = optimize.least_squares(compute_residuals, p0, kwargs = {
    'func': sigmoid_curve,
    'X': X,
    'Y': y
    })['x']

    no_pts = 500

    fitted_points = np.empty((2, no_pts))
    x_min = pdata[1, :, :].mean(axis = 1).min()
    x_max = pdata[1, :, :].mean(axis = 1).max()
    fitted_points[1, :] = np.linspace(x_min, x_max, no_pts)
    fitted_points[0, :] = sigmoid_curve(p, fitted_points[1, :])

    return p, fitted_points


peakact_params, peakact_fittedpts       = optimizer_wrapper(peakact_pdata, [12, 1, -30])
peakinact_params, peakinact_fittedpts   = optimizer_wrapper(peakinact_pdata, [12, -1, -60])
ss_params, ss_fittedpts                 = optimizer_wrapper(ss_pdata, [12, 1, -25])

param_pickle_df = pd.DataFrame(
{
'm': peakact_params,
'h': peakinact_params,
'n': ss_params
},
index = ('A', 'k', 'V_half')
)

with open(FIGDATA_PATH + 'gating_params.pyc', 'wb') as f:
    pickle.dump(param_pickle_df, f)
