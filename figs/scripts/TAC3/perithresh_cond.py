#%% IMPORT MODULES

from __future__ import division

import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

import sys
sys.path.append('./analysis/gating/')
sys.path.append('./figs/scripts')

from cell_class import Cell, Recording
import pltools


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
baseline.set_dt(0.1)
TEA = Cell().read_ABF('./figs/figdata/18411013.abf')[0]
TEA.set_dt(0.1)
TEA_4AP = Cell().read_ABF('./figs/figdata/18411015.abf')[0]
TEA_4AP.set_dt(0.1)

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
beautiful_gating_1.set_dt(0.1)

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

baseline.plot(downsample = 1)
TEA.plot(downsample = 1)
TEA_4AP.plot(downsample = 1)

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

#%% LOAD GATING DATA

with open(FIGDATA_PATH + 'peakact_pdata.pyc', 'rb') as f:
    peakact_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'ss_pdata.pyc', 'rb') as f:
    ss_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'peakinact_pdata.pyc', 'rb') as f:
    peakinact_pdata = pickle.load(f)


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

ss_params

#%% MAKE FIGURE

plt.rc('text', usetex = True)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

IMG_PATH = './figs/ims/TAC3/'

spec = gs.GridSpec(2, 1, hspace = 0.45, top = 0.95, bottom = 0.08, right = 0.95)
spec_Kslow = gs.GridSpecFromSubplotSpec(2, 1, spec[0, :], hspace = 0.4)
spec_Kslow_traces = gs.GridSpecFromSubplotSpec(2, 2, spec_Kslow[0, :], height_ratios = [4, 1], wspace = 0.3)
spec_Kslow_gating = gs.GridSpecFromSubplotSpec(1, 2, spec_Kslow[1, :], width_ratios = [4, 1])

spec_IA = gs.GridSpecFromSubplotSpec(2, 1, spec[1, :], hspace = 0.4)
spec_IA_traces = gs.GridSpecFromSubplotSpec(2, 2, spec_IA[0, :], height_ratios = [4, 1], wspace = 0.3)
spec_IA_gating = gs.GridSpecFromSubplotSpec(1, 2, spec_IA[1, :], width_ratios = [4, 1])

plt.figure(figsize = (6, 6))

plt.subplot(spec_Kslow_traces[0, 0])
plt.title('\\textbf{{A1}} Non-inactivating K-current', loc = 'left')
Kslow_xlim = (2500, 5500)
plt.plot(
    beautiful_gating_1.t_mat[0, :, -3::-4],
    beautiful_gating_1[0, :, -3::-4], 'k-',
    linewidth = 0.5)
plt.xlim(Kslow_xlim)
plt.ylim(-100, 600)
pltools.add_scalebar(
    'ms', 'pA', x_on_left = False, anchor = (0.3, 0.4),
    x_label_space = 0.02, y_size = 250, bar_space = 0
)

plt.subplot(spec_Kslow_traces[1, 0])
plt.plot(
    beautiful_gating_1.t_mat[1, :, -3::-4],
    beautiful_gating_1[1, :, -3::-4],
    color = 'gray',
    linewidth = 0.5
)
plt.xlim(Kslow_xlim)
plt.annotate('-30mV', (5500, -28), ha = 'right', va = 'bottom')
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (-0.03, 0.2))


plt.subplot(spec_Kslow_traces[0, 1])
plt.title('\\textbf{{A2}} Block by TEA', loc = 'left')
plt.plot(
    baseline.t_mat[0, :, -3],
    baseline[0, :, -3],
    linewidth = 0.5, color = 'k',
    label = 'TTX'
)
plt.plot(
    TEA.t_mat[0, :, -3],
    TEA[0, :, -3],
    linewidth = 0.5, color = (0.9, 0.2, 0.2),
    label = 'TTX + TEA'
)
plt.xlim(Kslow_xlim)
plt.ylim(-100, 550)
pltools.add_scalebar(
    'ms', 'pA', x_on_left = False, anchor = (0.3, 0.35), y_size = 250,
    bar_space = 0, x_label_space = 0.02
)
plt.legend()

plt.subplot(spec_Kslow_traces[1, 1])
plt.plot(
    baseline.t_mat[1, :, -3],
    baseline[1, :, -3],
    linewidth = 0.5, color = 'gray'
)
plt.xlim(Kslow_xlim)
plt.annotate('-30mV', (5500, -28), ha = 'right', va = 'bottom')
pltools.add_scalebar(omit_x = True, y_units = 'mV', anchor = (-0.03, 0.2))


plt.subplot(spec_Kslow_gating[0, 0])
plt.title('\\textbf{{B1}} Steady-state conductance', loc = 'left')

x_ss = ss_pdata[1, :, :].mean(axis = 1)
y_mean_ss = np.mean(ss_pdata[0, :, :] / ss_pdata[0, -1, :], axis = 1)
y_std_ss = np.std(ss_pdata[0, :, :] / ss_pdata[0, -1, :], axis = 1)
plt.fill_between(
    x_ss, y_mean_ss - y_std_ss, y_mean_ss + y_std_ss,
    color = (0.9, 0.2, 0.2), alpha = 0.5
)
plt.plot(
    x_ss, y_mean_ss,
    color = (0.9, 0.2, 0.2), linewidth = 2,
    label = 'Activation ($n$)'
)
plt.plot(
    ss_fittedpts[1, :],
    ss_fittedpts[0, :],
    color = 'gray', lw = 2, ls = '--', dashes = (5, 2), alpha = 0.8
)
plt.legend()
plt.ylabel('$g/g_{{-20\mathrm{{mV}}}}$')
plt.xlabel('Voltage (mV)')
pltools.hide_border('tr')

plt.subplot(spec_Kslow_gating[0, 1])
plt.title('\\textbf{{B2}} $g_{{\mathrm{{max}}}}$', loc = 'left')
sns.swarmplot(y = ss_pdata[0, -1, :], color = (0.9, 0.2, 0.2))
plt.ylim(0, plt.ylim()[1])
plt.ylabel('$g_{{-20\mathrm{{mV}}}}$')
pltools.hide_border('trb')
plt.gca().set_xticks([])



plt.subplot(spec_IA_traces[0, 0])
plt.title('\\textbf{{C1}} A-type K-current', loc = 'left')
IA_tr_xlim = (5600, 5800)
baseline_subtracted = subtract_baseline(deepcopy(beautiful_gating_1), slice(56000, 56050),0)
baseline_subtracted.set_dt(0.1)
plt.plot(
    baseline_subtracted.t_mat[0, :, -3::-4],
    baseline_subtracted[0, :, -3::-4], 'k-',
    linewidth = 0.5)
plt.xlim(IA_tr_xlim)
plt.ylim(-100, 750)
pltools.add_scalebar(
    'ms', 'pA', x_on_left = False, anchor = (0.65, 0.5), x_label_space = 0.02,
    y_size = 300, bar_space = 0
)

plt.subplot(spec_IA_traces[1, 0])
plt.plot(
    beautiful_gating_1.t_mat[1, :, -3::-4],
    beautiful_gating_1[1, :, -3::-4],
    color = 'gray',
    linewidth = 0.5
)
plt.xlim(IA_tr_xlim)
plt.annotate('-20mV', (5800, -18), ha = 'right', va = 'bottom')
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (-0.03, 0.2))


plt.subplot(spec_IA_traces[0, 1])
plt.title('\\textbf{{C2}} Block by TEA', loc = 'left')
IA_pharm_xlim = (2600, 2750)
plt.plot(
    baseline.t_mat[0, :, -2],
    baseline[0, :, -2],
    linewidth = 0.5, color = 'k',
    label = 'TTX'
)
plt.plot(
    TEA_4AP.t_mat[0, :, -2],
    TEA_4AP[0, :, -2],
    linewidth = 0.5, color = (0.2, 0.2, 0.9),
    label = 'TTX + TEA/4AP'
)
plt.xlim(IA_pharm_xlim)
plt.ylim(-100, plt.ylim()[1])
pltools.add_scalebar('ms', 'pA', x_on_left = False, anchor = (-0.03, 0))
plt.legend()

plt.subplot(spec_IA_traces[1, 1])
plt.plot(
    baseline.t_mat[1, :, -1],
    baseline[1, :, -1],
    linewidth = 0.5, color = 'gray'
)
plt.xlim(IA_pharm_xlim)
plt.annotate('-20mV', (2750, -18), ha = 'right', va = 'bottom')
pltools.add_scalebar(omit_x = True, y_units = 'mV', anchor = (-0.03, 0.2))


plt.subplot(spec_IA_gating[0, 0])
plt.title('\\textbf{{D1}} Steady-state conductance', loc = 'left')

x_act = peakact_pdata[1, :, :].mean(axis = 1)
y_mean_act = np.mean(peakact_pdata[0, :, :] / peakact_pdata[0, -1, :], axis = 1)
y_std_act = np.std(peakact_pdata[0, :, :] / peakact_pdata[0, -1, :], axis = 1)

x_inact = peakinact_pdata[1, :, :].mean(axis = 1)
y_mean_inact = np.mean(peakinact_pdata[0, :, :] / peakinact_pdata[0, 0, :], axis = 1)
y_std_inact = np.std(peakinact_pdata[0, :, :] / peakinact_pdata[0, 0, :], axis = 1)

plt.fill_between(
    x_act, y_mean_act - y_std_act, y_mean_act + y_std_act,
    color = (0.2, 0.2, 0.9), alpha = 0.5
)
plt.fill_between(
    x_inact, y_mean_inact - y_std_inact, y_mean_inact + y_std_inact,
    color = (0.2, 0.8, 0.2), alpha = 0.5
)
plt.plot(
    x_act, y_mean_act,
    color = (0.2, 0.2, 0.9), linewidth = 2,
    label = 'Activation ($m$)'
)
plt.plot(
    peakact_fittedpts[1, :],
    peakact_fittedpts[0, :],
    color = 'gray', lw = 2, ls = '--', dashes = (5, 2), alpha = 0.8
)
plt.plot(
    x_inact, y_mean_inact,
    color = (0.2, 0.8, 0.2), linewidth = 2,
    label = 'Inactivation ($h$)'
)
plt.plot(
    peakinact_fittedpts[1, :],
    peakinact_fittedpts[0, :],
    color = 'gray', lw = 2, ls = '--', dashes = (5, 2), alpha = 0.8
)
plt.legend()
plt.ylabel('$g/g_{{\mathrm{{ref}}}}$')
plt.xlabel('Voltage (mV)')
pltools.hide_border('tr')

plt.subplot(spec_IA_gating[0, 1])
plt.title('\\textbf{{D2}} $g_{{\mathrm{{max}}}}$', loc = 'left')
sns.swarmplot(y = peakact_pdata[0, -1, :], color = (0.2, 0.2, 0.9))
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.ylabel('$g_{{-20\mathrm{{mV}}}}$')
pltools.hide_border('tbr')
plt.gca().set_xticks([])

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'perithresh_cond_characterization.png', dpi = 300)

plt.show()
