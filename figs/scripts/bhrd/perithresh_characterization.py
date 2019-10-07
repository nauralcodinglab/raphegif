#%% IMPORT MODULES

from __future__ import division

import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes, mark_inset
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

import sys
sys.path.append('./analysis/gating/')


from grr.cell_class import Cell, Recording
import src.pltools as pltools


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
GATING_PATH = './data/raw/5HT/gating/'

# Load V-steps files for sample pharma traces.
baseline = Cell().read_ABF('./figs/figdata/18411010.abf')[0]
baseline.set_dt(0.1)
TEA = Cell().read_ABF('./figs/figdata/18411013.abf')[0]
TEA.set_dt(0.1)
TEA_4AP = Cell().read_ABF('./figs/figdata/18411015.abf')[0]
TEA_4AP.set_dt(0.1)

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


#%% SUBTRACT BASELINE & LEAK

beautiful_gating_1 = subtract_baseline(beautiful_gating_1, slice(1000, 2000), 0)
beautiful_gating_1 = subtract_leak(beautiful_gating_1, slice(1000, 2000), slice(3000, 3400))
beautiful_gating_1.set_dt(0.1)

baseline = subtract_baseline(baseline, slice(1000, 2000), 0)
baseline = subtract_leak(baseline, slice(1000, 2000), slice(3000, 3400))
baseline.set_dt(0.1)

TEA_4AP = subtract_baseline(TEA_4AP, slice(1000, 2000), 0)
TEA_4AP = subtract_leak(TEA_4AP, slice(1000, 2000), slice(3000, 3400))
TEA_4AP.set_dt(0.1)

#%% MAKE FIGURE

def dashed_border(ax = None):

    if ax is None:
        ax = plt.gca()

    for side in ['right', 'left', 'top', 'bottom']:
        ax.spines[side].set_linestyle('--')
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_edgecolor('gray')

IA_color = (0.2, 0.2, 0.9)
IA_inact_color = (0.1, 0.65, 0.1)
Kslow_color = (0.8, 0.1, 0.1)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/thesis/'

spec_outer          = gs.GridSpec(2, 1, top = 0.95, bottom = 0.1, right = 0.98, left = 0.1, hspace = 0.4)
spec_traces         = gs.GridSpecFromSubplotSpec(2, 3, spec_outer[0, :], height_ratios = [1, 0.2], hspace = 0)
spec_gating_outer   = gs.GridSpecFromSubplotSpec(1, 2, spec_outer[1, :], wspace = 0.4)
spec_gating_IA      = gs.GridSpecFromSubplotSpec(1, 2, spec_gating_outer[:, 0], width_ratios = [1, 0.2], wspace = 0.35)
spec_gating_Kslow   = gs.GridSpecFromSubplotSpec(1, 2, spec_gating_outer[:, 1], width_ratios = [1, 0.2], wspace = 0.35)

plt.figure(figsize = (6.5, 5))

act_ax = plt.subplot(spec_traces[0, :2])
plt.title('\\textbf{{A}} Outward currents in 5HT neurons', loc = 'left')
Kslow_xlim = (2450, 8500)
plt.plot(
    beautiful_gating_1.t_mat[0, :, -2::-3],
    beautiful_gating_1[0, :, -2::-3], 'k-',
    linewidth = 0.5
)
plt.text(4500, 200, '+TTX', ha = 'center', va = 'center', size = 'small')
plt.annotate(
    '$n$', (5500, 170),
    xytext = (-10, 20), textcoords = 'offset points', ha = 'center',
    arrowprops = {'arrowstyle': '->'}
)
plt.xlim(Kslow_xlim)
plt.ylim(-100, 800)
pltools.add_scalebar(
    'ms', 'pA', anchor = (0.8, 0.1),
    x_label_space = -0.02, y_size = 100, x_size = 500, bar_space = 0
)

act_ins = inset_axes(act_ax, '20%', '40%', loc = 'upper left', borderpad = 2)
act_ins.plot(
    beautiful_gating_1.t_mat[0, :, -2::-3],
    beautiful_gating_1[0, :, -2::-3], 'k-',
    linewidth = 0.5
)
plt.annotate(
    '$m$', (2625, 705),
    xytext = (15, -10), textcoords = 'offset points',
    arrowprops = {'arrowstyle': '->'}
)
dashed_border(act_ins)
#pltools.add_scalebar(y_units = 'pA', x_units = 'ms', remove_frame = False, ax = act_ins)
pltools.hide_ticks()
plt.xlim(2590, 2750)
plt.ylim(-50, 780)
mark_inset(act_ax, act_ins, loc1 = 1, loc2 = 4, ls = '--', color = 'gray', lw = 0.5)

inact_ins = inset_axes(act_ax, '20%', '40%', loc = 'upper right', borderpad = 2)
inact_ins.plot(
    beautiful_gating_1.t_mat[0, :, -3::-3],
    beautiful_gating_1[0, :, -3::-3], 'k-',
    linewidth = 0.5
)
plt.annotate(
    '$h$', (5630, 600),
    xytext = (15, -10), textcoords = 'offset points',
    arrowprops = {'arrowstyle': '->'}
)
dashed_border(inact_ins)
#pltools.add_scalebar(y_units = 'pA', x_units = 'ms', remove_frame = False, ax = inact_ins)
pltools.hide_ticks()
plt.xlim(5550, 5800)
plt.ylim(-50, 750)
mark_inset(act_ax, inact_ins, loc1 = 1, loc2 = 4, ls = '--', color = 'gray', lw = 0.5)

plt.subplot(spec_traces[1, :2])
plt.plot(
    beautiful_gating_1.t_mat[1, :, -2::-3],
    beautiful_gating_1[1, :, -2::-3], '-',
    color = 'gray', linewidth = 0.5
)
plt.xlim(Kslow_xlim)
plt.text(Kslow_xlim[1], -22, '$-20$mV', ha = 'right', va = 'top', size = 'small')
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (0.8, 0.1))

pharm_ax = plt.subplot(spec_traces[0, 2])
plt.title('\\textbf{{B}} Pharmacology', loc = 'left')
plt.plot(
    baseline.t_mat[0, :, -3],
    baseline[0, :, -3],
    linewidth = 0.5, color = 'k',
    label = 'Baseline'
)
plt.plot(
    TEA_4AP.t_mat[0, :, -3],
    TEA_4AP[0, :, -3],
    linewidth = 0.5, color = (0.9, 0.2, 0.2),
    label = 'TEA + 4AP'
)
plt.text(3800, 200, '+TTX', ha = 'center', va = 'center', size = 'small')
plt.axhline(0, ls = '--', lw = 0.5, color = 'k', dashes = (10, 10))
plt.xlim(Kslow_xlim[0], 4500)
plt.ylim(-100, 550)
pltools.add_scalebar(
    'ms', 'pA', anchor = (1, 0.35), y_size = 100, x_size = 500,
    bar_space = 0, x_label_space = -0.02
)
plt.legend()

pharm_ins = inset_axes(pharm_ax, '30%', '50%', loc = 'upper left', borderpad = 2.2)
plt.plot(
    baseline.t_mat[0, :, -3],
    baseline[0, :, -3],
    linewidth = 0.5, color = 'k',
    label = 'Baseline'
)
plt.plot(
    TEA_4AP.t_mat[0, :, -3],
    TEA_4AP[0, :, -3],
    linewidth = 0.5, color = (0.9, 0.2, 0.2),
    label = 'TEA + 4AP'
)
plt.xlim(Kslow_xlim[0] + 100, Kslow_xlim[0] + 300)
plt.ylim(-70, 500)
dashed_border(pharm_ins)
pltools.hide_ticks()
mark_inset(pharm_ax, pharm_ins, loc1 = 1, loc2 = 4, ls = '--', color = 'gray', lw = 0.5)

plt.subplot(spec_traces[1, 2])
plt.plot(
    baseline.t_mat[1, :, -3],
    baseline[1, :, -3],
    linewidth = 0.5, color = 'gray'
)
pltools.add_scalebar(y_units = 'mV', anchor = (0.5, 0.1), omit_x = True, y_size = 25)
plt.text(4500, -32, '$-30$mV', ha = 'right', va = 'top', size = 'small')
plt.xlim(Kslow_xlim[0], 4500)

plt.subplot(spec_gating_IA[:, 0])
plt.title('\\textbf{{C1}} $I_A$ gating', loc = 'left')

x_act = peakact_pdata[1, :, :].mean(axis = 1)
y_mean_act = np.mean(peakact_pdata[0, :, :] / peakact_pdata[0, -1, :], axis = 1)
y_std_act = np.std(peakact_pdata[0, :, :] / peakact_pdata[0, -1, :], axis = 1)

x_inact = peakinact_pdata[1, :, :].mean(axis = 1)
y_mean_inact = np.mean(peakinact_pdata[0, :, :] / peakinact_pdata[0, 0, :], axis = 1)
y_std_inact = np.std(peakinact_pdata[0, :, :] / peakinact_pdata[0, 0, :], axis = 1)

plt.fill_between(
    x_act, y_mean_act - y_std_act, y_mean_act + y_std_act,
    color = IA_color, alpha = 0.5
)
plt.fill_between(
    x_inact, y_mean_inact - y_std_inact, y_mean_inact + y_std_inact,
    color = IA_inact_color, alpha = 0.5
)
plt.plot(
    x_act, y_mean_act,
    color = IA_color, linewidth = 2,
    label = 'Activation ($m$)'
)
plt.plot(
    peakact_fittedpts[1, :],
    peakact_fittedpts[0, :],
    color = 'gray', lw = 2, ls = '--', dashes = (5, 2), alpha = 0.8,
    label = 'Fitted'
)
plt.plot(
    x_inact, y_mean_inact,
    color = IA_inact_color, linewidth = 2,
    label = 'Inactivation ($h$)'
)
plt.plot(
    peakinact_fittedpts[1, :],
    peakinact_fittedpts[0, :],
    color = 'gray', lw = 2, ls = '--', dashes = (5, 2), alpha = 0.8
)

plt.ylabel('$g/g_\mathrm{{ref}}$')
plt.xlabel('$V$ (mV)')
pltools.hide_border('tr')
plt.legend()

plt.subplot(spec_gating_IA[:, 1])
plt.title('\\textbf{{C2}}', loc = 'left')
plt.ylim(0, 18)
sns.swarmplot(y = peakact_pdata[0, -1, :], color = IA_color, edgecolor = 'gray', linewidth = 0.5)
plt.xticks([])
plt.gca().set_yticks([0, 5, 10, 15])
pltools.hide_border('trb')
plt.ylabel('$g_\mathrm{{ref}}$ (nS)')

plt.subplot(spec_gating_Kslow[:, 0])
plt.title('\\textbf{{D1}} $K_\mathrm{{slow}}$ gating', loc = 'left')

x_ss = ss_pdata[1, :, :].mean(axis = 1)
y_mean_ss = np.mean(ss_pdata[0, :, :] / ss_pdata[0, -1, :], axis = 1)
y_std_ss = np.std(ss_pdata[0, :, :] / ss_pdata[0, -1, :], axis = 1)

plt.fill_between(
    x_ss, y_mean_ss - y_std_ss, y_mean_ss + y_std_ss,
    color = Kslow_color, alpha = 0.5
)
plt.plot(
    x_ss, y_mean_ss,
    color = Kslow_color, linewidth = 2,
    label = 'Activation ($n$)'
)
plt.plot(
    ss_fittedpts[1, :],
    ss_fittedpts[0, :],
    color = 'gray', lw = 2, ls = '--', dashes = (5, 2), alpha = 0.8,
    label = 'Fitted'
)
plt.ylabel('$g/g_\mathrm{{ref}}$')
plt.xlabel('$V$ (mV)')
pltools.hide_border('tr')
plt.legend()

plt.subplot(spec_gating_Kslow[:, 1])
plt.title('\\textbf{{D2}}', loc = 'left')
plt.ylim(0, 2.9)
sns.swarmplot(y = ss_pdata[0, -1, :], color = Kslow_color, edgecolor = 'gray', linewidth = 0.5)
plt.xticks([])
plt.gca().set_yticks([0, 1, 2])
pltools.hide_border('trb')
plt.ylabel('$g_\mathrm{{ref}}$ (nS)')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'perithresh_cond_char.png')

plt.show()


#%% PRINT SUMMARY STATISTICS

def printconf(label, data, units = None):
    print(
        '{:>10}: {:>10.2f} +- {:>10.2f} {}; Shapiro W={:>5.3f} p={:>5.3f}'.format(
            label,
            np.nanmean(data),
            np.nanstd(data),
            units,
            stats.shapiro(data)[0],
            stats.shapiro(data)[1]
        )
    )

stats.pearsonr(peakact_pdata[0, -1, :], ss_pdata[0, -1, :])
printconf('gA', peakact_pdata[0, -1, :], 'nS')
printconf('gslow', ss_pdata[0, -1, :], 'nS')

#%% MAKE SUPPLEMENTARY FIGURE

def plot_best_fit(x, y):

    x = np.unique(x)
    y = np.poly1d(np.polyfit(x, y, 1))(np.unique(x))

    plt.plot(x, y, 'k-', label = 'Best fit')

supp_spec = gs.GridSpec(1, 2, wspace = 0.4)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

plt.figure(figsize = (6, 3))

plt.subplot(supp_spec[:, 0])
plt.title('\\textbf{{A}} Correlation at --20mV', loc = 'left')

plot_best_fit(ss_pdata[0, -1, :], peakact_pdata[0, -1, :])
plt.text(
    0.95, 0.05,
    '$R = {:.3f}$'.format(stats.pearsonr(peakact_pdata[0, -1, :], ss_pdata[0, -1, :])[0]),
    ha = 'right', transform = plt.gca().transAxes
)
plt.plot(ss_pdata[0, -1, :], peakact_pdata[0, -1, :], 'o', color = 'gray', markeredgecolor = 'k')
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.xlim(0, plt.xlim()[1] * 1.1)
plt.ylabel('$I_A$ conductance (nS)')
plt.xlabel('$K_\mathrm{{slow}}$ conductance (nS)')
pltools.hide_border('tr')
plt.legend(loc = 'upper left')


plt.subplot(supp_spec[:, 1])
plt.title('\\textbf{{B}} Correlation at --40mV', loc = 'left')

plot_best_fit(ss_pdata[0, -5, :], peakact_pdata[0, -5, :])
plt.text(
    0.95, 0.05,
    '$R = {:.3f}$'.format(stats.pearsonr(peakact_pdata[0, -5, :], ss_pdata[0, -5, :])[0]),
    ha = 'right', transform = plt.gca().transAxes
)
plt.plot(ss_pdata[0, -5, :], peakact_pdata[0, -5, :], 'o', color = 'gray', markeredgecolor = 'k')
plt.ylim(0, plt.ylim()[1] * 1.1)
plt.ylabel('$I_A$ conductance (nS)')
plt.xlabel('$K_\mathrm{{slow}}$ conductance (nS)')
pltools.hide_border('tr')
plt.legend()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'cond_corr.png')

plt.show()
