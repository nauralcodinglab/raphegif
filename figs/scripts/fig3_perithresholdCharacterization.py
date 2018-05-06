#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

import sys
sys.path.append('./src/')
sys.path.append('./analysis/gating/')
sys.path.append('./figs/scripts')

from SubthreshGIF_K import SubthreshGIF_K
from cell_class import Cell
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
TEA = Cell().read_ABF('./figs/figdata/18411013.abf')[0]
TEA_4AP = Cell().read_ABF('./figs/figdata/18411015.abf')[0]

# Load drug washin files
TEA_washin = Cell().read_ABF([FIGDATA_PATH + '18411020.abf',
                              FIGDATA_PATH + '18411012.abf',
                              FIGDATA_PATH + '18412002.abf'])
TEA_4AP_washin = Cell().read_ABF([FIGDATA_PATH + '18411022.abf',
                                  FIGDATA_PATH + '18411014.abf'])

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
                          GATING_PATH + 'c6_inact_18213020.abf'])


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

#%% SIMULATE GATING

gatingGIF = SubthreshGIF_K(0.1)

gatingGIF_params = {
'C': 70,
'gl': 1,
'El': -70,
'gbar_K1': 50,
'm_tau': 1.,
'h_tau': 50.,
'gbar_K2': 50,
'n_tau': 100.
}

# Define parameters
gatingGIF.C = gatingGIF_params['C'] * 1e-3
gatingGIF.gl = gatingGIF_params['gl'] * 1e-3
gatingGIF.El = gatingGIF_params['El']

gatingGIF.gbar_K1 = gatingGIF_params['gbar_K1'] * 1e-3
gatingGIF.m_Vhalf = peakact_params[2]
gatingGIF.m_k = peakact_params[1]
gatingGIF.m_tau = gatingGIF_params['m_tau']

gatingGIF.h_Vhalf = peakinact_params[2]
gatingGIF.h_k = peakinact_params[1]
gatingGIF.h_tau = gatingGIF_params['h_tau']

gatingGIF.gbar_K2 =  gatingGIF_params['gbar_K2'] * 1e-3
gatingGIF.n_Vhalf = ss_params[2]
gatingGIF.n_k = ss_params[1]
gatingGIF.n_tau =  gatingGIF_params['n_tau']

gatingGIF.E_K = -101.

prepad = 50
V_pre = -90
V_const = -35
simulated_Vclamp = list(gatingGIF.simulateVClamp(400, V_const, V_pre))
simulated_Vclamp[0] = np.concatenate((np.ones(prepad * 10) * V_pre, simulated_Vclamp[0]))
simulated_Vclamp[1] = np.concatenate((np.ones(prepad * 10) * simulated_Vclamp[1][0], simulated_Vclamp[1]))

simulated_g = {
'm': gatingGIF.computeGating(
simulated_Vclamp[0], gatingGIF.mInf(simulated_Vclamp[0]), gatingGIF_params['m_tau']
),
'h': gatingGIF.computeGating(
simulated_Vclamp[0], gatingGIF.hInf(simulated_Vclamp[0]), gatingGIF_params['h_tau']
),
'n': gatingGIF.computeGating(
simulated_Vclamp[0], gatingGIF.nInf(simulated_Vclamp[0]), gatingGIF_params['n_tau']
)
}

#%% MAKE LATEX MODEL DEFINITION

gkfast_latex = '$g_{{Kfast}} = \\bar{{g}}_{{Kfast}}mh \\times (V(t) - E_k)$'
gkslow_latex = '$g_{{Kslow}} = \\bar{{g}}_{{Kslow}}n \\times (V(t) - E_k)$'

minf_latex = '$m_{{\infty}}(V) = \\frac{{m_{{max}}}}{{1 + \exp{{[-k(V - V_0)]}}}}$'
mdot_latex = '$\dot{{m}}(t) = \\frac{{m(t) - m_{{\infty}}(V)}}{{\\tau_m}}$'


#%% MAKE FIGURE

mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]
mpl.rc('text', usetex = True)
mpl.rc('svg', fonttype = 'none')

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


IMG_PATH = './figs/ims/'

plt.figure(figsize = (14.67, 12))

spec = gridspec.GridSpec(
6, 3, height_ratios = (0.75, 0.25, 0.75, 0.25, 0.75, 0.25),
left = 0.05, bottom = 0.05, right = 0.95, top = 0.95,
wspace = 0.3, hspace = 0.9
)

m_color = (0.2, 0.2, 0.8)
h_color = (0.2, 0.8, 0.2)
n_color = (0.8, 0.2, 0.2)
simlinewidth = 2

# A: pharmacology

inset_pos_ll = (80, -45)
inset_pos_ur = (250, 500)

Iax = plt.subplot(spec[2, 0])
cmdax = plt.subplot(spec[3, 0])
pltools.join_plots(Iax, cmdax)
Iax.set_title('\\textbf{{B1}} $K_{{slow}}$ pharmacology', loc = 'left')
Iax.set_ylim(-10, 170)
Iax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
baseline_sweep,
'k-', linewidth = 2,
label = 'TTX (baseline)'
)
Iax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
TEA_sweep,
'-', linewidth = 2, color = n_color,
label = 'TTX + TEA'
)
Iax.legend()
pltools.add_scalebar(x_units = 'ms', y_units = 'pA', anchor = (0.9, 0.6), text_spacing = (0.02, -0.02), bar_spacing = 0, ax = Iax)
cmdax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
cmd_sweep,
'-', linewidth = 2, color = 'gray'
)
pltools.hide_border()
pltools.hide_ticks()


Iax = plt.subplot(spec[0, 0])
cmdax = plt.subplot(spec[1, 0])
pltools.join_plots(Iax, cmdax)
Iax.set_title('\\textbf{{A1}} $K_{{fast}}$ pharmacology', loc = 'left')
Iax.set_ylim(inset_pos_ll[1], inset_pos_ur[1])
Iax.set_xlim(inset_pos_ll[0], inset_pos_ur[0])
Iax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
baseline_sweep,
'k-', linewidth = 2,
label = 'TTX (baseline)'
)
Iax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
TEA_4AP_sweep,
'-', linewidth = 2, color = m_color,
label = 'TTX + 4AP + TEA'
)
pltools.add_scalebar(x_units = 'ms', y_units = 'pA', anchor = (0.9, 0.4), text_spacing = (0.02, -0.02), bar_spacing = 0, ax = Iax)
Iax.legend()
cmdax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
cmd_sweep,
'-', linewidth = 2, color = 'gray'
)
cmdax.set_xlim(inset_pos_ll[0], inset_pos_ur[0])
pltools.hide_border()
pltools.hide_ticks()

"""
plt.subplot(spec[2:4, 1])
plt.title('\\textbf{{A2}} TEA washin', loc = 'left')
plt.plot(
np.arange(-1, TEA_washin_pdata.shape[0] / 6. - 1, 1./6.),
TEA_washin_pdata,
'-', color = n_color
)
plt.ylim(0, plt.ylim()[1])
plt.ylabel('Leak-subtracted current (pA)')
plt.xlabel('Time from TEA washin (min)')
pltools.hide_border('tr')

plt.subplot(spec[:2, 1])
plt.title('\\textbf{{A3}} 4AP washin', loc = 'left')
plt.plot(
np.broadcast_to(np.arange(-1, TEA_4AP_washin_pdata.shape[0] / 6. - 1, 1./6.).reshape((-1, 1)), TEA_4AP_washin_pdata.shape),
TEA_4AP_washin_pdata,
'-', color = m_color
)
plt.axhline(0, color = 'k', linewidth = 0.5, linestyle = 'dashed')
plt.ylabel('Leak-subtracted current (pA)')
plt.xlabel('Time from 4AP washin (min)')
"""

# B: kinetics

plt.subplot(spec[:2, 1:])
plt.title('\\textbf{{A2}} $K_{{fast}}$ steady-state voltage dependence', loc = 'left')
peakact_y_mean = max_normalize(peakact_pdata[0, :, :]).mean(axis = 1)
peakact_y_std = max_normalize(peakact_pdata[0, :, :]).std(axis = 1)
plt.fill_between(
peakact_pdata[1, :, :].mean(axis = 1),
peakact_y_mean - peakact_y_std, peakact_y_mean + peakact_y_std,
facecolor = m_color, edgecolor = 'none', alpha = 0.3
)
plt.plot(
peakact_pdata[1, :, :].mean(axis = 1),
peakact_y_mean,
'-', color = m_color,
label = 'Activation (mean $\pm$ SD)'
)
plt.plot(
peakact_fittedpts[1, :],
peakact_fittedpts[0, :],
'--', linewidth = 2, color = 'gray',
label = 'Fitted'
)
peakinact_y_mean = max_normalize(peakinact_pdata[0, :, :]).mean(axis = 1)
peakinact_y_std = max_normalize(peakinact_pdata[0, :, :]).std(axis = 1)
plt.fill_between(
peakinact_pdata[1, :, :].mean(axis = 1),
peakinact_y_mean - peakinact_y_std, peakinact_y_mean + peakinact_y_std,
facecolor = h_color, edgecolor = 'none', alpha = 0.3
)
plt.plot(
peakinact_pdata[1, :, :].mean(axis = 1),
peakinact_y_mean,
'-', color = h_color,
label = 'Inactivation (mean $\pm$ SD)'
)
plt.plot(
peakinact_fittedpts[1, :],
peakinact_fittedpts[0, :],
'--', linewidth = 2, color = 'gray'
)
plt.text(
0.95, 0.1, '$N = {}$ cells'.format(11),
horizontalalignment = 'right', transform = plt.gca().transAxes
)
plt.ylabel('$g/g_{{-20\mathrm{{mV}}}}$')
plt.xlabel('$V$ (mV)')
plt.legend()
pltools.hide_border('tr')

plt.subplot(spec[2:4, 1:])
plt.title('\\textbf{{B2}} $K_{{slow}}$ steady-state voltage dependence', loc = 'left')
ss_y_mean = np.delete(max_normalize(ss_pdata[0, :, :]), 4, axis = 1).mean(axis = 1) # Cell #4 has substantial negative conductance near -70
ss_y_std = np.delete(max_normalize(ss_pdata[0, :, :]), 4, axis = 1).std(axis = 1)
plt.fill_between(
ss_pdata[1, :, :].mean(axis = 1),
ss_y_mean - ss_y_std, ss_y_mean + ss_y_std,
facecolor = n_color, edgecolor = 'none', alpha = 0.3
)
plt.plot(
ss_pdata[1, :, :].mean(axis = 1),
ss_y_mean,
'-', color = n_color,
label = 'Activation (mean $\pm$ SD)'
)
plt.plot(
ss_fittedpts[1, :],
ss_fittedpts[0, :],
'--', linewidth = 2, color = 'gray',
label = 'Fitted'
)
plt.text(
0.95, 0.1, '$N = {}$ cells'.format(10),
horizontalalignment = 'right', transform = plt.gca().transAxes
)
plt.ylabel('$g/g_{{-20\mathrm{{mV}}}}$')
plt.xlabel('$V$ (mV)')
plt.legend()
pltools.hide_border('tr')

# C: model
plt.subplot(spec[4:6, 2])
plt.title('\\textbf{{C3}} Model definition', loc = 'left')
plt.text(
0, 0.5,
'\n'.join([gkfast_latex, gkslow_latex,
'Where $m$, $n$, $h$ have the form:',
minf_latex,
mdot_latex]),
transform = plt.gca().transAxes
)
pltools.hide_ticks()

gatingax = plt.subplot(spec[4, 1])
cmdax = plt.subplot(spec[5, 1])
pltools.join_plots(gatingax, cmdax)
gatingax.set_title('\\textbf{{C2}} Model gating dynamics', loc = 'left')
x = np.arange(0, int(len(simulated_g['m']) * 0.1), 0.1)
gatingax.plot(x, simulated_g['m'], '-', color = m_color, linewidth = simlinewidth, label = 'm ($K_{{fast}}$ activation)')
gatingax.plot(x, simulated_g['h'], '-', color = h_color, linewidth = simlinewidth, label = 'h ($K_{{fast}}$ inactivation)')
gatingax.plot(x, simulated_g['n'], '-', color = n_color, linewidth = simlinewidth, label = 'n ($K_{{slow}}$ activation)')
gatingax.set_ylim(-0.05, 1.05)
pltools.add_scalebar(x_units = 'ms', omit_y = True, remove_frame = False, anchor = (0.98, 0.35), ax = gatingax)
gatingax.set_xticks([])
gatingax.set_ylabel('$g/g_{{-20\mathrm{{mV}}}}$')
pltools.hide_border('rtb', ax = gatingax)
gatingax.legend()

cmdax.plot(np.arange(0, len(simulated_Vclamp[0]) / 10, 0.1), simulated_Vclamp[0], '-', color = 'gray', linewidth = simlinewidth)
#cmdax.set_ylabel('Simulated $V_{{cmd}}$', rotation = 'horizontal')
pltools.hide_ticks()
pltools.hide_border()

Iax = plt.subplot(spec[4, 0])
cmdax = plt.subplot(spec[5, 0])
pltools.join_plots(Iax, cmdax)
Iax.set_title('\\textbf{{C1}} Simulated voltage step', loc = 'left')
Iax.plot(np.arange(0, len(simulated_Vclamp[1]) / 10., 0.1), simulated_Vclamp[1] * 1e3, 'k-', linewidth = simlinewidth)
"""
Iax.text(0, 0,
'Simulated neuron parameters:'
'\n$C = {C}$pF'
'\n$g_l = {gl}$pS'
'\n$E_l = {El}$mV'
'\n$\\tau_m = {m_tau}$ms'
'\n$\\tau_h = {h_tau}$ms'
'\n$\\tau_n = {n_tau}$ms'.format(**gatingGIF_params))
"""
pltools.add_scalebar(x_units = 'ms', y_units = 'pA', anchor = (0.9, 0.2), text_spacing = (0.02, -0.02), bar_spacing = 0, ax = Iax, remove_frame = False)
pltools.hide_border(ax = Iax)
Iax.set_xticks([])
Iax.set_yticks([])
#Iax.set_ylabel('Simulated $I$', rotation = 'horizontal')

cmdax.plot(np.arange(0, len(simulated_Vclamp[0]) / 10, 0.1), simulated_Vclamp[0], '-', color = 'gray', linewidth = simlinewidth)
#cmdax.set_ylabel('Simulated $V_{{cmd}}$', rotation = 'horizontal')
pltools.hide_ticks()
pltools.hide_border()

#plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, hspace = 0.4, wspace = 0.4)
plt.savefig(IMG_PATH + 'fig3_perithresholdCharacterization2.png', dpi = 300)
plt.show()
