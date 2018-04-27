#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

import sys
sys.path.append('./analysis/gating/')
sys.path.append('./figs/scripts')

from cell_class import Cell, Recording
import pltools


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
sweep_to_use        = 11
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
    Vtest_step = (cell[1, xrange_baseline, :].mean(axis = 0) - cell[1, xrange_testpulse, :].mean(axis = 0)).mean()
    Itest_step = (cell[0, xrange_baseline, :].mean(axis = 0) - cell[0, xrange_testpulse, :].mean(axis = 0)).mean()
    Rm = Vtest_step/Itest_step

    I_leak = cell[1, :, :] / Rm

    TEA_washin_pdata[i, :, :cell.shape[2]] = cell[0, :, :] - I_leak
    TEA_washin_pdata[i, :, :] -= np.nanmean(TEA_washin_pdata[i, xrange_baseline, :], axis = 0)
    #TEA_washin_pdata[i, :, :] /= np.nanmean(TEA_washin_pdata[i, xrange_ss, :6])

TEA_washin_pdata = np.nanmean(TEA_washin_pdata[:, xrange_ss, :], axis = 1).T


# 4AP washin
# Probably won't use...
xrange_baseline = slice(10000, 11000)
xrange_ss = slice(50000, 51000)
TEA_4AP_washin_pdata = np.empty((len(TEA_4AP_washin), TEA_4AP_washin[0].shape[1], 44))
TEA_4AP_washin_pdata[:, :] = np.NAN

for i, cell in enumerate(TEA_4AP_washin):
    TEA_4AP_washin_pdata[i, :, :cell.shape[2]] = cell[0, :, :]
    TEA_4AP_washin_pdata[i, :, :] -= np.nanmean(TEA_4AP_washin_pdata[i, xrange_baseline, :], axis = 0)
    #TEA_4AP_washin_pdata[i, :, :] /= np.nanmean(TEA_4AP_washin_pdata[i, xrange_ss, :6])

TEA_4AP_washin_pdata = np.nanmean(TEA_4AP_washin_pdata[:, xrange_ss, :], axis = 1).T

#%% PROCESS GATING DATA



#%% MAKE FIGURE

plt.figure(figsize = (14.67, 18))

grid_dims = (3, 4)

# A: pharmacology
Iax, cmdax = pltools.subplots_in_grid((3, 2), (0, 0), ratio = 4)
Iax.set_title('A1 Pharmacology example traces', loc = 'left')
Iax.set_ylim(-50, 750)
Iax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
baseline_sweep,
'k-', linewidth = 0.5,
label = 'Baseline'
)
Iax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
TEA_sweep,
'-', linewidth = 0.5, color = (0.8, 0.2, 0.2),
label = '20mM TEA'
)
Iax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
TEA_4AP_sweep,
'-', linewidth = 0.5, color = (0.2, 0.2, 0.8),
label = '20mM TEA + 3mM 4AP'
)
Iax.legend()
pltools.add_scalebar(x_units = 'ms', y_units = 'pA', anchor = (0.6, 0.4), ax = Iax)

cmdax.plot(
np.arange(0, len(baseline_sweep)/10, 0.1),
cmd_sweep,
'k-', linewidth = 0.5
)
pltools.hide_border()
pltools.hide_ticks()

plt.subplot2grid((6, 2), (0, 1))
plt.title('A2 TEA washin', loc = 'left')
plt.plot(TEA_washin_pdata, '-', color = (0.8, 0.2, 0.2))
plt.axhline(0, linewidth = 0.5, linestyle = 'dashed')

plt.subplot2grid((6, 2), (1, 1))
plt.title('A3 4AP washin', loc = 'left')

# B: kinetics
Iax, cmdax = pltools.subplots_in_grid((3, 2), (1, 0), ratio = 4)
Iax.set_title('B1 Pharmacology example traces', loc = 'left')
pltools.hide_ticks(ax = Iax)
pltools.hide_ticks(ax = cmdax)

plt.subplot2grid((6, 4), (2, 2))
plt.title('B2 $\\bar{{g}}_{{k1}}$ histogram', loc = 'left')
plt.xlabel('$\\bar{{g}}_{{k1}}$')
pltools.hide_border(sides = 'rlt')
plt.yticks([])

plt.subplot2grid((6, 4), (2, 3))
plt.title('B3 $g_{{k1}}$ fitted curves', loc = 'left')
plt.ylabel('$g_{{k1}}/\\bar{{g}}_{{k1}}$')
plt.xlabel('$V$ (mV)')

plt.subplot2grid((6, 4), (3, 2))
plt.title('B4 $\\bar{{g}}_{{k2}}$ histogram', loc = 'left')
plt.xlabel('$\\bar{{g}}_{{k2}}$')
pltools.hide_border(sides = 'rlt')
plt.yticks([])

plt.subplot2grid((6, 4), (3, 3))
plt.title('B5 $g_{{k2}}$ fitted curves', loc = 'left')
plt.ylabel('$g_{{k2}}/\\bar{{g}}_{{k2}}$')
plt.xlabel('$V$ (mV)')

# C: model
plt.subplot2grid((3, 3), (2, 0))
plt.title('C1 Model', loc = 'left')
pltools.hide_ticks()

plt.subplot2grid((3, 3), (2, 1))
plt.title('C2 Gating response of model', loc = 'left')
pltools.hide_ticks()

plt.subplot2grid((3, 3), (2, 2))
plt.title('C3 Simulated voltage step', loc = 'left')
pltools.hide_ticks()

plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
plt.show()
