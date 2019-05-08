#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd
from scipy import stats

import sys
sys.path.append(os.path.join('analysis', 'gating'))
import src.pltools as pltools
from src.cell_class import Cell
from gating_tools import *


#%% LOAD DATA

### Load/preprocess membrane parameters data.
params = pd.read_csv(os.path.join(
    'data', 'current_steps', 'GABA', 'index.csv'
))
params['El_est'] = -params['hold'] * params['R'] * 1e-3 - 70

### Load/preprocess voltage steps.

GABA_PATH = os.path.join('data', 'GABA_cells')
gating = Cell().read_ABF([os.path.join(GABA_PATH, 'matched_I_V_steps', '18n22000.abf'),
                          os.path.join(GABA_PATH, 'matched_I_V_steps', '18n22003.abf'),
                          os.path.join(GABA_PATH, 'matched_I_V_steps', '18n22005.abf'),
                          os.path.join(GABA_PATH, 'matched_I_V_steps', '18n22008.abf'),
                          os.path.join(GABA_PATH, 'matched_I_V_steps', '18n22013.abf'),
                          os.path.join(GABA_PATH, 'matched_I_V_steps', '18n22017.abf'),
                          os.path.join(GABA_PATH, 'unmatched_V_steps', '18n16004.abf'),
                          os.path.join(GABA_PATH, 'unmatched_V_steps', '18n16005.abf'),
                          os.path.join(GABA_PATH, 'unmatched_V_steps', '18n22015.abf'),
                          os.path.join(GABA_PATH, 'DRN393_firing_vsteps', '18n16003.abf'),
                          os.path.join(GABA_PATH, 'DRN398_firing_vsteps', '18n16015.abf')])

# Subtract baseline/leak from gating recordings.
processed = []
for rec in gating:
    tmp = subtract_baseline(rec, slice(1000, 2000), 0)
    tmp = subtract_leak(tmp, slice(1000, 2000), slice(3000, 3400))
    processed.append(tmp)
del tmp, rec

# Load/preprocess long current steps.

LONG_CURR_PATH = os.path.join('data', 'long_curr_steps', 'GABA')
long_curr = {
    'DRN422': Cell().read_ABF(os.path.join(LONG_CURR_PATH, '19114024.abf'))[0],
    'DRN431': Cell().read_ABF(os.path.join(LONG_CURR_PATH, '19121013.abf'))[0]
}
for key in long_curr.keys():
    long_curr[key].set_dt(0.1)


#%% CREATE FIGURE

plt.style.use(os.path.join('figs', 'scripts', 'bhrd', 'poster_mplrc.dms'))

IMG_PATH = os.path.join('figs', 'ims', '2019BHRD')
hist_color = 'gray'

plt.figure(figsize = (16,9))

# Define layout with GridSpec objects.
spec_outer = gs.GridSpec(3, 1)
spec_firstrow = gs.GridSpecFromSubplotSpec(
    1, 3, spec_outer[0, :], hspace = 0.45
)
spec_wccurr = gs.GridSpecFromSubplotSpec(
    2, 1, spec_firstrow[:, 2], hspace = 0.05, height_ratios = [1, 0.2]
)
spec_currsteprow = gs.GridSpecFromSubplotSpec(
    2, 2, spec_outer[1, :], hspace = 0.05, height_ratios = [1, 0.2]
)
spec_histrow = gs.GridSpecFromSubplotSpec(
    1, 4, spec_outer[2, :]
)

### Lay out first row.
plt.subplot(spec_firstrow[:, 0])
plt.title(r'\textbf{A}', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec_firstrow[:, 1])
plt.title(r'\textbf{B} Identification of SOM neurons', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec_wccurr[0, :])
plt.title(r'\textbf{C} Whole cell currents', loc = 'left')
for tr in processed:
    tr.set_dt(0.1)
    plt.plot(
        tr.t_mat[0, :10000, -1], tr[0, 25000:35000, -1],
        'k-', lw = 1., alpha = 0.8
    )
plt.text(
    0.95, 0.95,
    '$N$ = ${}$ cells'.format(len(processed)),
    ha = 'right', va = 'top',
    transform = plt.gca().transAxes
)
plt.ylabel('$I$ (pA)')
plt.ylim(-100, plt.ylim()[1])
plt.xticks([])

plt.subplot(spec_wccurr[1, :])
plt.plot(
    tr.t_mat[1, :10000, -1], tr[1, 25000:35000, -1],
    '-', lw = 1., color = 'gray'
)
plt.yticks([-20, -80], ['$-20$', '$-80$'])
plt.ylabel('$V$ (mV)')
plt.xlabel('Time (ms)')


### Create second row with sample current steps.
trace_xlim = (2500, 19000)
plt.subplot(spec_currsteprow[0, 0])
plt.title(r'\textbf{D1} Burst firing SOM neuron', loc = 'left')
plt.plot(
    long_curr['DRN422'].t_mat[0, :165000, 0] * 1e-3,
    long_curr['DRN422'][0, 25000:190000, 0],
    'k-', lw = 1.
)
plt.ylabel('$V$ (mV)')
plt.xticks([])

plt.subplot(spec_currsteprow[1, 0])
plt.plot(
    long_curr['DRN422'].t_mat[1, :165000, 0] * 1e-3,
    long_curr['DRN422'][1, 25000:190000, 0],
    '-', lw = 1., color = 'gray'
)
plt.ylabel('$I$ (pA)')
plt.xlabel('Time (s)')

plt.subplot(spec_currsteprow[0, 1])
plt.title(r'\textbf{D2} Delayed firing SOM neuron', loc = 'left')
plt.plot(
    long_curr['DRN431'].t_mat[0, :165000, 0] * 1e-3,
    long_curr['DRN431'][0, 25000:190000, 0],
    'k-', lw = 1.
)
plt.xticks([])

plt.subplot(spec_currsteprow[1, 1])
plt.plot(
    long_curr['DRN431'].t_mat[1, :165000, 0] * 1e-3,
    long_curr['DRN431'][1, 25000:190000, 0],
    '-', lw = 1., color = 'gray'
)
plt.xlabel('Time (s)')

### Create row with histograms.

def plot_hist(data, xlabel, color = 'gray', ax = None):

    if ax is None:
        ax = plt.gca()

    plt.hist(data, color = color)
    pltools.hide_border(sides = 'rlt')
    plt.yticks([])
    plt.xlabel(xlabel)
    plt.ylim(0, plt.ylim()[1] * 1.2)
    shapiro_w, shapiro_p = stats.shapiro(data)
    plt.text(
        0.98, 0.98,
        'Normality test {}'.format(pltools.p_to_string(shapiro_p)),
        va = 'top', ha = 'right',
        transform = ax.transAxes
    )
    plt.text(
        0.5, 0.02,
        '$N = {}$ cells'.format(
            len(data)),
        va = 'bottom', ha = 'center',
        transform = ax.transAxes
    )

# Leak conductance
ax = plt.subplot(spec_histrow[:, 0])
plt.title(r'\textbf{E1} Leak conductance', loc = 'left')
plot_hist(1e3/params['R'], r'$g_l$ (pS)', color = hist_color)

# Capacitance
ax = plt.subplot(spec_histrow[:, 1])
plt.title(r'\textbf{E2} Capacitance', loc = 'left')
plot_hist(params['C'], '$C_m$ (pF)', color = hist_color)

# Membrane time constant
ax = plt.subplot(spec_histrow[:, 2])
plt.title(r'\textbf{E3} Time constant', loc = 'left')
plot_hist(params['R'] * params['C'] * 1e-3, r'$\tau$ (ms)', color = hist_color)

# Estimated resting membrane potential
ax = plt.subplot(spec_histrow[:, 3])
plt.title(r'\textbf{E4} Equilibrium potential', loc = 'left')
plot_hist(
    params['El_est'][~np.isnan(params['El_est'])],
    r'$E_l$ (mV)', color = hist_color
)

plt.tight_layout()

### Save figure.
if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'fig3_somphysiol.png'), dpi = 300)
