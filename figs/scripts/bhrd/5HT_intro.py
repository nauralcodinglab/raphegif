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
from grr import pltools
from grr.cell_class import Cell
from gating_tools import *


#%% LOAD DATA

### Load/preprocess membrane parameters data.
params = pd.read_csv('data/raw/5HT/membrane_parameters/DRN_membrane_parameters.csv')
params['El_est'] = -params['hold'] * params['R'] * 1e-3 - 70
params_5HT = params.loc[np.where(params['TdT'] == 1)]
params_5HT.drop('TdT', axis = 1, inplace = True)

### Load/preprocess voltage steps.

GATING_PATH = os.path.join('data', 'raw', '5HT', 'gating')
gating = Cell().read_ABF([os.path.join(GATING_PATH, '18411002.abf'),
                          os.path.join(GATING_PATH, '18411010.abf'),
                          os.path.join(GATING_PATH, '18411017.abf'),
                          os.path.join(GATING_PATH, '18411019.abf'),
                          os.path.join(GATING_PATH, 'c0_inact_18201021.abf'),
                          os.path.join(GATING_PATH, 'c1_inact_18201029.abf'),
                          os.path.join(GATING_PATH, 'c2_inact_18201034.abf'),
                          os.path.join(GATING_PATH, 'c3_inact_18201039.abf'),
                          os.path.join(GATING_PATH, 'c4_inact_18213011.abf'),
                          os.path.join(GATING_PATH, 'c5_inact_18213017.abf'),
                          os.path.join(GATING_PATH, 'c6_inact_18213020.abf'),
                          os.path.join(GATING_PATH, '18619018.abf'),
                          os.path.join(GATING_PATH, '18614032.abf')])

# Subtract baseline/leak from gating recordings.
processed = []
for rec in gating:
    tmp = subtract_baseline(rec, slice(1000, 2000), 0)
    tmp = subtract_leak(tmp, slice(1000, 2000), slice(3000, 3400))
    processed.append(tmp)
del tmp, rec


### Load long current steps
LONG_CURR_PATH = os.path.join('data', 'raw','5HT', 'long_curr_steps')
long_curr = {
    'DRN436': Cell().read_ABF(os.path.join(LONG_CURR_PATH, '19204008.abf'))[0],
    'DRN439': Cell().read_ABF(os.path.join(LONG_CURR_PATH, '19204097.abf'))[0]
}
for key in long_curr.keys():
    long_curr[key].set_dt(0.1)


# Load/preprocess long current steps.


#%% CREATE FIGURE

plt.style.use(os.path.join('figs', 'scripts', 'bhrd', 'poster_mplrc.dms'))

IMG_PATH = os.path.join('figs', 'ims', '2019BHRD')
hist_color = 'gray'

plt.figure(figsize = (16,9.5))

# Define layout with GridSpec objects.
spec_outer = gs.GridSpec(3, 1)
spec_firstrow = gs.GridSpecFromSubplotSpec(
    1, 3, spec_outer[0, :]
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
plt.title(r'\textbf{B} Identification of 5HT neurons', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec_wccurr[0, :])
plt.title(r'\textbf{C} Whole cell currents', loc = 'left')
for tr in processed:
    tr.set_dt(0.1)
    plt.plot(
        tr.t_mat[0, :10000, -1], tr[0, 25000:35000, -1] * 1e-3,
        'k-', lw = 0.5, alpha = 0.8
    )
plt.text(
    0.95, 0.95,
    'N = {} cells'.format(len(processed)),
    ha = 'right', va = 'top',
    transform = plt.gca().transAxes
)
#plt.xlim(2400, 3500)
plt.ylim(-.1, plt.ylim()[1])
plt.xticks([])
plt.ylabel('$I$ (nA)')

plt.subplot(spec_wccurr[1, :])
plt.plot(
    tr.t_mat[1, :10000, -1], tr[1, 25000:35000, -1],
    '-', lw = 0.5, color = 'gray'
)
#plt.xlim(2400, 3500)
plt.yticks([-25, -75], ['$-25$', '$-75$'])
plt.ylabel('$V$ (mV)')

### Create second row with sample current steps.
plt.subplot(spec_currsteprow[0, 0])
plt.title(r'\textbf{D1}', loc = 'left')
plt.plot(
    long_curr['DRN436'].t_mat[0, :165000, 0] * 1e-3,
    long_curr['DRN436'][0, 25000:190000, 0],
    'k-', lw = 1.
)
plt.ylabel('$V$ (mV)')
plt.xticks([])
plt.yticks([50, 0, -50], ['$50$', '$0$', '$-50$'])

plt.subplot(spec_currsteprow[1, 0])
plt.plot(
    long_curr['DRN436'].t_mat[1, :165000, 0] * 1e-3,
    long_curr['DRN436'][1, 25000:190000, 0],
    '-', lw = 1., color = 'gray'
)
plt.ylabel('$I$ (pA)')
plt.xlabel('Time (s)')
plt.yticks([25, 75], ['$25$', '$75$'])

plt.subplot(spec_currsteprow[0, 1])
plt.title(r'\textbf{D2}', loc = 'left')
plt.plot(
    long_curr['DRN439'].t_mat[0, :165000, 0] * 1e-3,
    long_curr['DRN439'][0, 25000:190000, 0],
    'k-', lw = 1.
)
plt.xticks([])

plt.subplot(spec_currsteprow[1, 1])
plt.plot(
    long_curr['DRN439'].t_mat[1, :165000, 0] * 1e-3,
    long_curr['DRN439'][1, 25000:190000, 0],
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
    plt.ylim(0, plt.ylim()[1] * 1.3)
    shapiro_w, shapiro_p = stats.shapiro(data)
    plt.text(
        0.98, 0.98,
        'Normality test {}'.format(pltools.p_to_string(shapiro_p)),
        verticalalignment = 'top', horizontalalignment = 'right',
        transform = ax.transAxes
    )
    plt.text(
        0.5, 0.02,
        '$N = {}$ cells'.format(len(data)),
        verticalalignment = 'bottom', horizontalalignment = 'center',
        transform = ax.transAxes
    )

# Leak conductance
ax = plt.subplot(spec_histrow[:, 0])
plt.title(r'\textbf{E1} Leak conductance', loc = 'left')
plot_hist(1e3/params_5HT['R'], r'$g_l$ (pS)', color = hist_color)

# Capacitance
ax = plt.subplot(spec_histrow[:, 1])
plt.title(r'\textbf{E2} Capacitance', loc = 'left')
plot_hist(params_5HT['C'], '$C_m$ (pF)', color = hist_color)

# Membrane time constant
ax = plt.subplot(spec_histrow[:, 2])
plt.title(r'\textbf{E3} Time constant', loc = 'left')
plot_hist(params_5HT['R'] * params_5HT['C'] * 1e-3, r'$\tau$ (ms)', color = hist_color)

# Estimated resting membrane potential
ax = plt.subplot(spec_histrow[:, 3])
plt.title(r'\textbf{E4} Equilibrium potential', loc = 'left')
plot_hist(
    params_5HT['El_est'][~np.isnan(params_5HT['El_est'])],
    r'$E_l$ (mV)', color = hist_color
)

plt.tight_layout()

### Save figure.
if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'fig1_5HTphysiol.png'), dpi = 300)
