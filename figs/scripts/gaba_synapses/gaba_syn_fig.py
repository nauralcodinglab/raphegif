#%% IMPORT MODULES

from __future__ import division

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import scipy.optimize as optimize

import sys
sys.path.append('./analysis/gating')
sys.path.append('./analysis/gaba_synapses')
sys.path.append('./figs/scripts')
from cell_class import Cell
import MiniDetector
import pltools

#%% LOAD DATA

MINI_SAVE_PATH = './data/GABA_synapses/detected_minis/'

mini_detectors = []

for fname in os.listdir(MINI_SAVE_PATH):
    with open(MINI_SAVE_PATH + fname, 'rb') as f:
        mini_detectors.append(pickle.load(f))


SAMPLE_TR_PATH = './data/GABA_synapses/sample_traces/'

with open(SAMPLE_TR_PATH + 'averaged_traces.pyc', 'rb') as f:
    averaged_traces = pickle.load(f)

with open(SAMPLE_TR_PATH + 'sample_decay_fit.pyc', 'rb') as f:
    sample_tr = pickle.load(f)


#%% MAKE FIGURE

SAVE_PATH = './figs/ims/gaba_synapses/'

sample_minis_tr = 1

def subtract_baseline(arr, samples = 10):
    return np.copy(arr) - arr[:samples, :].mean(axis = 0)

bl_sub_sample_minis = subtract_baseline(mini_detectors[sample_minis_tr].minis)

spec_outer = gs.GridSpec(2, 3, wspace = 0.4, hspace = 0.4, top = 0.9, right = 0.9, left = 0.1)
spec_tau = gs.GridSpecFromSubplotSpec(1, 2, spec_outer[1, 2], width_ratios = [1, 0.4], wspace = 1)
spec_mini_tr = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[0, 0], hspace = 0)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')
plt.rc('font', size = 8)

plt.figure(figsize = (6, 5))

for i in range(3):
    plt.subplot(spec_mini_tr[i, :])
    if i == 0:
        plt.title('\\textbf{{A1}} Sample mIPSC traces', loc = 'left')
        plt.text(0.05, 0.9, 'TTX', ha = 'left', va = 'top', transform = plt.gca().transAxes)

    plt.plot(mini_detectors[i].t_vec, mini_detectors[i].I[:, 16], 'k-', lw = 0.5)
    plt.ylim(-12, 50)
    plt.xlim(500, 2500)

    if i == 2:
        pltools.add_scalebar(
            x_units = 'ms', y_units = 'pA', x_size = 200, y_size = 20,
            anchor = (0.7, 0.65), bar_space = 0, x_label_space = -0.05)
    else:
        plt.xticks([])
        plt.yticks([])
        pltools.hide_border()


plt.subplot(spec_outer[0, 1])
plt.title('\\textbf{{A2}} Sample mIPSCs', loc = 'left')
plt.text(
    0.95, 0.95, 'N = {}'.format(mini_detectors[sample_minis_tr].no_minis),
    ha = 'right', va = 'top', transform = plt.gca().transAxes
)
t_vec = np.arange(
    0,
    mini_detectors[sample_minis_tr].minis.shape[0] * mini_detectors[sample_minis_tr].dt,
    mini_detectors[sample_minis_tr].dt
)
plt.plot(
    np.tile(
        t_vec[:, np.newaxis],
        (1, mini_detectors[sample_minis_tr].no_minis)
    ),
    bl_sub_sample_minis,
    'k-', lw = 0.5, alpha = 0.05
)
plt.plot(t_vec, np.nanmean(bl_sub_sample_minis, axis = 1), 'g-')
pltools.add_scalebar(
    x_units = 'ms', y_units = 'pA', anchor = (0.9, 0.6), bar_space = 0,
    x_size = 10, x_label_space = -0.03
)

plt.subplot(spec_outer[0, 2])
plt.title('\\textbf{{A3}} mIPSC ampli. (+20mV)', loc = 'left')
for maxes in [np.max(subtract_baseline(MiDe.minis), axis = 0) for MiDe in mini_detectors]:
    sorted_maxes = np.sort(maxes)
    pctiles = np.linspace(0, 100, len(maxes))
    plt.plot(sorted_maxes, pctiles, 'k-')
plt.xlabel('Amplitude (pA)')
plt.ylabel('Percentile')
plt.xlim(-5, plt.xlim()[1])


plt.subplot(spec_outer[1, 0])
plt.title('\\textbf{{B1}} Sample eIPSCs', loc = 'left')
plt.plot(
    averaged_traces['t_mats'][sample_tr['ind']][:, :-2],
    averaged_traces['traces'][sample_tr['ind']][:, :-2],
    'k-', lw = 0.5
)
plt.ylim(-50, 175)
pltools.add_scalebar(
    x_units = 'ms', y_units = 'pA', x_size = 20, y_size = 50, bar_space = 0,
    x_on_left = False, anchor = (0.7, 0.6)
)

plt.subplot(spec_outer[1, 1])
plt.title('\\textbf{{B2}} I/V curves', loc = 'left')
plt.axhline(0, color = 'k', lw = 0.5)
plt.axvline(0, color = 'k', lw = 0.5)
for i in range(len(averaged_traces['t_mats'])):
    mask = 0 < np.gradient(averaged_traces['voltages'][i])
    plt.plot(
        averaged_traces['voltages'][i][mask],
        averaged_traces['traces'][i][250, mask],
        'k-'
    )
plt.xlabel('$V$ (mV)')
plt.ylabel('$I$ (pA)')

plt.subplot(spec_tau[:, 0])
plt.title('\\textbf{{B3}} $\\tau$ fit', loc = 'left')
plt.plot(sample_tr['x_tr'], sample_tr['y_tr'], 'k-', lw = 0.5)
plt.plot(sample_tr['x_fit'], sample_tr['y_fit'], 'g--')
plt.ylim(-10, 175)
pltools.add_scalebar(
    x_units = 'ms', y_units = 'pA', x_size = 20, y_size = 50, bar_space = 0,
    x_on_left = False, anchor = (0.7, 0.6)
)

plt.subplot(spec_tau[:, 1])
plt.title('\\textbf{{B4}}', loc = 'left')
plt.ylim(0, 35)
sns.swarmplot(y = averaged_traces['decay'], color = 'g', edgecolor = 'gray')
plt.xticks([])
plt.ylabel('$\\tau$ (ms)')
pltools.hide_border('trb')

if SAVE_PATH is not None:
    plt.savefig(SAVE_PATH + 'gaba_synapses_v1.png')

plt.show()
