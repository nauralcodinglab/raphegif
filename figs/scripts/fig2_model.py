#%% IMPORT MODULES

from __future__ import division

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import pandas as pd
import scipy.stats as stats

import sys
sys.path.append('./src/')
sys.path.append('./figs/scripts/')
sys.path.append('analysis/subthresh_mod_selection')

import pltools
from ModMats import ModMats
from Experiment import Experiment
from SubthreshGIF_K import SubthreshGIF_K
from AEC_Badel import AEC_Badel


#%% LOAD DATA

PICKLE_PATH = './figs/figdata/'

with open(PICKLE_PATH + 'ohmic_mod.pyc', 'rb') as f:
    ohmic_mod_coeffs = pickle.load(f)


#%%
IMG_PATH = './figs/ims/'

mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]
mpl.rc('text', usetex = True)
mpl.rc('svg', fonttype = 'none')

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

linear_color = (.83, .43, .14)

plt.figure(figsize = (16, 12))

spec = gridspec.GridSpec(4, 4, height_ratios = [1, 1.5, 1.5, 0.5],
left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, hspace = 0.3, wspace = 0.4)

plt.subplot(spec[0, :2])
plt.title('\\textbf{{A1}} Neurons as RC circuits', loc = 'left')
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec[0, 2:])
plt.title('\\textbf{{A2}} Linear model of RC circuit dynamics', loc = 'left')
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec[1, :])
plt.title('\\textbf{{B}} Experimental approach', loc = 'left')
pltools.hide_ticks()
pltools.hide_border()

ax0 = plt.subplot(spec[2, :2])
plt.title('\\textbf{{C1}} Sample model performance on test data', loc = 'left')

plt.xlim(4000, 8000)
cell_no     = 5
real        = ohmic_mod_coeffs['real_testset_traces'][cell_no].mean(axis = 1)
sim         = ohmic_mod_coeffs['simulated_testset_traces'][cell_no].mean(axis = 1)
t = np.arange(0, len(real) /10, 0.1)
bins = np.arange(-122.5, -26, 5)[7:]

plt.axhline(-70, color = 'k', linewidth = 1, linestyle = '--', dashes = (10, 10),
zorder = 0)
plt.text(
5200, -69,
'$V_m = -70$mV',
horizontalalignment = 'center',
verticalalignment = 'bottom'
)

for i in range(1, len(bins)):

    pastel_factor = 0.3
    colour = np.array([0.1, i / len(bins), 0.1]) * (1 - pastel_factor) + pastel_factor

    plt.fill_between(
    t,
    real,
    sim,
    np.logical_and(real >= bins[i - 1], real < bins[i]),
    facecolor = colour, edgecolor = colour, zorder = 1
    )

plt.plot(
t, real,
'-', color = (0.1, 0.1, 0.1), linewidth = 2,
label = 'Real neuron'
)
plt.plot(
t, sim,
'-', color = linear_color, linewidth = 2,
label = 'Linear model'
)
plt.annotate(
'Model error binned \naccording to $V_{{real}}$',
(5700, -50),
xytext = (6000, -45),
arrowprops={'arrowstyle': '->'}
)
pltools.add_scalebar('ms', 'mV', (0.9, 0.35), text_spacing = (0.02, -0.02), bar_spacing = 0, round = True)
plt.legend()

ax1 = plt.subplot(spec[3, :2])
plt.xlim(4000, 8000)
plt.plot(
t, 1e3 * ohmic_mod_coeffs['real_testset_current'][cell_no].mean(axis = 1),
'-', color = (0.5, 0.5, 0.5), linewidth = 2
)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (0.77, 0.4))

bbox = ax0.get_position()
bbox.y0 = ax1.get_position().y1
ax0.set_position(bbox)

plt.subplot(spec[2:, 2:])
plt.title('\\textbf{{C2}} Test error is voltage-dependent', loc = 'left')
plt.plot(
np.delete(ohmic_mod_coeffs['binned_e2_centres'], cell_no, axis = 1),
np.delete(ohmic_mod_coeffs['binned_e2_values'], cell_no, axis = 1),
'-', linewidth = 0.7, color = linear_color, alpha = 0.5
)
plt.plot(
ohmic_mod_coeffs['binned_e2_centres'][:, cell_no],
ohmic_mod_coeffs['binned_e2_values'][:, cell_no],
'-', linewidth = 0.7, color = linear_color, alpha = 0.5,
label = 'Individual neuron'
)

for i in range(7, ohmic_mod_coeffs['binned_e2_centres'].shape[0]):

    pastel_factor = 0.3
    fill_colour = np.array([0.1, (i + 1) / ohmic_mod_coeffs['binned_e2_centres'].shape[0], 0.1]) * (1 - pastel_factor) + pastel_factor
    edge_colour = np.array([0.1, (i + 1) / ohmic_mod_coeffs['binned_e2_centres'].shape[0] * 0.7, 0.1]) * (1 - pastel_factor) + pastel_factor


    if i == 12:
        plt.plot(
        ohmic_mod_coeffs['binned_e2_centres'][i, cell_no],
        ohmic_mod_coeffs['binned_e2_values'][i, cell_no],
        'o', markeredgecolor = edge_colour, markerfacecolor = fill_colour, markersize = 10,
        label = 'Binned error from sample trace'
        )
    else:
        plt.plot(
        ohmic_mod_coeffs['binned_e2_centres'][i, cell_no],
        ohmic_mod_coeffs['binned_e2_values'][i, cell_no],
        'o', markeredgecolor = edge_colour, markerfacecolor = fill_colour, markersize = 10
        )

plt.plot(
np.nanmedian(ohmic_mod_coeffs['binned_e2_centres'], axis = 1),
np.nanmedian(ohmic_mod_coeffs['binned_e2_values'], axis = 1),
'-', linewidth = 2, color = linear_color,
label = 'Median'
)
plt.legend()
plt.ylim(0, plt.ylim()[1])
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
plt.xlabel('Binned $V_{{real}}$ (mV)')
pltools.hide_border('tr')

plt.savefig(IMG_PATH + 'fig2_model.png', dpi = 300)
plt.show()
