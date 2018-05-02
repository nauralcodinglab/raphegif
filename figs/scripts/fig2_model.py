#%% IMPORT MODULES

from __future__ import division

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
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

plt.figure(figsize = (14.67, 10))

spec = gridspec.GridSpec(4, 4, height_ratios = [1, 1, 1.5, 0.5])

plt.subplot(spec[0, :2])
plt.title('A1 Neuron as RC circuit', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec[0, 2:])
plt.title('A2 Model definition', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec[1, :2])
plt.title('B1 Fitting model', loc = 'left')
pltools.hide_ticks()

plt.subplot(spec[1, 2:])
plt.title('B2 Using model to predict cellular responses on hold-out data', loc = 'left')
pltools.hide_ticks()

ax0 = plt.subplot(spec[2, :2])
plt.title('C1. Example quantification of model performance on new data', loc = 'left')

plt.xlim(4000, 8000)
cell_no     = 5
real        = ohmic_mod_coeffs['real_testset_traces'][cell_no].mean(axis = 1)
sim         = ohmic_mod_coeffs['simulated_testset_traces'][cell_no].mean(axis = 1)
t = np.arange(0, len(real) /10, 0.1)
bins = np.arange(-122.5, -26, 5)[7:]

plt.axhline(-70, color = 'k', linewidth = 1, linestyle = '--', dashes = (10, 10),
zorder = 0)
plt.text(
5200, -68,
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
'-', color = (0.9, 0.2, 0.2), linewidth = 2,
label = 'Linear model'
)
plt.annotate(
'Model error binned \naccording to $V_{{real}}$',
(5700, -50),
xytext = (6000, -45),
arrowprops={'arrowstyle': '->'}
)
pltools.add_scalebar('ms', 'mV', (0.95, 0.3))
plt.legend()

ax1 = plt.subplot(spec[3, :2])
plt.xlim(4000, 8000)
plt.plot(
t, ohmic_mod_coeffs['real_testset_current'][cell_no].mean(axis = 1),
'-', color = (0.5, 0.5, 0.5), linewidth = 2
)
pltools.add_scalebar()

bbox = ax0.get_position()
bbox.y0 = ax1.get_position().y1
ax0.set_position(bbox)

plt.subplot(spec[2:, 2:])
plt.title('C2 Quantification of model error (V-dependence)', loc = 'left')
plt.plot(
ohmic_mod_coeffs['binned_e2_centres'],
ohmic_mod_coeffs['binned_e2_values'],
'-', linewidth = 0.7, color = (0.2, 0.2, 0.2), alpha = 0.5
)
plt.plot(
np.nanmedian(ohmic_mod_coeffs['binned_e2_centres'], axis = 1),
np.nanmedian(ohmic_mod_coeffs['binned_e2_values'], axis = 1),
'-', linewidth = 2, color = (0.2, 0.2, 0.2)
)
plt.ylim(0, plt.ylim()[1])
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
plt.xlabel('Binned $V_{{real}}$ (mV)')

plt.savefig(IMG_PATH + 'fig2_model.png', dpi = 300)
plt.show()
