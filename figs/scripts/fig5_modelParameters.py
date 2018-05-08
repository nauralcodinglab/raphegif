#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats

import sys
sys.path.append('./figs/scripts/')

import pltools


#%% LOAD DATA

PICKLE_PATH = './figs/figdata/'

with open(PICKLE_PATH + 'gk2_mod.pyc', 'rb') as f:
    gk2_mod_coeffs = pickle.load(f)

params = pd.read_csv('data/DRN_membrane_parameters.csv')
params['El_est'] = -params['hold'] * params['R'] * 1e-3 - 70

params_5HT = params.loc[np.where(params['TdT'] == 1)]
params_5HT.drop('TdT', axis = 1, inplace = True)

#%% PAIR TEST PULSE PARAMETERS WITH MODEL-EXTRACTED PARAMETERS

cell_ids = ['DRN227', 'DRN229', 'DRN232', 'DRN235', 'DRN237', 'DRN239', 'DRN240',
            'DRN241', 'DRN243', 'DRN245', 'DRN247', 'DRN248', 'DRN281', 'DRN282']

testpulse_params = params_5HT.set_index(['Cell_ID']).loc[cell_ids, :]
testpulse_params

#%% MAKE FIGURE

IMG_PATH = './figs/ims/'

def corr_plot(x, y, outliers = None, ax = None):

    if outliers is None:
        outliers = []

    if ax is None:
        ax = plt.gca()

    x = np.array(x).copy()
    y = np.array(y).copy()

    outlier_mask = np.array([i not in outliers for i in range(len(x))])

    x_masked = x[outlier_mask]
    y_masked = y[outlier_mask]

    plt.plot(x_masked, y_masked, 'ko')
    plt.plot(x[~outlier_mask], y[~outlier_mask], 'o', color = 'gray')
    plt.plot(np.unique(x_masked), np.poly1d(np.polyfit(x_masked, y_masked, 1))(np.unique(x_masked)), 'k-')
    plt.text(0.6, 0.1, '$r = {:.2f}$\n$N = {}$ cells'.format(stats.pearsonr(x_masked, y_masked)[0], len(x_masked)),
    horizontalalignment = 'left', transform = ax.transAxes)


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


plt.figure(figsize = (16, 4.5))

spec = mpl.gridspec.GridSpec(2, 3, height_ratios = (0.15, 0.85))
outliers = [3, 12]

plt.subplot(spec[:, 0])
plt.title('\\textbf{{A}} Leak conductance', loc = 'left')
corr_plot(1e3/testpulse_params['R'], 1e3/np.array(gk2_mod_coeffs['R']), outliers)
plt.ylabel('Model $g_l$ (pS)')
plt.xlabel('Test-pulse $g_l$ (pS)')
pltools.hide_border('tr')

plt.subplot(spec[:, 1])
plt.title('\\textbf{{B}} Capacitance', loc = 'left')
corr_plot(testpulse_params['C'], 1e3 * np.array(gk2_mod_coeffs['C']), outliers)
plt.ylabel('Model $C$ (pF)')
plt.xlabel('Test-pulse $C$ (pF)')
pltools.hide_border('tr')

top_p = plt.subplot(spec[0, 2])
top_p.set_title('\\textbf{{C}} Equilibrium potential', loc = 'left')
top_p.plot(testpulse_params['El_est'][3], gk2_mod_coeffs['El'][3], 'o', color = 'gray')
bottom_p = plt.subplot(spec[1, 2])
corr_plot(testpulse_params['El_est'].drop(index = 'DRN235'), np.delete(np.array(gk2_mod_coeffs['El']), 3), [12])
xlim = (-86, -38)
top_p.set_xlim(xlim)
bottom_p.set_xlim(xlim)
top_p.set_xticks([])
pltools.hide_border('trb', top_p)
plt.ylabel('Model $E_l$ (mV)')
plt.xlabel('Test-pulse $\hat{{E}}_l$ (mV)')
pltools.hide_border('tr')


plt.subplots_adjust(left = 0.1, top = 0.85, right = 0.95, bottom = 0.15, wspace = 0.4)

plt.savefig(IMG_PATH + 'fig5_modelParameters.png', dpi = 300)

plt.show()
