#TODO: Make raster spks line up with sample traces.

#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
from scipy import stats

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.AugmentedGIF import AugmentedGIF
from src.Filter_Rect_LogSpaced import *
from src.Filter_Rect_LinSpaced import *

import src.pltools as pltools
from grr.Tools import gagProcess


#%% LOAD DATA

DATA_PATH = './data/raw/5HT/fast_noise/'

with open(DATA_PATH + '5HT_good_aug_fast.pyc', 'rb') as f:
    obj = pickle.load(f)
    GIFs = obj['GIFs']
    AugmentedGIFs = obj['AugmentedGIFs']
    experiments = obj['experiments']

    del obj



#%% EVALUATE PERFORMANCE

precision = 8.
Md_vals_GIF = []
Md_vals_KGIF = []
predictions_GIF = []
predictions_KGIF = []
R2_GIF = []
R2_KGIF = []

for i, GIF_ls in enumerate([GIFs, AugmentedGIFs]):

    for expt, GIF_ in zip(experiments, GIF_ls):

        if not np.isnan(GIF_.Vt_star):

            with gagProcess():

                # Use the myGIF model to predict the spiking data of the test data set in myExp
                tmp_prediction = expt.predictSpikes(GIF_, nb_rep=500)

                # Compute Md* with a temporal precision of +/- 4ms
                Md = tmp_prediction.computeMD_Kistler(precision, 0.1)

        else:

            tmp_prediction = np.nan
            Md = np.nan

        if i == 0:
            predictions_GIF.append(tmp_prediction)
            Md_vals_GIF.append(Md)
            R2_GIF.append(GIF_.var_explained_V)
            tmp_label = 'GIF'
        elif i == 1:
            predictions_KGIF.append(tmp_prediction)
            Md_vals_KGIF.append(Md)
            R2_KGIF.append(GIF_.var_explained_V)
            tmp_label = 'KGIF'

        print '{} {} MD* {}ms: {:.2f}'.format(expt.name, tmp_label, precision, Md)

print('\nSummary statistics:')
print('GIF Md* {:.3f} +/- {:.3f}'.format(np.mean(Md_vals_GIF), np.std(Md_vals_GIF)))
print('KGIF Md* {:.3f} +/- {:.3f}'.format(np.mean(Md_vals_KGIF), np.std(Md_vals_KGIF)))


#%%

expt = experiments[6]

arr = np.empty((len(expt.testset_traces[0].getSpikeTrain()), len(expt.testset_traces)))
for i in range(len(expt.testset_traces)):
    arr[:, i] = expt.testset_traces[i].getSpikeTrain()

PSTH = np.convolve(arr.sum(axis = 1), np.ones(80), 'same')

plt.figure()
plt.plot(PSTH)
plt.show()

#%% MAKE FIGURE

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/5HTGIF_comparison/'

ex_cell = 3
xrange = (500, 7500)

spec_outer = plt.GridSpec(
    2, 2,
    width_ratios = [1, 0.2], wspace = 0.4, hspace = 0.4,
    top = 0.9, right = 0.95, left = 0.15, bottom = 0.1
)
spec_tr = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[:, 0], height_ratios = [0.2, 1, 0.4], hspace = 0)
spec_raster = gs.GridSpecFromSubplotSpec(3, 1, spec_tr[2, 0], hspace = 0)

plt.figure(figsize = (6, 4))

### Example neuron.
plt.subplot(spec_tr[0, :])
plt.title('\\textbf{{A}} Model comparison on 5HT neuron test data', loc = 'left')
plt.plot(
    experiments[ex_cell].testset_traces[0].getTime(),
    1e3 * experiments[ex_cell].testset_traces[0].I,
    color = 'gray',
    linewidth = 0.5
)
plt.xlim(xrange)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (-0.05, 0.4))

plt.subplot(spec_tr[1, :])
plt.plot(
    experiments[ex_cell].testset_traces[0].getTime(),
    experiments[ex_cell].testset_traces[0].V,
    color = 'k', linewidth = 0.5,
    label = 'Real neuron'
)

t, V, _, _, spks = GIFs[ex_cell].simulate(
    experiments[ex_cell].testset_traces[0].I,
    experiments[ex_cell].testset_traces[0].V[0]
)
V[np.array(spks / 0.1).astype(np.int32)] = 40
t, Vk, _, _, spksk = AugmentedGIFs[ex_cell].simulate(
    experiments[ex_cell].testset_traces[0].I,
    experiments[ex_cell].testset_traces[0].V[0]
)
Vk[np.array(spksk / 0.1).astype(np.int32)] = 40

plt.plot(
    t, V,
    color = 'r', linewidth = 0.5, alpha = 0.7,
    label = 'Linear model'
)
plt.plot(
    t, Vk,
    color = 'blue', linewidth = 0.5, alpha = 0.7,
    label = 'KGIF'
)
plt.xlim(xrange)
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (-0.05, 0.15))

plt.subplot(spec_raster[0, :])
for i, sweep_spks in enumerate(predictions_GIF[ex_cell].spks_data):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'k|', markersize = 2
    )

plt.xlim(xrange)
plt.axis('off')

plt.subplot(spec_raster[1, :])
for i, sweep_spks in enumerate(predictions_GIF[ex_cell].spks_model):

    if i >= len(predictions_GIF[ex_cell].spks_data):
        break

    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'r|', markersize = 2
    )

plt.xlim(xrange)
plt.axis('off')

plt.subplot(spec_raster[2, :])
for i, sweep_spks in enumerate(predictions_KGIF[ex_cell].spks_model):

    if i >= len(predictions_KGIF[ex_cell].spks_data):
        break

    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'b|', markersize = 2
    )


plt.xlim(xrange)
pltools.add_scalebar(
    anchor = (0.98, -0.12), x_units = 'ms', omit_y = True,
    x_label_space = -0.16
)

plt.subplot(spec_outer[0, 1])
plt.title('\\textbf{{B}}', loc = 'left')
plt.plot(
    np.array(([0 for i in R2_GIF], [1 for i in R2_GIF])),
    np.array((R2_GIF, R2_KGIF)),
    color = 'gray', alpha = 0.5
)
plt.plot([0 for i in R2_GIF], R2_GIF, 'ro', markeredgecolor = 'k', clip_on = False)
plt.plot([1 for i in R2_KGIF], R2_KGIF, 'bo', markeredgecolor = 'k', clip_on = False)
plt.text(
    0.5, 0.2, pltools.p_to_string(stats.wilcoxon(R2_GIF, R2_KGIF)[1]),
    ha = 'center', va = 'center', transform = plt.gca().transAxes
)
pltools.hide_border('tr')
#plt.yticks([0, 0.5, 1], ['0.0', '0.5', '1.0'])
plt.xticks([0, 1], ['GIF', 'KGIF'])
plt.ylim(0, 1)
plt.xlim(-0.25, 1.25)
plt.ylabel('$R^2$ on $V_\\mathrm{{test}}$')

plt.subplot(spec_outer[1, 1])
plt.title('\\textbf{{C}}', loc = 'left')
plt.plot(
    np.array(([0 for i in Md_vals_GIF], [1 for i in Md_vals_GIF])),
    np.array((Md_vals_GIF, Md_vals_KGIF)),
    color = 'gray', alpha = 0.5
)
plt.plot([0 for i in Md_vals_GIF], Md_vals_GIF, 'ro', markeredgecolor = 'k')
plt.plot([1 for i in Md_vals_KGIF], Md_vals_KGIF, 'bo', markeredgecolor = 'k')
plt.text(
    0.5, 0.9, pltools.p_to_string(stats.wilcoxon(Md_vals_GIF, Md_vals_KGIF)[1]),
    ha = 'center', va = 'center', transform = plt.gca().transAxes
)
pltools.hide_border('tr')
#plt.yticks([0, 0.5, 1], ['0.0', '0.5', '1.0'])
plt.xticks([0, 1], ['GIF', 'KGIF'])
plt.ylim(0, 1)
plt.xlim(-0.25, 1.25)
plt.ylabel('$M_d^*$ (8ms)')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'augmented_GIF_comparison.png', dpi = 300)

plt.show()

#%% STATS AND STUFF


no_spks = [experiments[i].getTrainingSetNbOfSpikes() for i in range(len(experiments))]
no_spks

vals = no_spks
print('{:.3f} +/- {:.3f}'.format(np.mean(vals), np.std(vals)))
stats.wilcoxon(R2_KGIF, R2_GIF)

#%% SUMMARY FIGURE OF AUGMENTED GIF PARAMETERS

aug_pdata_dict = {
'gl':[],
'C': [],
'gbar_K1': [],
'gbar_K2': [],
'DV': [],
'Vt_star': []
}

base_pdata_dict = {
'gl':[],
'C': [],
'DV': [],
'Vt_star': []
}

for key in aug_pdata_dict.keys():
    for GIF_ in AugmentedGIFs:
        aug_pdata_dict[key].append(GIF_.__getattribute__(key))
    aug_pdata_dict[key] = np.array(aug_pdata_dict[key])

for key in base_pdata_dict.keys():
    for GIF_ in GIFs:
        base_pdata_dict[key].append(GIF_.__getattribute__(key))
    base_pdata_dict[key] = np.array(base_pdata_dict[key])


plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/5HTGIF_comparison/'

spec = gs.GridSpec(3, 3, wspace = 1, hspace = 0.3)
spec_filters = gs.GridSpecFromSubplotSpec(2, 1, spec[:, 2], hspace = 0.5)

plt.figure(figsize = (6, 4))

plt.subplot(spec[0, 0])
plt.title('\\textbf{{A1}}', loc = 'left')
#plt.ylim(0, 2)
plt.ylabel('$g_l$ (nS)')
sns.swarmplot(
    y = 1e3 * np.concatenate((aug_pdata_dict['gl'], base_pdata_dict['gl'])),
    x = np.concatenate(([1 for i in aug_pdata_dict['gl']], [0 for i in base_pdata_dict['gl']])),
    palette = ['red', 'blue'], edgecolor = 'gray', linewidth = 0.5
)
plt.xticks([])
pltools.hide_border('trb')

plt.subplot(spec[0, 1])
plt.title('\\textbf{{A2}}', loc = 'left')
plt.ylabel('$C$ (pF)')
sns.swarmplot(
    y = 1e3 * np.concatenate((aug_pdata_dict['C'], base_pdata_dict['C'])),
    x = np.concatenate(([1 for i in aug_pdata_dict['C']], [0 for i in base_pdata_dict['C']])),
    palette = ['red', 'blue'], edgecolor = 'gray', linewidth = 0.5
)
plt.xticks([])
pltools.hide_border('trb')

plt.subplot(spec[1, 0])
plt.title('\\textbf{{A3}}', loc = 'left')
#plt.ylim(-70, -30)
plt.ylabel('$V_T^*$ (mV)')
sns.swarmplot(
    y = np.concatenate((aug_pdata_dict['Vt_star'], base_pdata_dict['Vt_star'])),
    x = np.concatenate(([1 for i in aug_pdata_dict['Vt_star']], [0 for i in base_pdata_dict['Vt_star']])),
    palette = ['red', 'blue'], edgecolor = 'gray', linewidth = 0.5
)
plt.xticks([])
pltools.hide_border('trb')

plt.subplot(spec[1, 1])
plt.title('\\textbf{{A4}}', loc = 'left')
plt.ylabel('$\Delta V$ (mV)')
plt.ylim(0, 9)
sns.swarmplot(
    y = np.concatenate((aug_pdata_dict['DV'], base_pdata_dict['DV'])),
    x = np.concatenate(([1 for i in aug_pdata_dict['DV']], [0 for i in base_pdata_dict['DV']])),
    palette = ['red', 'blue'], edgecolor = 'gray', linewidth = 0.5
)
plt.xticks([])
pltools.hide_border('trb')

plt.subplot(spec[2, 0])
plt.title('\\textbf{{A5}}', loc = 'left')
plt.ylabel('$\\bar{g}_A$ (nS)')
plt.axhline(0, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
sns.swarmplot(y = 1e3 * aug_pdata_dict['gbar_K1'], color = 'blue', edgecolor = 'gray', lw = 0.5)
plt.xticks([])
pltools.hide_border('trb')

plt.subplot(spec[2, 1])
plt.title('\\textbf{{A6}}', loc = 'left')
plt.ylabel('$\\bar{g}_\mathrm{Kslow}$ (nS)')
sns.swarmplot(y = 1e3 * aug_pdata_dict['gbar_K2'], color = 'blue', edgecolor = 'gray', lw = 0.5)
plt.xticks([])
pltools.hide_border('trb')


plt.subplot(spec_filters[0, :])
plt.title('\\textbf{{B1}} $\eta$', loc = 'left')
for GIF_ in GIFs:
    plt.loglog(GIF_.eta.filtersupport, GIF_.eta.filter, 'r-', linewidth = 0.7, alpha = 0.7)
for GIF_ in AugmentedGIFs:
    plt.loglog(GIF_.eta.filtersupport, GIF_.eta.filter, 'b-', linewidth = 0.7, alpha = 0.5)
plt.ylabel('Spike-triggered current (nA)')
plt.xlabel('Time (ms)')
pltools.hide_border('tr')

plt.subplot(spec_filters[1, :])
plt.title('\\textbf{{B2}} $\gamma$', loc = 'left')
for GIF_ in GIFs:
    supp, filt = GIF_.gamma.getInterpolatedFilter(0.1)
    plt.loglog(supp, filt, 'r-', linewidth = 0.7, alpha = 0.7)
for GIF_ in AugmentedGIFs:
    supp, filt = GIF_.gamma.getInterpolatedFilter(0.1)
    plt.loglog(supp, filt, 'b-', linewidth = 0.7, alpha = 0.5)
plt.ylabel('Threshold movement (mV)')
plt.xlabel('Time (ms)')
pltools.hide_border('tr')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'augGIF_good_params.png')

plt.show()

IMG_PATH
