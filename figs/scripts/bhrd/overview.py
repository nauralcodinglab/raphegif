#%% IMPORT MODULES

from __future__ import division

import os

import sys
sys.path.append('./src')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.patches import ConnectionPatch
import seaborn as sns
import pandas as pd
import numpy as np

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.Filter_Rect import Filter_Rect_LogSpaced
from grr.Filter_Rect import Filter_Rect_LinSpaced

import src.pltools as pltools


#%% IMPORT FILES

FNAMES_PATH = './data/raw/mPFC/fnames.csv'
DATA_PATH = './data/raw/mPFC/mPFC_spiking/'
dt = 0.1

make_plots = True

fnames = pd.read_csv(FNAMES_PATH)

experiments = []

for i in range(fnames.shape[0]):

    if fnames.loc[i, 'TTX'] == 0:

        tmp_experiment = Experiment(fnames.loc[i, 'Experimenter'] + fnames.loc[i, 'Cell_ID'], dt)
        tmp_experiment.setAECTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'AEC'],
            V_channel = 0, I_channel = 1)
        tmp_experiment.addTrainingSetTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'Train'],
            V_channel = 0, I_channel = 1)
        tmp_experiment.addTestSetTrace(FILETYPE = 'Axon', fname = DATA_PATH + fnames.loc[i, 'Test'],
            V_channel = 0, I_channel = 1)

        if make_plots:
            tmp_experiment.plotTrainingSet()

        experiments.append(tmp_experiment)

    else:
        continue


#%% QUALITY CONTROL

for expt in experiments:
    expt.plotTestSet()

# Exclude cell 0 due to drift in test set.
cells_to_exclude = [0]

experiments = [expt for i, expt in enumerate(experiments) if i not in cells_to_exclude]


#%% PERFORM AEC

for expt in experiments:

    tmp_AEC = AEC_Badel(expt.dt)

    # Define metaparametres
    tmp_AEC.K_opt.setMetaParameters(length=150.0, binsize_lb=expt.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
    tmp_AEC.p_expFitRange = [3.0,150.0]
    tmp_AEC.p_nbRep = 15

    # Assign tmp_AEC to myExp and compensate the voltage recordings
    expt.setAEC(tmp_AEC)
    expt.performAEC()


#%% FIT GIF

GIFs = []

for expt in experiments:

    tmp_GIF = GIF(0.1)

    # Define parameters
    tmp_GIF.Tref = 4.0

    tmp_GIF.eta = Filter_Rect_LogSpaced()
    tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


    tmp_GIF.gamma = Filter_Rect_LogSpaced()
    tmp_GIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

    # Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
    expt.trainingset_traces[0].setROI([[1000,59000]])

    # Perform the fit
    tmp_GIF.fit(expt, DT_beforeSpike=5.0)

    GIFs.append(tmp_GIF)

#%% MAKE SPIKETRAIN PREDICTIONS

make_plots = False

mPFC_Md_vals = []
mPFC_predictions = []

for expt, GIF_ in zip(experiments, GIFs):

    tmp_prediction = expt.predictSpikes(GIF_, nb_rep = 500)
    tmp_Md = tmp_prediction.computeMD_Kistler(4.0, 0.1)

    mPFC_Md_vals.append(tmp_Md)
    mPFC_predictions.append(tmp_prediction)

    if make_plots:
        tmp_prediction.plotRaster(delta = 1000.)


#%% MAKE FIGURE

ex_cell = 2

ex_experiment = experiments[ex_cell]
ex_GIF = GIFs[ex_cell]
ex_prediction = mPFC_predictions[ex_cell]

IMG_PATH = os.path.join('figs', 'ims', '2019BHRD')
plt.style.use('./figs/scripts/bhrd/poster_mplrc.dms')

spec_outer          = gs.GridSpec(
    4, 1, top = 0.95, left = 0.05, right = 0.95, bottom = 0.05,
    height_ratios = [0.8, 1, 1, 1.2]
)

spec_model          = gs.GridSpecFromSubplotSpec(1, 2, spec_outer[0, :])
spec_simulations    = gs.GridSpecFromSubplotSpec(3, 1, spec_model[:, 1], height_ratios = [0.2, 1, 0.6], hspace = 0)

spec_structure      = gs.GridSpecFromSubplotSpec(1, 2, spec_outer[1, :], width_ratios = [6, 1])
spec_training       = gs.GridSpecFromSubplotSpec(4, 1, spec_structure[:, 0], height_ratios = [0.2, 0.2, 1, 0.6], hspace = 0)
spec_test           = gs.GridSpecFromSubplotSpec(4, 1, spec_structure[:, 1], height_ratios = [0.2, 0.2, 1, 0.6], hspace = 0)

spec_fitting        = gs.GridSpecFromSubplotSpec(1, 3, spec_outer[2, :])
spec_data           = gs.GridSpecFromSubplotSpec(3, 1, spec_fitting[:, 0], height_ratios = [0.2, 0.2, 1], hspace = 0)
spec_data_split     = gs.GridSpecFromSubplotSpec(5, 1, spec_fitting[:, 1], height_ratios = [0.2, 0.1, 0.5, 0.5, 0.05], hspace = 0)

spec_performance    = gs.GridSpecFromSubplotSpec(1, 2, spec_outer[3, :], width_ratios = [1, 0.4], wspace = 0.4)
spec_performance_tr = gs.GridSpecFromSubplotSpec(4, 1, spec_performance[:, 0], height_ratios = [0.2, 1, 0.5, 0.5], hspace = 0)
spec_performance_quant = gs.GridSpecFromSubplotSpec(1, 2, spec_performance[:, 1], wspace = 0.6)

plt.figure(figsize = (16, 14))

plt.subplot(spec_model[:, 0])
plt.title('\\textbf{{A1}} RC-circuit subthreshold model', loc = 'left')
pltools.hide_ticks()

I_ax = plt.subplot(spec_simulations[0, :])
plt.title('\\textbf{{A2}} Sample GIF behaviour', loc = 'left')
I_vec = np.zeros(10000)
I_vec[2000:5000] = 0.6

toy_GIF = GIF(0.1)
toy_GIF.DV = 3
t, V, _, V_T, _ = toy_GIF.simulate(I_vec, toy_GIF.El)
V[V > -10] = 40

plt.plot(t, I_vec, '-', color = 'gray', lw = 1.)
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec_simulations[1, :], sharex = I_ax)

p_spk = 1 - np.exp(-np.exp((V - V_T)/toy_GIF.DV) * (0.1/1000.))
p_spk[V > -10] = np.nan

# Scale for presentation
p_spk /= np.nanmax(p_spk)
p_spk *= np.abs(np.max(V[V < -10]))
p_spk += toy_GIF.El

plt.plot(t, p_spk, 'r--', lw = 1., label = '$P_\mathrm{{spike}}$')
plt.plot(t, V, 'r-', lw = 1., label = '$V$')

plt.annotate(
    'Deterministic\n$V$ dynamics', (195, -49),
    xytext = (-15, 10), textcoords = 'offset points',
    ha = 'right', va = 'center', size = 'small',
    arrowprops = {'arrowstyle': '->'}
)

plt.legend(loc = 'upper right')
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec_simulations[2, :], sharex = I_ax)
for i in range(10):
    _, _, _, _, spks_tmp = toy_GIF.simulate(I_vec, toy_GIF.El)
    plt.plot(spks_tmp, [i for j in spks_tmp], 'r|', markersize = 5)

plt.annotate(
    'Stochastic\nspiking', (510, 4),
    xytext = (20, 0), textcoords = 'offset points',
    ha = 'left', va = 'center', size = 'small',
    arrowprops = {'arrowstyle': '->'}
)

pltools.hide_ticks()
pltools.hide_border()



### Panel B: Structure of Experiment

plt.subplot(spec_training[0, :])
plt.title('\\textbf{{B}} Structure of experiment', loc = 'left')
plt.text(0.5, 0.5, 'Training dataset', ha = 'center', va = 'center', transform = plt.gca().transAxes)
pltools.hide_border()
pltools.hide_ticks()

tr_I_ax = plt.subplot(spec_training[1, :])
plt.plot(
    1e-3 * ex_experiment.trainingset_traces[0].getTime(),
    ex_experiment.trainingset_traces[0].I,
    '-', color = 'gray', lw = 1.
)
pltools.add_scalebar(y_units = 'nA', anchor = (1.1, 0.1), y_label_space = 0.005, omit_x = True)
I_ylim = plt.ylim()

plt.subplot(spec_training[2, :], sharex = tr_I_ax)
plt.plot(
    1e-3 * ex_experiment.trainingset_traces[0].getTime(),
    ex_experiment.trainingset_traces[0].V,
    'k-', lw = 1.
)
pltools.add_scalebar(
    y_units = 'mV', anchor = (1.1, 0.2), y_label_space = 0.005,
    x_units = 's', x_size = 5, bar_space = 0, x_label_space = -0.05)
V_ylim = plt.ylim()

plt.subplot(spec_training[3, :], sharex = tr_I_ax)
spk_times = ex_experiment.trainingset_traces[0].getSpikeTimes()
plt.plot(
    1e-3 * spk_times, [0 for i in spk_times],
    '|k', markersize = 5
)
plt.xlim(0, 60)
pltools.hide_border()
pltools.hide_ticks()


plt.subplot(spec_test[0, :])
plt.text(0.5, 0.5, 'Test dataset', ha = 'center', va = 'center', transform = plt.gca().transAxes)
pltools.hide_border()
pltools.hide_ticks()

tst_I_ax = plt.subplot(spec_test[1, :])
plt.plot(
    ex_experiment.testset_traces[0].getTime(),
    ex_experiment.testset_traces[0].I,
    '-', color = 'gray', lw = 1.
)
pltools.hide_border()
pltools.hide_ticks()
plt.ylim(I_ylim)

plt.subplot(spec_test[2, :], sharex = tst_I_ax)
plt.plot(
    ex_experiment.testset_traces[0].getTime(),
    ex_experiment.testset_traces[0].V,
    'k-', lw = 1.
)
pltools.hide_border()
pltools.hide_ticks()
plt.ylim(V_ylim)

plt.subplot(spec_test[3, :], sharex = tst_I_ax)
for i, sw in enumerate(ex_experiment.testset_traces):
    spk_times = sw.getSpikeTimes()
    plt.plot(
        spk_times, [i for j in spk_times],
        '|k', markersize = 5
    )
pltools.hide_border()
pltools.hide_ticks()
plt.xlim(0, 10000)


### Panel C: Fitting procedure

proc_xlim = (15000, 18500)

plt.subplot(spec_data[0, :])
plt.title('\\textbf{{C}} Fitting procedure', loc = 'left')
plt.text(0.5, 0.5, 'Training data', ha = 'center', va = 'center', transform = plt.gca().transAxes)
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec_data[1, :])
plt.plot(
    ex_experiment.trainingset_traces[0].getTime(),
    ex_experiment.trainingset_traces[0].I,
    color = 'gray', lw = 1.
)
plt.xlim(proc_xlim)
pltools.hide_ticks()
pltools.hide_border()

split_left_anchor = plt.subplot(spec_data[2, :])
plt.plot(
    ex_experiment.trainingset_traces[0].getTime(),
    ex_experiment.trainingset_traces[0].V,
    'k-', lw = 1.
)
plt.xlim(proc_xlim)
pltools.hide_ticks()
pltools.hide_border()


plt.subplot(spec_data_split[0, :])
plt.text(0.5, 0.5, 'Subthreshold fluctuations', ha = 'center', va = 'center', transform = plt.gca().transAxes)
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec_data_split[1, :])
ROI_tmp = ex_experiment.trainingset_traces[0].getROI_FarFromSpikes(4, 4)
mask_tmp = np.zeros_like(ex_experiment.trainingset_traces[0].I, dtype = np.bool)
mask_tmp[ROI_tmp] = True
I_tmp = np.copy(ex_experiment.trainingset_traces[0].I)
I_tmp[~mask_tmp] = np.nan
plt.plot(
    ex_experiment.trainingset_traces[0].getTime(),
    I_tmp,
    '-', color = 'gray', lw = 1.
)
plt.xlim(proc_xlim)
pltools.hide_ticks()
pltools.hide_border()

split_tr_anchor = plt.subplot(spec_data_split[2, :])
V_tmp = np.copy(ex_experiment.trainingset_traces[0].V)
V_tmp[~mask_tmp] = np.nan
plt.plot(
    ex_experiment.trainingset_traces[0].getTime(),
    V_tmp,
    'k-', lw = 1.
)
plt.xlim(proc_xlim)
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec_data_split[3, :])
plt.text(0.5, 0.1, 'Set of spike times', ha = 'center', transform = plt.gca().transAxes)
plt.gca().set_facecolor('none')
pltools.hide_ticks()
pltools.hide_border()

split_br_anchor = plt.subplot(spec_data_split[4, :])
spk_times = ex_experiment.trainingset_traces[0].getSpikeTimes()
plt.plot(spk_times, [0 for i in spk_times], 'k|', markersize = 5)
plt.xlim(proc_xlim)
pltools.hide_ticks()
pltools.hide_border()

final_steps = plt.subplot(spec_fitting[:, 2])
plt.annotate(
    'Fit subthreshold dynamics\nvia multilinear regression',
    (0.5, 0.3), xytext = (0.5, 0.95),
    xycoords = 'axes fraction',
    textcoords = 'axes fraction',
    ha = 'center', va = 'top',
    arrowprops = {'arrowstyle': '->'}
)
plt.text(
    0.5, 0.05,
    'Fit spiking rule\nvia gradient ascent',
    ha = 'center', transform = plt.gca().transAxes
)
plt.gca().set_facecolor('none')
pltools.hide_border()
pltools.hide_ticks()

# Connection patches.

con_top_split = ConnectionPatch(
    xyA = (1, 0.5), xyB = (-0.02, 0.8),
    coordsA = 'axes fraction', coordsB = 'axes fraction',
    axesA = split_left_anchor, axesB = split_tr_anchor,
    arrowstyle = '->'
)
con_bottom_split = ConnectionPatch(
    xyA = (1, 0.5), xyB = (0.1, 3),
    coordsA = 'axes fraction', coordsB = 'axes fraction',
    axesA = split_left_anchor, axesB = split_br_anchor,
    arrowstyle = '->'
)
con_top_steps = ConnectionPatch(
    xyA = (1, 0.8), xyB = (0.02, 0.85),
    coordsA = 'axes fraction', coordsB = 'axes fraction',
    axesA = split_tr_anchor, axesB = final_steps,
    arrowstyle = '->'
)
con_bottom_steps = ConnectionPatch(
    xyA = (1, 3), xyB = (0.1, 0.15),
    coordsA = 'axes fraction', coordsB = 'axes fraction',
    axesA = split_br_anchor, axesB = final_steps,
    arrowstyle = '->'
)
split_left_anchor.add_artist(con_top_split)
split_left_anchor.add_artist(con_bottom_split)
split_tr_anchor.add_artist(con_top_steps)
split_br_anchor.add_artist(con_bottom_steps)


### Panel D: Quantification of performance

plt.subplot(spec_performance_tr[0, :])
perf_xlim = (5000, 8000)
plt.title('\\textbf{{D1}} Evaluation of performance', loc = 'left')
plt.plot(
    ex_experiment.testset_traces[0].getTime(),
    ex_experiment.testset_traces[0].I,
    '-', color = 'gray', lw = 1.
)
plt.xlim(perf_xlim)
pltools.add_scalebar(y_units = 'nA', y_size = 1, anchor = (1.02, 0.2), omit_x = True, y_label_space = -0.02)

plt.subplot(spec_performance_tr[1, :])
plt.plot(
    ex_experiment.testset_traces[0].getTime(),
    ex_experiment.testset_traces[0].V,
    'k-', lw = 1.
)

t, V, _, _, spks = GIFs[ex_cell].simulate(
    ex_experiment.testset_traces[0].I,
    ex_experiment.testset_traces[0].V[0]
)
V[np.array(spks / 0.1).astype(np.int32)] = 0

plt.plot(
    t, V,
    color = 'r', lw = 1., alpha = 0.7,
    label = 'Linear model'
)
plt.xlim(perf_xlim)
pltools.add_scalebar(y_units = 'mV', y_size = 50, anchor = (1.02, 0.2), omit_x = True, y_label_space = -0.02)

plt.subplot(spec_performance_tr[2, :])
for i, sw in enumerate(ex_experiment.testset_traces):
    spk_times = sw.getSpikeTimes()
    plt.plot(spk_times, [i for j in spk_times], 'k|', markersize = 5)
plt.xlim(perf_xlim)
pltools.hide_ticks()
pltools.hide_border()

plt.subplot(spec_performance_tr[3, :])
for i, spks in enumerate(ex_prediction.spks_model):
    plt.plot(spks, [i for j in spks], 'r|', markersize = 5)
    if i > len(ex_experiment.testset_traces):
        break
plt.xlim(np.array(perf_xlim))
pltools.add_scalebar(x_units = 'ms', anchor = (0.9, -0.15), omit_y = True, x_label_space = 0.08)

plt.subplot(spec_performance_quant[:, 0])
plt.title('\\textbf{{D2}}', loc = 'left')
plt.ylim(0, 1)
sigmas = [GIF_.var_explained_V for GIF_ in GIFs]
sns.swarmplot(y = sigmas, color = 'gray', edgecolor = 'k', size = 10, lw = 1.)
plt.xticks([])
plt.ylabel('$R^2$ on $V_\mathrm{{test}}$')
pltools.hide_border('trb')

plt.subplot(spec_performance_quant[:, 1])
plt.title('\\textbf{{D3}}', loc = 'left')
plt.ylim(0, 1)
sns.swarmplot(y = mPFC_Md_vals, color = 'gray', edgecolor = 'k', size = 10, lw = 1.)
plt.xticks([])
plt.ylabel('$M_d^*$ (4ms)')
pltools.hide_border('trb')

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'overview.png'))
