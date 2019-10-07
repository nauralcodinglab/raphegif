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

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.AugmentedGIF import AugmentedGIF
from src.Filter_Rect_LogSpaced import *
from grr.Filter_Exps import Filter_Exps

import src.pltools as pltools
from grr.Tools import gagProcess


#%% READ IN DATA

DATA_PATH = './data/fast_noise_5HT/'

file_index = pd.read_csv(DATA_PATH + 'index.csv')

experiments = []

for i in range(file_index.shape[0]):

    with gagProcess():

        tmp_experiment = Experiment(file_index.loc[i, 'Cell'], 0.1)
        tmp_experiment.setAECTrace(
            'Axon', fname = DATA_PATH + file_index.loc[i, 'AEC2'],
            V_channel = 0, I_channel = 1
        )

        for ind in ['1', '2', '3']:

            tmp_experiment.addTrainingSetTrace(
                'Axon', fname = DATA_PATH + file_index.loc[i, 'Train' + ind],
                V_channel = 0, I_channel = 1
            )
            tmp_experiment.addTestSetTrace(
                'Axon', fname = DATA_PATH + file_index.loc[i, 'Test' + ind],
                V_channel = 0, I_channel = 1
            )


    experiments.append(tmp_experiment)


#%% PLOT DATA

for expt in experiments:

    for tr in expt.testset_traces:
        tr.detectSpikes()

    #expt.plotTestSet()


#%% KEEP GOOD EXPERIMENTS

bad_cells = []

for i in np.flip([2, 3, 4, 7, 9, 10, 13], -1):
    bad_cells.append(experiments.pop(i))


#%% PERFORM AEC

AECs = []

for expt in experiments:

    with gagProcess():

        tmp_AEC = AEC_Badel(expt.dt)

        tmp_AEC.K_opt.setMetaParameters(length=150.0, binsize_lb=expt.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
        tmp_AEC.p_expFitRange = [3.0,150.0]
        tmp_AEC.p_nbRep = 15

        # Assign tmp_AEC to expt and compensate the voltage recordings
        expt.setAEC(tmp_AEC)
        expt.performAEC()

    AECs.append(tmp_AEC)


#%% FIT GIFs

GIFs = []
AugmentedGIFs = []

for i, expt in enumerate(experiments):

    print('Fitting GIFs {:.1f}%'.format(100 * (i + 1) / len(experiments)))

    for j, tmp_GIF in enumerate([GIF(0.1), AugmentedGIF(0.1)]):

        with gagProcess():

            # Define parameters
            tmp_GIF.Tref = 4.0

            tmp_GIF.eta = Filter_Rect_LogSpaced()
            tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


            tmp_GIF.gamma = Filter_Exps()
            tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

            # Define the ROI of the training set to be used for the fit
            for tr in expt.trainingset_traces:
                tr.setROI([[1000,59000]])
            for tr in expt.testset_traces:
                tr.setROI([[500, 14500]])

            tmp_GIF.fit(expt, DT_beforeSpike=5.0)

        if j == 0:
            GIFs.append(tmp_GIF)
        elif j == 1:
            AugmentedGIFs.append(tmp_GIF)

        tmp_GIF.printParameters()

#%%
with open(DATA_PATH + '5HT_aug_fast.pyc', 'wb') as f:
    obj = {'GIFs': GIFs, 'AugmentedGIFs': AugmentedGIFs, 'experiments': experiments}
    pickle.dump(obj, f)

#%%
with open(DATA_PATH + '5HT_aug_fast.pyc', 'rb') as f:
    obj = pickle.load(f)
    GIFs = obj['GIFs']
    AugmentedGIFs = obj['AugmentedGIFs']
    experiments = obj['experiments']

    del obj

#%%

ex_cell = 6

ISIs = []
for tr in experiments[ex_cell].trainingset_traces:
    ISIs.extend(np.diff(tr.getSpikeTimes()))

ISIs = np.array(ISIs)

GIFs[ex_cell].gamma.plot()

plt.figure()
plt.hist(ISIs)
plt.show()



#%%

test_filt = Filter_Rect_LogSpaced(5000., 150, 2000., slope = 2)
test_filt.setFilter_Function(lambda x: 100 * np.exp(-x / 1000.))
test_filt.plot()

#%% EVALUATE PERFORMANCE

precision = 8.
Md_vals_GIF = []
Md_vals_KGIF = []
predictions_GIF = []
predictions_KGIF = []

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
            tmp_label = 'GIF'
        elif i == 1:
            predictions_KGIF.append(tmp_prediction)
            Md_vals_KGIF.append(Md)
            tmp_label = 'KGIF'

        print '{} {} MD* {}ms: {:.2f}'.format(expt.name, tmp_label, precision, Md)


#%% MAKE FIGURE

np.mean([0.19, 0.40, 0.47, 0.19, 0.34, 0.29, 0.44])
np.mean([0.21, 0.75, 0.48, 0.42, 0.52, 0.57])

plt.rc('text', usetex = True)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

IMG_PATH = None#'./figs/ims/thesis/'

ex_cell = 3
xrange = (500, 14000)

predictions_GIF[ex_cell].spks_data
predictions_GIF[ex_cell].spks_model

spec_outer = plt.GridSpec(3, 1, height_ratios = [0.2, 1, 0.5])
spec_raster = gs.GridSpecFromSubplotSpec(3, 1, spec_outer[2, :])

plt.figure(figsize = (6, 6))

### Example neuron.
plt.subplot(spec_outer[0, :])
plt.title('\\textbf{{A}} Example trace from DRN 5HT neuron', loc = 'left')
plt.plot(
    experiments[ex_cell].testset_traces[0].getTime(),
    1e3 * experiments[ex_cell].testset_traces[0].I,
    color = 'gray',
    linewidth = 0.5
)
plt.xlim(xrange)
pltools.add_scalebar(y_units = 'pA', omit_x = True, anchor = (-0.05, 0.4))

plt.subplot(spec_outer[1, :])
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

plt.legend()

plt.subplot(spec_raster[0, :])
plt.title('\\textbf{{B}} Spike raster', loc = 'left')
for i, sweep_spks in enumerate(predictions_GIF[ex_cell].spks_data):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'k|', markersize = 3
    )

plt.xlim(xrange)
plt.axis('off')

plt.subplot(spec_raster[1, :])
for i, sweep_spks in enumerate(predictions_GIF[ex_cell].spks_model):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'r|', markersize = 3
    )

    if i > len(predictions_GIF[ex_cell].spks_data):
        break

plt.xlim(xrange)
plt.axis('off')

plt.subplot(spec_raster[2, :])
for i, sweep_spks in enumerate(predictions_KGIF[ex_cell].spks_model):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'b|', markersize = 3
    )

    if i > len(predictions_KGIF[ex_cell].spks_data):
        break

plt.xlim(xrange)
pltools.add_scalebar(
    anchor = (0.98, -0.12), x_units = 'ms', omit_y = True,
    x_label_space = -0.08
)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'sert_cell.png', dpi = 300)

plt.show()


#%% MD DISTRIBUTION FIGURE

plt.figure(figsize = (3, 3))

plt.subplot(121)
plt.plot(np.array(([0 for i in Md_vals_GIF], [1 for i in Md_vals_GIF])), np.array((Md_vals_GIF, Md_vals_KGIF)), color = 'gray', alpha = 0.5)
plt.plot([0 for i in Md_vals_GIF], Md_vals_GIF, 'ko', markersize = 10, alpha = 0.7)
plt.plot([1 for i in Md_vals_KGIF], Md_vals_KGIF, 'ko', markersize = 10, alpha = 0.7)
pltools.hide_border('trb')
plt.xticks([])
plt.ylim(0, 1)
plt.xlim(-1, 2)
plt.ylabel('Md* (8ms)')

plt.subplot(122)
plt.plot([0 for i in Md_vals_GIF],
    np.array(Md_vals_KGIF) - np.array(Md_vals_GIF),
    'ko', markersize = 10, alpha = 0.7
)
plt.xticks([])
pltools.hide_border('trb')
plt.ylim(0, 1)
plt.ylabel('$\Delta$Md* (8ms)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'sert_md_distribution.png', dpi = 300)

plt.show()


#%%

gl_ls = []
C_ls = []
El_ls = []
VT_ls = []
DV_ls = []
spks_ls = []

for expt, GIF_ in zip(experiments, GIFs):

    gl_ls.append(GIF_.gl)
    C_ls.append(GIF_.C)
    El_ls.append(GIF_.El)
    VT_ls.append(GIF_.Vt_star)
    DV_ls.append(GIF_.DV)

    spks_ls.append(expt.getTrainingSetNbOfSpikes())


plt.figure()

plt.suptitle('DRN 5HT neuron vanilla GIF fit and extracted parameters')

spec = gs.GridSpec(2, 3)

gl_ax   = plt.subplot(spec[0, 0])
plt.ylabel('gl')
plt.xlabel('Md*')
plt.xlim(0, 1)
C_ax    = plt.subplot(spec[0, 1])
plt.ylabel('C')
plt.xlabel('Md*')
plt.xlim(0, 1)
El_ax   = plt.subplot(spec[0, 2])
plt.ylabel('El (mV)')
plt.xlabel('Md*')
plt.xlim(0, 1)
VT_ax   = plt.subplot(spec[1, 0])
plt.ylabel('Threshold (mV)')
plt.xlabel('Md*')
plt.xlim(0, 1)
DV_ax   = plt.subplot(spec[1, 1])
plt.ylabel('$\Delta V$ (mV)')
plt.xlabel('Md*')
plt.xlim(0, 1)
spks_ax = plt.subplot(spec[1, 2])
plt.ylabel('No. spikes in training set')
plt.xlabel('Md*')
plt.xlim(0, 1)

for i in range(len(gl_ls)):

    gl_ax.plot(Md_vals[i], gl_ls[i], 'ko', alpha = 0.7)
    C_ax.plot(Md_vals[i], C_ls[i], 'ko', alpha = 0.7)
    El_ax.plot(Md_vals[i], El_ls[i], 'ko', alpha = 0.7)
    VT_ax.plot(Md_vals[i], VT_ls[i], 'ko', alpha = 0.7)
    DV_ax.plot(Md_vals[i], DV_ls[i], 'ko', alpha = 0.7)
    spks_ax.plot(Md_vals[i], spks_ls[i], 'ko', alpha = 0.7)

plt.tight_layout()
plt.subplots_adjust(top = 0.9)

plt.savefig(IMG_PATH + '5HT_fit_and_parameters.png', dpi = 300)

plt.show()
