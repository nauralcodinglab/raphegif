"""
Makes a nice figure of an example L5 mPFC cell, including example simulated
trace and spk raster comparison.
"""

#%% IMPORT MODULES

from __future__ import division

import sys
sys.path.append('./src')
sys.path.append('./figs/scripts')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd
import numpy as np

from Experiment import *
from AEC_Badel import *
from GIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

import pltools


#%% IMPORT FILES

FNAMES_PATH = './data/mPFC/fnames.csv'
DATA_PATH = './data/mPFC/mPFC_spiking/'
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

make_plots = True

mPFC_Md_vals = []
mPFC_predictions = []

for expt, GIF in zip(experiments, GIFs):

    tmp_prediction = expt.predictSpikes(GIF, nb_rep = 500)
    tmp_Md = tmp_prediction.computeMD_Kistler(4.0, 0.1)

    mPFC_Md_vals.append(tmp_Md)
    mPFC_predictions.append(tmp_prediction)

    if make_plots:
        tmp_prediction.plotRaster(delta = 1000.)


#%% MAKE FIGURE

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

IMG_PATH = './figs/ims/TAC3/'

ex_cell = 2
xrange = (2000, 5000)

mPFC_predictions[ex_cell].spks_data
mPFC_predictions[ex_cell].spks_model

spec_outer = plt.GridSpec(3, 1, height_ratios = [0.2, 1, 0.5])
spec_raster = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[2, :])

plt.figure(figsize = (6, 6))

### Example neuron.
plt.subplot(spec_outer[0, :])
plt.title('\\textbf{{A}} Example trace from mPFC L5 pyramidal cell', loc = 'left')
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
V[np.array(spks / 0.1).astype(np.int32)] = 0

plt.plot(
    t, V,
    color = 'r', linewidth = 0.5, alpha = 0.7,
    label = 'Linear model'
)
plt.xlim(xrange)
pltools.add_scalebar(y_units = 'mV', omit_x = True, anchor = (-0.05, 0.15))

plt.legend()

plt.subplot(spec_raster[0, :])
plt.title('\\textbf{{B}} Spike raster', loc = 'left')
for i, sweep_spks in enumerate(mPFC_predictions[ex_cell].spks_data):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'k|', markersize = 3
    )

plt.xlim(xrange)
plt.axis('off')

plt.subplot(spec_raster[1, :])
for i, sweep_spks in enumerate(mPFC_predictions[ex_cell].spks_model):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'r|', markersize = 3
    )

    if i > len(mPFC_predictions[ex_cell].spks_data):
        break

plt.xlim(xrange)
pltools.add_scalebar(
    anchor = (0.98, -0.12), x_units = 'ms', omit_y = True,
    x_label_space = -0.08
)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'mPFC_ex_cell.png', dpi = 300)

plt.show()
