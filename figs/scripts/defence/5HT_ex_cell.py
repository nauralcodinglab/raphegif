#%% IMPORT MODULES

from __future__ import division

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd

import sys
sys.path.append('./src')
sys.path.append('./figs/scripts')

from Experiment import *
from AEC_Badel import *
from GIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

import pltools

#%% DEFINE FUNCTION TO GAG VERBOSE POZZORINI FUNCTIONS

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

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

    expt.plotTestSet()


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

for expt in experiments:

    with gagProcess():

        tmp_GIF = GIF(0.1)

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


        tmp_GIF.gamma = Filter_Rect_LogSpaced()
        tmp_GIF.gamma.setMetaParameters(length=5000, binsize_lb=100, binsize_ub=1000.0, slope=2)

        # Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
        for i in range(len(expt.trainingset_traces)):
            expt.trainingset_traces[i].setROI([[2000,58000]])
        for i in range(len(expt.testset_traces)):
            expt.testset_traces[i].setROI([[1000, 14000]])

        tmp_GIF.fit(expt, DT_beforeSpike=5.0)

    GIFs.append(tmp_GIF)

    tmp_GIF.printParameters()


#%% EVALUATE PERFORMANCE

precision = 8.
Md_vals = []
predictions = []

for expt, GIF_ in zip(experiments, GIFs):

    with gagProcess():

        # Use the myGIF model to predict the spiking data of the test data set in myExp
        tmp_prediction = expt.predictSpikes(GIF_, nb_rep=500)

        # Compute Md* with a temporal precision of +/- 4ms
        Md = tmp_prediction.computeMD_Kistler(precision, 0.1)

    predictions.append(tmp_prediction)
    Md_vals.append(Md)

    print '{} MD* {}ms: {:.2f}'.format(expt.name, precision, Md)


#%% MAKE FIGURE

ex_cell = 4

IMG_PATH = './figs/ims/defence/'

spec = gs.GridSpec(
    2, 2, width_ratios = [1, 0.3],
    top = 0.9, bottom = 0.1, left = 0.05, right = 0.98,
    wspace = 0.9
)
spec_tr = gs.GridSpecFromSubplotSpec(4, 1, spec[:, 0], height_ratios = [0.2, 1, 0.3, 0.3], hspace = 0)
spec_quant = gs.GridSpecFromSubplotSpec(2, 1, spec[:, 1], hspace = 0.4)

plt.style.use('./figs/scripts/defence/defence_mplrc.dms')

plt.figure(figsize = (3, 2))

I_ax = plt.subplot(spec_tr[0, :])
#plt.title('\\textbf{{A}} 5HT neuron test data \& model predictions', loc = 'left')
plt.plot(
    1e-3 * experiments[ex_cell].testset_traces[0].getTime(),
    experiments[ex_cell].testset_traces[0].I,
    '-', color = 'gray', lw = 0.5
)
pltools.add_scalebar(y_units = 'nA', omit_x = True, anchor = (1.02, 0.1), y_label_space = -0.02, round = False)

plt.subplot(spec_tr[1, :], sharex = I_ax)
plt.plot(
    1e-3 * experiments[ex_cell].testset_traces[0].getTime(),
    experiments[ex_cell].testset_traces[0].V,
    'k-', lw = 0.5, label = 'Data'
)

t, V, _, _, spks = GIFs[ex_cell].simulate(
    experiments[ex_cell].testset_traces[0].I,
    experiments[ex_cell].testset_traces[0].V[0]
)
V[np.array(spks / 0.1).astype(np.int32)] = 45

plt.plot(
    1e-3 * t, V,
    color = 'r', linewidth = 0.5, alpha = 0.7,
    label = 'Linear model'
)
pltools.add_scalebar(y_units = 'mV', y_size = 50, omit_x = True, anchor = (1.02, 0.2), y_label_space = -0.02)

real_raster_ax = plt.subplot(spec_tr[2, :], sharex = I_ax)
for i, sw in enumerate(experiments[ex_cell].testset_traces):
    spks = sw.getSpikeTimes()
    plt.plot(1e-3 * spks, [i for j in spks], 'k|', markersize = 2)
pltools.hide_border()

plt.subplot(spec_tr[3, :], sharex = I_ax)
for i, spks in enumerate(predictions[ex_cell].spks_model):
    if i >= len(experiments[ex_cell].testset_traces):
        break
    plt.plot(1e-3 * spks, [i for j in spks], 'r|', markersize = 2)


plt.xlim(0, 15)
pltools.add_scalebar(x_units = 's', omit_y = True, anchor = (1, -0.05), x_label_space = -0.05)

pltools.hide_ticks(ax = real_raster_ax)

plt.subplot(spec_quant[0, :])
#plt.title('\\textbf{{B}}', loc = 'left')
plt.ylim(0, 1)
sigmas = [GIF.var_explained_V for GIF in GIFs]
sns.swarmplot(y = sigmas, color = 'gray', edgecolor = 'k', linewidth = 0.5)
pltools.hide_border('trb')
plt.xticks([])
plt.ylabel('$R^2$ on $V_\mathrm{{test}}$')

plt.subplot(spec_quant[1, :])
#plt.title('\\textbf{{C}}', loc = 'left')
plt.ylim(0, 1)
sns.swarmplot(y = Md_vals, color = 'gray', edgecolor = 'k', linewidth = 0.5)
pltools.hide_border('trb')
plt.xticks([])
plt.ylabel('$M_d^*$ (8ms)')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + '5HT_ex_cell.png')

plt.show()
