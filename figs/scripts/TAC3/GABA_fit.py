#%% IMPORT MODULES

from __future__ import division

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd

from src.Experiment import *
from src.AEC_Badel import *
from src.GIF import *
from src.Filter_Rect_LogSpaced import *
from src.Filter_Rect_LinSpaced import *

import src.pltools as pltools
from src.Tools import gagProcess


#%% READ IN DATA

DATA_PATH = './data/GABA_cells/'

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

    expt.plotTestSet()

# Exclude first cell based on crummy test set.
# (Too few spikes.)
experiments.pop(0)


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
        tmp_GIF.gamma.setMetaParameters(length=500.0, binsize_lb=5.0, binsize_ub=1000.0, slope=5.0)

        # Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
        expt.trainingset_traces[0].setROI([[2000,58000]])

        tmp_GIF.fit(expt, DT_beforeSpike=5.0)

    GIFs.append(tmp_GIF)

    tmp_GIF.printParameters()


#%% EVALUATE PERFORMANCE

precision = 4.
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

ex_cell = 0
xrange = (2000, 5000)

predictions[ex_cell].spks_data
predictions[ex_cell].spks_model

spec_outer = plt.GridSpec(3, 1, height_ratios = [0.2, 1, 0.5])
spec_raster = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[2, :])

plt.figure(figsize = (6, 6))

### Example neuron.
plt.subplot(spec_outer[0, :])
plt.title('\\textbf{{A}} Example trace from positively identified DRN SOM neuron', loc = 'left')
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
for i, sweep_spks in enumerate(predictions[ex_cell].spks_data):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'k|', markersize = 3
    )

plt.xlim(xrange)
plt.axis('off')

plt.subplot(spec_raster[1, :])
for i, sweep_spks in enumerate(predictions[ex_cell].spks_model):
    plt.plot(
        sweep_spks,
        [i for i_ in range(len(sweep_spks))],
        'r|', markersize = 3
    )

    if i > len(predictions[ex_cell].spks_data):
        break

plt.xlim(xrange)
pltools.add_scalebar(
    anchor = (0.98, -0.12), x_units = 'ms', omit_y = True,
    x_label_space = -0.08
)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'GABA_bad_cell.png', dpi = 300)

plt.show()


#%% MD DISTRIBUTION FIGURE

plt.figure(figsize = (1, 3))

plt.plot([0 for i in Md_vals], Md_vals, 'ko', markersize = 10, alpha = 0.7)
pltools.hide_border('trb')
plt.xticks([])
plt.ylim(0, 1)
plt.ylabel('Md* (4ms)')

plt.tight_layout()

plt.savefig(IMG_PATH + 'gaba_md_distribution.png', dpi = 300)

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

plt.suptitle('DRN SOM neuron vanilla GIF fit and extracted parameters')

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

plt.savefig(IMG_PATH + 'GABA_fit_and_parameters.png', dpi = 300)

plt.show()


#%%

plt.figure()

plt.subplot(111)
plt.title('$\gamma$')
plt.axhline(0, ls = '--', color = 'k', lw = 0.5, dashes = (10, 10))

for i in range(len(GIFs)):
    x, y = GIFs[i].gamma.getInterpolatedFilter(0.1)
    plt.plot(x, y, 'k-')

plt.ylim(plt.ylim()[0], 30)
plt.ylabel('Threshold movement (mV)')
plt.xlabel('Time (ms)')

plt.savefig(IMG_PATH + 'GABA_gamma.png', dpi = 300)

plt.show()


#%%

plt.figure()

plt.subplot(111)
plt.title('$\gamma$-filter first bin amplitude vs. fit')
plt.ylabel('$\gamma$ first bin')
plt.xlabel('Md* (4ms)')
plt.xlim(0, 1)

for i in range(len(GIFs)):
    plt.plot(Md_vals[i], GIFs[i].gamma.getCoefficients()[0], 'ko', markersize = 10, alpha = 0.7)

plt.savefig(IMG_PATH + 'GABA_gamma_vs_fit.png', dpi = 300)

plt.show()

#%%

def compute_ISIs(trace):

    spk_times = trace.getSpikeTimes()

    ISIs = np.diff(spk_times)

    return ISIs

plt.figure()

no_bins = 35
plt.hist(compute_ISIs(experiments[-1].trainingset_traces[0]), color = 'k', label = 'Reference cell 1', bins = no_bins)
plt.hist(compute_ISIs(experiments[-2].trainingset_traces[0]), color = 'gray', alpha = 0.5, label = 'Reference cell 2', bins = no_bins)
plt.hist(compute_ISIs(experiments[0].trainingset_traces[0]), color = 'r', alpha = 0.5, label = 'Worst cell', bins = no_bins)
plt.legend()

pltools.hide_border('trl')
plt.yticks([])
plt.xlabel('ISI (ms)')

plt.savefig(IMG_PATH + 'GABA_ISI_distribution.png', dpi = 300)

plt.show()

#%%

violin_df = pd.DataFrame(columns = ('Label', 'ISI'))

for cell_no in range(len(experiments)):

    ISIs_tmp = []
    for tr_no in range(3):
        ISIs_tmp.extend(compute_ISIs(experiments[cell_no].trainingset_traces[tr_no]))


    name_vec_tmp = ['{}\nMd = {:.2f}'.format(experiments[cell_no].name, Md_vals[cell_no]) for i in range(len(ISIs_tmp))]

    if cell_no == 0:
        violin_df = pd.DataFrame({'Label': name_vec_tmp, 'ISI': ISIs_tmp})
    else:
        violin_df = violin_df.append(pd.DataFrame({'Label': name_vec_tmp, 'ISI': ISIs_tmp}))

violin_df

plt.figure(figsize = (8, 8))

violin_ax = plt.subplot(111)
plt.title('ISI distribution of DRN SOM neurons')

sns.violinplot(x = 'ISI', y = 'Label', data = violin_df, cut = 0, bw = 0.1)
plt.ylabel('')
plt.xlabel('ISI (ms)')
plt.xlim(0, plt.xlim()[1])

ins_ax = inset_axes(violin_ax, width = '20%', height = '20%', loc = 'center right', borderpad = 3)
plt.title('First decile')

for i, lab in enumerate(['DRN351', 'DRN354', 'DRN355', 'DRN356', 'DRN357']):
    tmp_quantile = np.percentile(violin_df.loc[[lab in x for x in violin_df['Label']], 'ISI'], 10)
    ins_ax.barh(len(violin_df['Label']) - i - 1, tmp_quantile)

#ins_ax.set_yticklabels(np.flip(['DRN351', 'DRN354', 'DRN355', 'DRN356', 'DRN357'], -1))
ins_ax.set_yticks([])
ins_ax.set_xlabel('ISI (ms)')


plt.savefig(IMG_PATH + 'ISI_violin.png', dpi = 300)

plt.show()
