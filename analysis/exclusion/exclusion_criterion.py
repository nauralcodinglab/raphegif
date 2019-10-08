#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.AugmentedGIF import AugmentedGIF
from grr.Filter_Rect import Filter_Rect_LogSpaced
from grr.Filter_Exps import Filter_Exps
from grr.SpikeTrainComparator import intrinsic_reliability

from grr import pltools
from grr.Tools import gagProcess


#%% READ IN DATA

DATA_PATH = './data/raw/5HT/fast_noise/'

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

#%% FIND TRENDS IN NUMBER OF SPIKES

drifting_cells = []

for i, expt in enumerate(experiments):
    if len(expt.testset_traces) != 9:
        print('{:>16}Wrong no. of traces. Skipping...'.format(''))
        drifting_cells.append(i)
        continue

    spks = []
    for j in range(len(expt.testset_traces)):
        spks.append(expt.testset_traces[j].spks)

    no_spks_per_sweep = [len(s_) for s_ in spks]

    r, p = stats.pearsonr(no_spks_per_sweep, [0, 0, 0, 1, 1, 1, 2, 2, 2])
    if p > 0.1:
        stars = ''
    elif p > 0.05:
        stars = 'o'
    elif p > 0.01:
        stars = '*'
    elif p > 0.001:
        stars = '**'
    else:
        stars = '***'

    if np.abs(r) > 0.8:
        drifting_cells.append(i)

    print('{:>2}    {}    R = {:>6.3f}, p = {:>5.3f}   {}'.format(i, expt.name, r, p, stars))


#%% COMPUTE INTRINSIC RELIABILITY


unreliable_cells = []
reliability_ls = []
for i, expt in enumerate(experiments):

    reliability_tmp = intrinsic_reliability(expt.testset_traces, 8, 0.1)
    reliability_ls.append(reliability_tmp)

    if reliability_tmp < 0.2:
        unreliable_cells.append(i)
        stars = '*'
    else:
        stars = ''

    print('{:>2}    {} IR = {:.3f} {}'.format(
        i, expt.name, reliability_tmp, stars)
    )

plt.figure()
plt.hist(reliability_ls)
plt.show()

drifting_cells
unreliable_cells


#%% KEEP GOOD EXPERIMENTS

bad_cell_inds = []
[bad_cell_inds.extend(x) for x in [drifting_cells, unreliable_cells, [7]]] #Manually add 7, which has hf noise
bad_cell_inds = np.unique(bad_cell_inds)

bad_cells = []

for i in np.flip(np.sort(bad_cell_inds), -1):
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

    print('Fitting GIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

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


#%% DUMP DATA

output_dict = {
    'GIFs': GIFs,
    'AugmentedGIFs': AugmentedGIFs,
    'experiments': experiments
}

with open('./data/raw/5HT/fast_noise/5HT_good_aug_fast.pyc', 'wb') as f:
    pickle.dump(output_dict, f)

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

IMG_PATH = './figs/ims/exclusion/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

plt.figure(figsize = (3, 3))

md_df = pd.DataFrame({'GIF': Md_vals_GIF, 'KGIF': Md_vals_KGIF}).melt(var_name = 'Model', value_name = 'Md*')

plt.ylim(0, 1)
plt.xlim(-0.2, 1.2)
sns.swarmplot('Model', 'Md*', 'Model', md_df, palette = ('b', 'r'), linewidth = 1)
plt.plot(
    np.array([[0.2 for i in Md_vals_GIF], [0.8 for i in Md_vals_KGIF]]),
    np.array([Md_vals_GIF, Md_vals_KGIF]),
    '-', color = 'gray', lw = 1.5
)
plt.text(0.5, 0.95, '$p = {:.3f}$'.format(stats.wilcoxon(Md_vals_GIF, Md_vals_KGIF)[1]), ha = 'center', va = 'top')
plt.ylabel('$M_d^*$')
plt.gca().get_legend().remove()
pltools.hide_border('tr')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'Md_all_good_cells_systematic.png')

plt.show()
