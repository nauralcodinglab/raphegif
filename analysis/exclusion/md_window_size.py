#%% IMPORT MODULES

from __future__ import division

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd
from scipy import stats

from grr.Experiment import Experiment
from grr.SpikeTrainComparator import intrinsic_reliability

from grr import pltools
from grr.Tools import gagProcess


#%% LOAD DATA

data_paths = {
    '5HT': './data/raw/5HT/fast_noise/',
    'GABA': './data/raw/GABA/OU_noise/',
    'mPFC': './data/raw/mPFC/mPFC_spiking/'
}

fname_paths = {
    '5HT': './data/raw/5HT/fast_noise/index.csv',
    'GABA': './data/raw/GABA/OU_noise/index.csv',
    'mPFC': './data/raw/mPFC/fnames.csv'
}

experiments = {
    '5HT': [],
    'GABA': [],
    'mPFC': []
}

# Read in GABA and 5HT cells
for key in ['5HT', 'GABA']:

    file_index = pd.read_csv(fname_paths[key])

    for i in range(file_index.shape[0]):
        try:
            with gagProcess():

                tmp_experiment = Experiment(file_index.loc[i, 'Cell'], 0.1)
                tmp_experiment.setAECTrace(
                    'Axon', fname = data_paths[key] + file_index.loc[i, 'AEC2'],
                    V_channel = 0, I_channel = 1
                )

                for ind in ['1', '2', '3']:

                    tmp_experiment.addTrainingSetTrace(
                        'Axon', fname = data_paths[key] + file_index.loc[i, 'Train' + ind],
                        V_channel = 0, I_channel = 1
                    )
                    tmp_experiment.addTestSetTrace(
                        'Axon', fname = data_paths[key] + file_index.loc[i, 'Test' + ind],
                        V_channel = 0, I_channel = 1
                    )

                for tr in tmp_experiment.testset_traces:
                    tr.detectSpikes()


            experiments[key].append(tmp_experiment)

        except RuntimeError:
            print 'Problem with {} import. Skipping.'.format(file_index.loc[i, 'Cell'])

# Exclude GABA0 because of problems.
experiments['GABA'].pop(0)


# Read in PFC files, which are formatted slightly differently.
fnames = pd.read_csv(fname_paths['mPFC'])

for i in range(fnames.shape[0]):

    if fnames.loc[i, 'TTX'] == 0:

        with gagProcess():

            tmp_experiment = Experiment(fnames.loc[i, 'Experimenter'] + fnames.loc[i, 'Cell'], 0.1)
            tmp_experiment.setAECTrace(FILETYPE = 'Axon', fname = data_paths['mPFC'] + fnames.loc[i, 'AEC'],
                V_channel = 0, I_channel = 1)
            tmp_experiment.addTrainingSetTrace(FILETYPE = 'Axon', fname = data_paths['mPFC'] + fnames.loc[i, 'Train'],
                V_channel = 0, I_channel = 1)
            tmp_experiment.addTestSetTrace(FILETYPE = 'Axon', fname = data_paths['mPFC'] + fnames.loc[i, 'Test'],
                V_channel = 0, I_channel = 1)

            for tr in tmp_experiment.testset_traces:
                tr.detectSpikes()

        experiments['mPFC'].append(tmp_experiment)

    else:
        continue

#%% EXTRACT RELIABILITIES
windows = np.logspace(0, 2, 30)

reliabilities = {
    '5HT' : [],
    'GABA': [],
    'mPFC': []
}
windows_arrs = {}

for key in experiments.keys():
    print('{}'.format(key))
    for i, win in enumerate(windows):
        print('Extracting reliability {:.1f}%'.format(100*(i + 1)/len(windows)))
        reliabilities_tmp = []
        for expt in experiments[key]:
            reliabilities_tmp.append(intrinsic_reliability(expt.testset_traces, win, 0.1))

        reliabilities[key].append(reliabilities_tmp)

    reliabilities[key] = np.array(reliabilities[key])
    windows_arrs[key] = np.tile(windows[:, np.newaxis], (1, reliabilities[key].shape[1]))

#%%

experiments['GABA'][0].testset_traces

#%% CREATE FIGURE

IMG_PATH = './figs/ims/exclusion/'

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

plt.figure(figsize = (6, 3))

plt.subplot(131)
plt.title(r'\textbf{A} mPFC L5 pyramidal', loc = 'left')
plt.axvline(8, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
plt.semilogx(windows_arrs['mPFC'], reliabilities['mPFC'], 'k-', lw = 0.8, alpha = 0.8)
plt.ylim(0,1)
plt.ylabel('Intrinsic reliability')
plt.xlabel('Window width (ms)')

plt.subplot(132)
plt.title(r'\textbf{B} DRN 5HT', loc = 'left')
plt.axvline(8, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
plt.semilogx(windows_arrs['5HT'], reliabilities['5HT'], '-', color = (0.9, 0.1, 0.1), lw = 0.8, alpha = 0.8)
plt.ylim(0,1)
plt.xlabel('Window width (ms)')

plt.subplot(133)
plt.title(r'\textbf{C} DRN SOM', loc = 'left')
plt.axvline(8, color = 'k', ls = '--', dashes = (10, 10), lw = 0.5)
plt.semilogx(windows_arrs['GABA'], reliabilities['GABA'], '-', color = (0.1, 0.7, 0.1), lw = 0.8, alpha = 0.8)
plt.ylim(0,1)
plt.xlabel('Window width (ms)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'intrinsic_reliability_updated.png')

plt.show()
