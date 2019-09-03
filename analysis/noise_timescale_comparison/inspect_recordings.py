#%% IMPORT MODULES

from __future__ import division

import os

import pandas as pd
import numpy as np

from src.SpikeTrainComparator import intrinsic_reliability
from src.Experiment import *
from src.Tools import gagProcess

#%% LOAD DATA FILES
file_index = pd.read_csv(os.path.join('.', 'data', 'raw', '5HT', 'noise_comparison', 'index.csv'))
data_path = os.path.join('.', 'data', 'raw', '5HT', 'noise_comparison/')

experiments = {
    '50': [],
    '3': []
}

for i in range(file_index.shape[0]):
    try:
        with gagProcess():

            tmp_experiment = Experiment(file_index.loc[i, 'Cell'], 0.1)
            tmp_experiment.setAECTrace(
                'Axon', fname = data_path + file_index.loc[i, 'AEC2'],
                V_channel = 0, I_channel = 1
            )

            for ind in ['1', '2', '3']:

                tmp_experiment.addTrainingSetTrace(
                    'Axon', fname = data_path + file_index.loc[i, 'Train' + ind],
                    V_channel = 0, I_channel = 1
                )
                tmp_experiment.addTestSetTrace(
                    'Axon', fname = data_path + file_index.loc[i, 'Test' + ind],
                    V_channel = 0, I_channel = 1
                )

            for tr in tmp_experiment.testset_traces:
                tr.detectSpikes()


        if file_index.loc[i, 'OU_tau'] == 3:
            experiments['3'].append(tmp_experiment)
        elif file_index.loc[i, 'OU_tau'] == 50:
            experiments['50'].append(tmp_experiment)

    except RuntimeError:
        print 'Problem with {} import. Skipping.'.format(file_index.loc[i, 'Cell'])

#%% COMPUTE RELIABILITIES

# Allocate dicts to store output.
reliabilities = {}
windows_arrs = {}

# Compute window sizes.
windows = np.logspace(
    np.log10(1),
    np.log10(100),
    20
)

# Iterate over cell types.
for key in experiments.keys():
    reliabilities[key] = []

    # Iterate over window sizes.
    for i, win in enumerate(windows):
        if True:
            print 'Extracting {} reliability {:.1f}%'.format(
                key, 100*(i + 1)/len(windows)
            )

        # Iterate over experiments.
        reliabilities_tmp = []
        for expt in experiments[key]:
            reliabilities_tmp.append(intrinsic_reliability(expt.testset_traces, win, 0.1))

        reliabilities[key].append(reliabilities_tmp)

    reliabilities[key] = np.array(reliabilities[key])
    windows_arrs[key] = np.tile(windows[:, np.newaxis], (1, reliabilities[key].shape[1]))

#%% SAVE DIAGNOSTIC PLOT

plt.figure()

plt.subplot(111)
plt.semilogx(windows_arrs['3'], reliabilities['3'], 'k', label = '3ms')
plt.semilogx(windows_arrs['50'], reliabilities['50'], 'r', label = '50ms')
plt.ylabel('Intrinsic reliability')
plt.xlabel('Precision (s)')
plt.ylim(0, 1)

plt.legend()

plt.savefig(os.path.join('figs', 'ims', 'reliability', '5HT_comparison.png'), dpi = 200)

plt.show()
