#%% IMPORT MODULES

from __future__ import division

import os
import pickle
import argparse

import numpy as np
import pandas as pd

from grr.Experiment import Experiment

from grr.Tools import gagProcess


#%% PARSE COMMANDLINE ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument(
    'output', help = 'Path to output processed Experiments.'
)
parser.add_argument(
    '-v', '--verbose', help = 'Print information about progress.',
    action = 'store_true'
)
args = parser.parse_args()


#%% LOAD DATA

data_paths = {
    '5HT': './data/raw/5HT/fast_noise/',
    'GABA': './data/GABA_cells/',
    'mPFC': './data/raw/mPFC/mPFC_spiking/'
}

fname_paths = {
    '5HT': './data/raw/5HT/fast_noise/index.csv',
    'GABA': './data/GABA_cells/index.csv',
    'mPFC': './data/raw/mPFC/fnames.csv'
}

experiments = {
    '5HT': [],
    'GABA': [],
    'mPFC': []
}

if args.verbose:
    print 'Loading data.'

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

            tmp_experiment = Experiment(fnames.loc[i, 'Experimenter'] + fnames.loc[i, 'Cell_ID'], 0.1)
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


#%% SAVE DATA

if args.verbose:
    print 'Saving preprocessed experiments to {}'.format(args.output)
with open(args.output, 'wb') as f:
    pickle.dump(experiments, f)
    f.close()
if args.verbose:
    print 'Done!'

