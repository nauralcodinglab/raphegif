#%% IMPORT MODULES

from __future__ import division

import os

import numpy as np
import pandas as pd

import sys
sys.path.append('./src')
sys.path.append('./figs/scripts')

from Experiment import *
from AEC_Badel import *


#%% READ IN DATA

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout



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


for expt in experiments:

    for tr in expt.trainingset_traces:
        tr.detectSpikes()

    for tr in expt.testset_traces:
        tr.detectSpikes()

    #expt.plotTestSet()
