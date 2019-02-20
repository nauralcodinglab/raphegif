#%% IMPORT MODULES

from __future__ import division

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('./src')

from Experiment import Experiment
from AEC_Badel import AEC_Badel


#%% DEFINE FUNCTION TO GAG VERBOSE POZZORINI FUNCTIONS

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


#%% READ IN DATA

DATA_PATH = './data/GABA_cells/'

file_index = pd.read_csv(DATA_PATH + 'index.csv')

experiments = []

for i in range(file_index.shape[0]):

    try:
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

    except RuntimeError:
        # Seems to be due to an issue with units expected by Experiment._readABF().
        # Probably a data problem rather than code problem.
        print 'Problem with {} import. Skipping.'.format(file_index.loc[i, 'Cell'])


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


#%% DETECT SPIKES

for expt in experiments:

    for tr in expt.trainingset_traces:
        tr.detectSpikes()

    for tr in expt.testset_traces:
        tr.detectSpikes()
