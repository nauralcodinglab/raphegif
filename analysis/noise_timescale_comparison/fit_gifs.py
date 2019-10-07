#%%

from __future__ import division

import os
import pickle

import pandas as pd
import numpy as np

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.Tools import gagProcess
from grr.Filter_Exps import Filter_Exps


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


#%% AEC

AECs = []

for key in experiments.keys():
    for expt in experiments[key]:

        with gagProcess():

            tmp_AEC = AEC_Badel(expt.dt)

            tmp_AEC.K_opt.setMetaParameters(length=150.0, binsize_lb=expt.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
            tmp_AEC.p_expFitRange = [3.0,150.0]
            tmp_AEC.p_nbRep = 15

            # Assign tmp_AEC to expt and compensate the voltage recordings
            expt.setAEC(tmp_AEC)
            expt.performAEC()

        AECs.append(tmp_AEC)

for key in experiments:
    with open('data/processed/noise_comparison/experiments_{}ms.ldat'.format(key), 'wb') as f:
        pickle.dump(experiments[key], f)
        f.close()


#%% FIT GIFs
GIFs = {}

for key in experiments.keys():
    GIFs[key] = []
    for i, expt in enumerate(experiments[key]):

        print('Fitting GIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

        tmp_GIF = GIF(0.1)
        tmp_GIF.name = expt.name

        with gagProcess():

            # Define parameters
            tmp_GIF.Tref = 6.5

            tmp_GIF.eta = Filter_Exps()
            tmp_GIF.eta.setFilter_Timescales([3, 10, 30, 100, 300, 1000, 3000])

            tmp_GIF.gamma = Filter_Exps()
            tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

            # Define the ROI of the training set to be used for the fit
            for tr in expt.trainingset_traces:
                tr.setROI([[1000,59000]])
            for tr in expt.testset_traces:
                tr.setROI([[500, 14500]])

            tmp_GIF.fit(expt, DT_beforeSpike=1.5)

        GIFs[key].append(tmp_GIF)
        tmp_GIF.printParameters()

for key in experiments:
    with open('data/processed/noise_comparison/models_{}ms.mod'.format(key), 'wb') as f:
        pickle.dump(GIFs[key], f)
        f.close()
