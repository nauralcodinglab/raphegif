#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd

from src.Experiment import *
from src.AEC_Badel import *
from src.GIF import *
from src.AugmentedGIF import *
from src.Filter_Rect_LogSpaced import *
from src.Filter_Exps import Filter_Exps

import src.pltools as pltools
from src.Tools import gagProcess


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

    #expt.plotTestSet()


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
AugmentedGIFs = []

for i, expt in enumerate(experiments):

    print('Fitting GIFs {:.1f}%'.format(100 * (i + 1) / len(experiments)))

    for j, tmp_GIF in enumerate([GIF(0.1), AugmentedGIF(0.1)]):

        with gagProcess():

            # Define parameters
            tmp_GIF.Tref = 4.0

            tmp_GIF.eta = Filter_Rect_LogSpaced()
            tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


            tmp_GIF.gamma = Filter_Rect_LogSpaced()
            tmp_GIF.gamma.setMetaParameters(5000., 150, 2000., slope = 2)

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


#%% PICKLE DATA

with open(DATA_PATH + '5HT_gamma_rect.pyc', 'wb') as f:
    obj = {'GIFs': GIFs, 'AugmentedGIFs': AugmentedGIFs, 'experiments': experiments}
    pickle.dump(obj, f)
