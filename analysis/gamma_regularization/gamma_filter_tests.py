#%% IMPORT MODULES

from __future__ import division

import pickle
import os
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.AugmentedGIF import AugmentedGIF
from src.Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from grr.Filter_Exps import Filter_Exps

import src.pltools as pltools
from grr.Tools import gagProcess


#%% READ IN DATA

if __name__ == '__main__':

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

if __name__ == '__main__':

    for expt in experiments:

        for tr in expt.testset_traces:
            tr.detectSpikes()

        #expt.plotTestSet()


#%% KEEP GOOD EXPERIMENTS

if __name__ == '__main__':
    bad_cells = []

    for i in np.flip([2, 3, 4, 7, 9, 10, 13], -1):
        bad_cells.append(experiments.pop(i))


#%% PERFORM AEC

if __name__ == '__main__':

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



#%% DEFINE FUNCTION FOR PARALLELIZED GIF FITTING

def fit_GIF_exp_gamma(args):

    if len(args) != 3:
        raise ValueError('args must be composed of a tuple of (timescales, experiments, GIFs)')

    Md_precision = 8.

    timescales  = args[0]
    experiments = args[1]
    GIFs        = args[2]

    models = [[] for i in range(len(GIFs))]
    Md_vals = [[] for i in range(len(GIFs))]

    for i, expt in enumerate(experiments):

        print('Fitting experiment {} of {}'.format(i + 1, len(experiments)))

        for j, tmp_GIF in enumerate(GIFs):

            with gagProcess():

                ### Create GIF
                # Define parameters
                tmp_GIF.Tref = 4.0

                tmp_GIF.eta = Filter_Rect_LogSpaced()
                tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

                tmp_GIF.gamma = Filter_Exps()
                tmp_GIF.gamma.setFilter_Timescales(timescales)

                # Define the ROI of the training set to be used for the fit
                for tr in expt.trainingset_traces:
                    tr.setROI([[1000,59000]])
                for tr in expt.testset_traces:
                    tr.setROI([[500, 14500]])

                ### Fit GIF
                tmp_GIF.fit(expt, DT_beforeSpike=5.0)

                ### Compute Md*
                tmp_prediction = expt.predictSpikes(tmp_GIF, nb_rep=500)
                Md_vals[j].append(tmp_prediction.computeMD_Kistler(Md_precision, 0.1))

            models[j].append(tmp_GIF)

    print('Done!')

    return {'timescales': timescales, 'experiments': experiments, 'models': models, 'Md': Md_vals}

#fit_GIF_exp_gamma([[500], experiments[:2], [GIF(0.1), AugmentedGIF(0.1)]])


#%% PERFORM PARALLELIZED GIF FITTING

if __name__ == '__main__':

    DATA_PATH = './data/raw/5HT/fast_noise/'

    timescales_list = [[10], [30], [50], [100], [300], [500], [1000],
                       [10, 100], [30, 300], [50, 500], [100, 1000], [300, 3000],
                       [10, 100, 1000], [30, 300, 3000]]

    experiments_list    = [experiments for i in range(len(timescales_list))]
    models_list         = [[GIF(0.1), AugmentedGIF(0.1)] for i in range(len(timescales_list))]

    p = mp.Pool(mp.cpu_count() // 2)
    out = p.map(fit_GIF_exp_gamma, zip(timescales_list, experiments_list, models_list))
    p.close()

    with open(DATA_PATH + 'gamma_filter_tests_output.pyc', 'wb') as f:
        pickle.dump(out, f)
