#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from scipy import stats

import sys
sys.path.append('./src')
sys.path.append('./figs/scripts')
sys.path.append('./analysis/gls_regression')

from Experiment import *
from AEC_Badel import *
from GIF import *
from AugmentedGIF import *
from Filter_Rect_LogSpaced import *
from Filter_Exps import Filter_Exps

import pltools

import bootstrap_proof_of_concept as bs

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


#%%

def OLS_fit(X, y):
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, y)
    betas = np.linalg.solve(XTX, XTY)
    return betas

def build_Xy(experiment, excl_cols = None):

    X = []
    y = []

    KGIF = AugmentedGIF(0.1)
    KGIF.Tref = 6.5

    for tr in expt.trainingset_traces:
        X_tmp, y_tmp = KGIF.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, 1.5)

        X.append(X_tmp)
        y.append(y_tmp)

    X = np.concatenate(X)
    if excl_cols is not None:
        X = X[:, [x for x in range(X.shape[1]) if x not in excl_cols]]
    y = np.concatenate(y)

    return X, y


#%%

X, y = build_Xy(experiments[0])
OLS_fit(X, y)
np.random.uniform(X[:, 0].min(), X[:, 0].max(), size = int(X.shape[0] * 0.25))

#%%

subsample = 50
for i, expt in enumerate([experiments[0]]):

    print 'Boostrapping experiment {} of {}'.format(i + 1, len(experiments))

    X, y = build_Xy(expt)

    OLS_est = OLS_fit(X, y)
    print 'Done OLS_fit'

    beta_est = []

    no_reps = 5

    np.random.seed(24)
    for j in range(no_reps):
        print '{:.1f}'.format(100*j/no_reps),

        no_bs_pts = int(X.shape[0] / subsample * 0.10)
        bs_pts = np.array([
            np.random.uniform(X[:, 0].min(), X[:, 0].max(), size = (no_bs_pts)),
            np.random.uniform(X[:, 1].min(), X[:, 1].max(), size = (no_bs_pts))
        ]).T
        _, bs_y, inds = bs.subsample_cs(X[::subsample, (0, 1)], y, bs_pts)

        bs_X = X[inds, :]

        beta_est.append(OLS_fit(bs_X, bs_y))

    beta_est = np.array(beta_est)

bs_X.sum(axis = 0)
#%%
