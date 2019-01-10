#%% IMPORT MODULES

from __future__ import division

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

import sys
sys.path.append('./src')
sys.path.append('./figs/scripts')

from Experiment import *
from AEC_Badel import *
from GIF import *
from AugmentedGIF import *
from Filter_Rect_LogSpaced import *
from Filter_Exps import Filter_Exps
from SpikeTrainComparator import intrinsic_reliability

import pltools

#%% DEFINE FUNCTION TO GAG VERBOSE POZZORINI FUNCTIONS

class gagProcess(object):

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

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


#%%

for expt in experiments:

    for tr in expt.trainingset_traces:
        tr.detectSpikes()

    for tr in expt.testset_traces:
        tr.detectSpikes()


#%%

tr = experiments[0].trainingset_traces[0]

selection = tr.getROI_FarFromSpikes(5., 4.)
I = tr.I[selection]
V = tr.V[selection]


plt.figure()
plt.plot(I, V, 'k.', alpha = 0.01)
plt.show()



#%%

def cov_(X):

    X_norm = X - X.mean(axis = 0)
    X_normT = X_norm.T.copy()
    return np.dot(X_normT, X_norm) / (X_norm.shape[0] - 1)

cov_tmp = cov_(X)[:5, :5]
plt.matshow(cov_tmp)
cov_tmp.diagonal()


#%%

KGIF = AugmentedGIF(0.1)

def gls_fit(X, y, downsample = 1000):

    if downsample > 1:
        X = np.copy(X[::downsample, :])
        y = np.copy(y[::downsample])
    elif downsample == 1:
        pass

    print('Computing covariance matrix.')
    V = cov_(X.T)

    print('Weighting observations.')
    Vinvy = np.linalg.solve(V, y)
    VinvX = np.linalg.solve(V, X)

    print('Obtaining coefficients.')
    XTVX = np.dot(X.T, VinvX)
    XTVy = np.dot(X.T, Vinvy)
    betas = np.linalg.solve(XTVX, XTVy)
    print('Done!')

    return betas

def wls_fit(X, y):
    voltage = X[:, 0]
    wts = np.exp((voltage - voltage.mean())/ voltage.std())

    wtsy = wts * y
    wtsX = wts[:, np.newaxis] * X

    XTwtsX = np.dot(X.T, wtsX)
    XTwtsy = np.dot(X.T, wtsy)
    betas = np.linalg.solve(XTwtsX, XTwtsy)

    return betas


def OLS_fit(X, y):
    XTX_inv = np.linalg.inv(np.dot(X.T, X))
    XTY = np.dot(X.T, y)
    return np.dot(XTX_inv, XTY)

def var_explained(X, betas, y):
    yhat = np.dot(X, betas)
    MSE = np.mean((y - yhat)**2)
    var = np.var(y)
    return 1. - MSE/var

def var_explained_binned(X, betas, y, bins = 'default'):
    """
    Returns a tuple of bin centres and binned means.
    """

    if bins == 'default':
        bins = np.arange(-90, -20, 5)

    V = X[:, 0]

    yhat = np.dot(X, betas)
    squared_errors = (y - yhat)**2

    mean_, edges, bin_no = stats.binned_statistic(
        V, squared_errors, 'mean', bins, [-80, -20]
    )

    bin_centres = (edges[1:] + edges[:-1]) / 2.

    return (bin_centres, mean_)


def generate_row(expt, method, betas, R2):

    tmp = {
        'Cell': expt.name,
        'method': method,
        'b1': betas[0],
        'b2': betas[1],
        'b3': betas[2],
        'b4': betas[3],
        'b5': betas[4],
        'R2': R2
    }

    return tmp

estimates = pd.DataFrame(columns = ['Cell', 'method', 'b1', 'b2', 'b3', 'b4', 'b5', 'R2'])
binned_mse = {
    'all': [],
    'wls': []
}

for i, expt in enumerate(experiments):

    print('{:.1f}%'.format(100*(i + 1)/len(experiments)))

    if i > 0:
        continue

    X = []
    y = []

    for tr in expt.trainingset_traces:
        X_tmp, y_tmp = KGIF.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr)

        X.append(X_tmp)
        y.append(y_tmp)

    X = np.concatenate(X)
    y = np.concatenate(y)

    cellname = expt.name

    # All coeffs method.
    betas = OLS_fit(X, y)
    var_exp_tmp = var_explained(X, betas, y)
    estimates = estimates.append(generate_row(expt, 'all', betas[:5], var_exp_tmp), ignore_index = True)

    # GLS method
    betas = wls_fit(X, y)
    var_exp_tmp = var_explained(X, betas, y)
    estimates = estimates.append(generate_row(expt, 'wls', betas[:5], var_exp_tmp), ignore_index = True)

    # gk2 only
    X_tmp = X[:, [x != 3 for x in range(X.shape[1])]]
    betas = OLS_fit(X_tmp, y)
    var_exp_tmp = var_explained(X_tmp, betas, y)
    betas = np.concatenate([betas[:3], [0], [betas[3]]])
    estimates = estimates.append(generate_row(expt, 'gk2', betas, var_exp_tmp), ignore_index = True)

    # gk1 only
    X_tmp = X[:, [x != 4 for x in range(X.shape[1])]]
    betas = OLS_fit(X_tmp, y)
    var_exp_tmp = var_explained(X_tmp, betas, y)
    betas = np.concatenate([betas[:4], [0]])
    estimates = estimates.append(generate_row(expt, 'gk1', betas, var_exp_tmp), ignore_index = True)


estimates['C'] = 1./estimates['b2']
estimates['gl'] = -estimates['b1'] * estimates['C']
estimates['El'] = estimates['b3'] * estimates['C'] / estimates['gl']
estimates['gk1'] = estimates['b4'] * estimates['C']
estimates['gk2'] = estimates['b5'] * estimates['C']
estimates.drop(['b' + str(i) for i in range(1, 6)], axis = 1, inplace = True)

#%%

for lab in ['C', 'gl', 'El', 'gk1', 'gk2', 'R2']:
    plt.figure(figsize = (6, 4))
    sns.swarmplot('method', lab, data = estimates)
    plt.tight_layout()
    plt.show()


#%%
