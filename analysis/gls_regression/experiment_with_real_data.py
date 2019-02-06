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

x = np.arange(-100, -20)
plt.plot(x, np.log(1 + 1.1**(x + 50)))

#%%

KGIF = AugmentedGIF(0.1)

def cross_validate(X, y, fitting_func, k = 10, random_seed = 42, verbose = False):
    """Returns tuple of mean cross-validated training and test error.
    """

    np.random.seed(random_seed)
    inds = np.random.permutation(len(y))
    groups = np.split(inds, k)

    var_explained_dict = {
        'train': [],
        'test': []
    }

    for i in range(k):

        if verbose:
            if i == 0:
                print '\n'
            print '\rCross-validating {:.1f}%'.format(100 * i / k),

        test_inds_tmp = groups[i]
        train_inds_tmp = np.concatenate([gr for j, gr in enumerate(groups) if j != i], axis = None)

        train_y = y[train_inds_tmp]
        train_X = X[train_inds_tmp, :]

        test_y = y[test_inds_tmp]
        test_X = X[test_inds_tmp, :]

        betas = fitting_func(train_X, train_y)
        var_explained_dict['train'].append(var_explained(train_X, betas, train_y))
        var_explained_dict['test'].append(var_explained(test_X, betas, test_y))

    if verbose:
        print '\rDone!                 '

    return np.mean(var_explained_dict['train']), np.mean(var_explained_dict['test'])


def WLS_fit(X, y):
    voltage = X[:, 0]
    #wts = np.exp((voltage - voltage.mean())/ voltage.std())
    wts = np.log(1 + 1.1**(voltage + 50))

    wtsy = wts * y
    wtsX = wts[:, np.newaxis] * X

    XTwtsX = np.dot(X.T, wtsX)
    XTwtsy = np.dot(X.T, wtsy)
    betas = np.linalg.solve(XTwtsX, XTwtsy)

    return betas


def OLS_fit(X, y):
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, y)
    betas = np.linalg.solve(XTX, XTY)
    return betas

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
    'WLS': []
}

for i, expt in enumerate(experiments):

    print('{:.1f}%'.format(100*(i + 1)/len(experiments)))

    if i > 5:
        pass

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
    betas = WLS_fit(X, y)
    var_exp_tmp = var_explained(X, betas, y)
    estimates = estimates.append(generate_row(expt, 'WLS', betas[:5], var_exp_tmp), ignore_index = True)

    # gk2 only
    X_tmp = X[:, [x != 3 for x in range(X.shape[1])]]
    betas = WLS_fit(X_tmp, y)
    var_exp_tmp = var_explained(X_tmp, betas, y)
    betas = np.concatenate([betas[:3], [0], [betas[3]]])
    estimates = estimates.append(generate_row(expt, 'gk2', betas, var_exp_tmp), ignore_index = True)

    # gk1 only
    X_tmp = X[:, [x != 4 for x in range(X.shape[1])]]
    betas = WLS_fit(X_tmp, y)
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
tmp = cross_validate(X, y, OLS_fit, verbose = True)
tmp
