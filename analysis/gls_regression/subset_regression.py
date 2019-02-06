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

from model_evaluation import *


#%% READ IN DATA

from load_experiments import experiments


#%%

bad_cells_ = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]
coeffs = {
    'good': [],
    'bad': []
}

for i, expt in enumerate(experiments):

    X, y = build_Xy(expt)

    mask = X[:, 0] > -60

    betas = OLS_fit(X[mask, :], y[mask])

    if i in bad_cells_:
        coeffs['bad'].append(betas)
    else:
        coeffs['good'].append(betas)

for key in ['good', 'bad']:
    coeffs[key] = pd.DataFrame(coeffs[key])
    coeffs[key]['group'] = key

coeffs = coeffs['good'].append(coeffs['bad'])

coeffs = convert_betas(coeffs)

#%%

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')
IMG_PATH = './figs/ims/var_dists/'

g = sns.jointplot(
    x = 'gk1', y = 'gk2', data = coeffs.loc[coeffs['group'] == 'good', :], kind = 'kde'
)
g.ax_joint.plot(
    coeffs.loc[coeffs['group'] == 'good', 'gk1'],
    coeffs.loc[coeffs['group'] == 'good', 'gk2'],
    'ko', markeredgecolor = 'white'
)
g.ax_joint.plot(
    coeffs.loc[coeffs['group'] == 'bad', 'gk1'],
    coeffs.loc[coeffs['group'] == 'bad', 'gk2'],
    'r.', alpha = 0.6
)
g.fig.set_size_inches(4, 4)
plt.tight_layout()
if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'gk_dist_-60cutoff.png')
plt.show()

#%%

method = {'base': [3, 4], 'gk1': [4], 'gk2': [3], 'all':[-1]}

cv_binned_error = {}

for key, excl_cols in method.iteritems():

    print('{}'.format(key))

    tmp_error = {
        'train': [],
        'test': []
    }

    for i, expt in enumerate(experiments):

        print '\r{:.1f}%'.format(100*(i + 1)/len(experiments)),

        if i > 5:
            pass

        X, y = build_Xy(expt, excl_cols)

        mask = X[:, 0] > -60

        train_err_tmp, test_err_tmp = cross_validate(X[mask, :], y[mask], OLS_fit)

        tmp_error['train'].append(train_err_tmp)
        tmp_error['test'].append(test_err_tmp)

    tmp_error['train'] = np.array(tmp_error['train'])
    tmp_error['test'] = np.array(tmp_error['test'])

    cv_binned_error[key] = tmp_error

    print 'Done!'

with open('analysis/gls_regression/cv_dV_error_1565spikecut_-60mV.pyc', 'wb') as f:
    pickle.dump(cv_binned_error, f)

#%%

with open('analysis/gls_regression/cv_dV_error_1565spikecut_-60mV.pyc', 'rb') as f:
    cv_binned_error = pickle.load(f)

def cv_plot(error_dict, gridspec_, bad_cells = None):

    if bad_cells is not None:
        good_cells = [i for i in range(error_dict['train'].shape[0]) if i not in bad_cells]
    else:
        good_cells = [i for i in range(error_dict['train'].shape[0])]

    spec = gs.GridSpecFromSubplotSpec(1, 3, gridspec_, wspace = 0.5)

    ax1 = plt.subplot(spec[0, 0])
    #plt.title('CV10 train error', loc = 'left')
    if bad_cells is not None:
        plt.semilogy(
            error_dict['train'][bad_cells, 0, :].T, error_dict['train'][bad_cells, 1, :].T,
            'r-', lw = 0.5, alpha = 0.7
        )
    plt.semilogy(
        error_dict['train'][good_cells, 0, :].T, error_dict['train'][good_cells, 1, :].T,
        'k-', lw = 0.5
    )
    plt.ylabel(r'Error $\left( \frac{\mathrm{mV}^2}{\mathrm{ms}^2} \right)$')
    plt.xlabel(r'$V_m$ (mV)')

    ax2 = plt.subplot(spec[0, 1])
    plt.title('CV10 test err.', loc = 'left')
    if bad_cells is not None:
        plt.semilogy(
            error_dict['test'][bad_cells, 0, :].T, error_dict['test'][bad_cells, 1, :].T,
            'r-', lw = 0.5, alpha = 0.7
        )
    plt.semilogy(
        error_dict['test'][good_cells, 0, :].T, error_dict['test'][good_cells, 1, :].T,
        'k-', lw = 0.5
    )
    plt.xlabel(r'$V_m$ (mV)')

    ax3 = plt.subplot(spec[0, 2])
    plt.title('Error ratio', loc = 'left')
    plt.axhline(1, color = 'k', lw = 0.5, ls = '--', dashes = (10, 10))
    if bad_cells is not None:
        plt.plot(
            error_dict['train'][bad_cells, 0, :].T, error_dict['test'][bad_cells, 1, :].T / error_dict['train'][bad_cells, 1, :].T,
            'r-', lw = 0.5, alpha = 0.7
        )
    plt.plot(
        error_dict['train'][good_cells, 0, :].T, error_dict['test'][good_cells, 1, :].T / error_dict['train'][good_cells, 1, :].T,
        'k-', lw = 0.5
    )
    #plt.gca().set_yticks([0.9, 0.95, 1, 1.05])
    #plt.gca().set_yticklabels(['$0.90$', '$0.95$', '$1.00$', '$1.05$'])
    plt.ylabel('Test/train error ratio')
    plt.xlabel(r'$V_m$ (mV)')

    return ax1, ax2, ax3

IMG_PATH = './figs/ims/regression_tinkering/'

bad_cells = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

spec = gs.GridSpec(4, 1, hspace = 0.6)

plt.figure(figsize = (6, 8))

for i, key in enumerate(['base', 'gk1', 'gk2', 'all']):
    ax1, _, _ = cv_plot(cv_binned_error[key], spec[i, :], bad_cells)
    ax1.set_title('{} CV10 train err.'.format(key), loc = 'left')

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'cv_dV_error_ols_1565spikecut_-60mV.png')

plt.show()
