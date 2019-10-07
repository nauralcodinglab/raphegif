#%% IMPORT MODULES

from __future__ import division

import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import optimize

import sys
sys.path.append('./src')

sys.path.append('./analysis/gls_regression')

from grr.AugmentedGIF import AugmentedGIF

import src.pltools as pltools

from model_evaluation import *


#%% READ IN DATA

from load_experiments import experiments


#%%

# Initialize models to use for fitting.
taus = np.arange(10, 200.5, 1)
mods = {}

for i, tau in enumerate(taus):
    mods[tau] = AugmentedGIF(0.1)
    mods[tau].h_tau = tau
del taus

# Dict to hold output
fitted_coeffs = {}

# List of bad cells
bad_cells_ = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]

# Iterate over models
j = 0
for key, mod in mods.iteritems():

    if j > 0:
        pass

    coeffs = []

    # Iterate over experiments.
    for i, expt in enumerate(experiments):

        print '\rFitting {} {:.1f}%'.format(key, 100 * i / len(experiments)),

        if i > 1:
            pass

        X, y = build_Xy(expt, GIFmod = mod)
        mask = X[:, 0] > -60

        betas = optimize.lsq_linear(
            X, y,
            bounds = (
                np.concatenate(([-np.inf, 0, -np.inf, 0, 0], np.full(X.shape[1] - 5, -np.inf))),
                np.concatenate(([0, np.inf, np.inf, np.inf, np.inf], np.full(X.shape[1] - 5, np.inf)))
            )
        )['x'].tolist()

        if i in bad_cells_:
            group_ = 'bad'
        else:
            group_ = 'good'

        var_expl_ = var_explained(X, betas, y)

        row = deepcopy(betas)
        row.extend([group_, var_expl_, expt.name])

        coeffs.append(row)

    coeffs = pd.DataFrame(coeffs)
    coeffs = coeffs.rename({
        coeffs.shape[1] - 3: 'group',
        coeffs.shape[1] - 2: 'var_explained',
        coeffs.shape[1] - 1: 'cell_ID'
        }, axis = 1)

    tmp = convert_betas(coeffs)
    tmp['var_explained'] = coeffs['var_explained']
    tmp['cell_ID'] = coeffs['cell_ID']

    fitted_coeffs[key] = tmp
    del tmp

    print 'Done!'

    j += 1


master_coeffs = pd.DataFrame()

for key in fitted_coeffs.keys():

    tmp = deepcopy(fitted_coeffs[key])
    tmp['tau_h'] = key

    master_coeffs = master_coeffs.append(tmp)

with open('./analysis/gls_regression/tauh_linesearch_coeffs_AEC.pyc', 'wb') as f:
    pickle.dump(master_coeffs, f)


#%%

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')
IMG_PATH = './figs/ims/var_dists/'

var_explained_pivot = master_coeffs.pivot(index = 'tau_h', values = 'var_explained', columns = 'cell_ID')
maxima = []
for colname in var_explained_pivot:
    maxima.append(
        [np.argmax(var_explained_pivot[colname]),
        np.max(var_explained_pivot[colname])]
    )
maxima = np.array(maxima)

cells_mask = np.array([i in bad_cells_ for i in range(var_explained_pivot.shape[1])])

plt.figure(figsize = (6, 6))
plt.subplot(221)
plt.title('Var. explained', loc = 'left')
plt.plot(var_explained_pivot.loc[:, cells_mask], 'r-', alpha = 0.7, lw = 0.7)
plt.plot(var_explained_pivot.loc[:, ~cells_mask], 'k-', alpha = 0.7, lw = 0.7)
plt.plot(
    maxima[cells_mask, 0], maxima[cells_mask, 1],
    'o', markeredgecolor = 'gray', markerfacecolor = 'r'
)
plt.plot(
    maxima[~cells_mask, 0], maxima[~cells_mask, 1],
    'o', markeredgecolor = 'gray', markerfacecolor = 'k'
)
plt.ylabel(r'$R^2$ on $\frac{dV}{dt}$')
plt.xlabel(r'$\tau_h$')

plt.subplot(222)
plt.title('Relative var. explained', loc = 'left')
plt.plot(
    var_explained_pivot.loc[:, cells_mask] / np.max(var_explained_pivot.loc[:, cells_mask], axis = 0),
    'r-', alpha = 0.7, lw = 0.7
)
plt.plot(
    var_explained_pivot.loc[:, ~cells_mask] / np.max(var_explained_pivot.loc[:, ~cells_mask], axis = 0),
    'k-', alpha = 0.7, lw = 0.7
)
#plt.plot(maxima[:, 0], np.ones_like(maxima[:, 0]), '.', color = 'gray')
plt.ylabel(r'Fraction of max $R^2$ on $\frac{dV}{dt}$')
plt.xlabel(r'$\tau_h$')

hist_ax = plt.subplot(223)
plt.title(r'Optimal $\tau_h$ ``good" cells', loc = 'left')
hist_ax.hist(maxima[~cells_mask, 0], color = 'k')
hist_ax.set_ylabel('Count')

kde_ax = hist_ax.twinx()
sns.kdeplot(maxima[~cells_mask, 0], ax = kde_ax, color = 'gray')
kde_ax.set_yticks([])

hist_ax.axvline(np.median(maxima[~cells_mask, 0]), color = 'gray', ls = '--', label = 'Median')
hist_ax.set_xlabel(r'$\tau_h$')
hist_ax.set_xticks([0, 50, 100, 150])
hist_ax.legend()


hist_ax = plt.subplot(224)
plt.title(r'Optimal $\tau_h$ ``bad" cells', loc = 'left')
hist_ax.hist(maxima[cells_mask, 0], color = 'r')
hist_ax.set_ylabel('Count')

kde_ax = hist_ax.twinx()
sns.kdeplot(maxima[cells_mask, 0], ax = kde_ax, color = (0.7, 0.2, 0.2))
kde_ax.set_yticks([])

hist_ax.axvline(np.median(maxima[cells_mask, 0]), color = (0.7, 0.2, 0.2), ls = '--', label = 'Median')
hist_ax.set_xlabel(r'$\tau_h$')
hist_ax.set_xticks([0, 50, 100, 150])
hist_ax.legend()


plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'tauh_linesearch_new.png', dpi = 300)

plt.show()
