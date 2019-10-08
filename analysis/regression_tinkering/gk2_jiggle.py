#%% DESCRIPTION

"""
gbar_k2 is zero most of the time for 5HT cells under the following conditions:
- Fast noise
- 1.5 pre to 6.5 post spikecut
- gl, C, gk1, gk2 restricted to positive values
- Fitting restricted to points above -60mV

Here I try to mess with the gk2 gating function in various ways to obtain more
reasonable estimates for gbar_k2.
"""

#%% IMPORT MODULES

from __future__ import division

import pickle

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

from grr import pltools

from model_evaluation import *


#%% READ IN DATA

from load_experiments import experiments

#%%

# Initialize models to use for fitting.
mods = {
    'base': AugmentedGIF(0.1),
    'V0 -5': AugmentedGIF(0.1),
    'V0 -10': AugmentedGIF(0.1),
    'tau_n 50': AugmentedGIF(0.1),
    'tau_n 50 V0 -5': AugmentedGIF(0.1)
}

mods['V0 -5'].n_Vhalf           -=5
mods['V0 -10'].n_Vhalf          -= 10
mods['tau_n 50'].n_tau          = 50
mods['tau_n 50 V0 -5'].n_Vhalf  -= 5
mods['tau_n 50 V0 -5'].n_tau    = 50

# Dict to hold output
fitted_coeffs = {}

# List of bad cells
bad_cells_ = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]

# Iterate over models
for key, mod in mods.iteritems():

    coeffs = {
        'good': [],
        'bad': []
    }

    # Iterate over experiments.
    for i, expt in enumerate(experiments):

        print '\rFitting {} {:.1f}%'.format(key, 100 * i / len(experiments)),


        X, y = build_Xy(expt, GIFmod = mod)
        mask = X[:, 0] > -60

        betas = optimize.lsq_linear(
            X, y,
            bounds = (
                np.concatenate(([-np.inf, 0, -np.inf, 0, 0], np.full(X.shape[1] - 5, -np.inf))),
                np.concatenate(([0, np.inf, np.inf, np.inf, np.inf], np.full(X.shape[1] - 5, np.inf)))
            )
        )['x']

        if i in bad_cells_:
            coeffs['bad'].append(betas)
        else:
            coeffs['good'].append(betas)

    for key_ in ['good', 'bad']:
        coeffs[key_] = pd.DataFrame(coeffs[key_])
        coeffs[key_]['group'] = key_

    coeffs = coeffs['good'].append(coeffs['bad'])
    coeffs = convert_betas(coeffs)

    fitted_coeffs[key] = coeffs

    print 'Done!'


#%% MAKE SWARM PLOTS OF GK PARAMETERS

def gkswarm(data, key, ax = None):

    if ax is None:
        ax = plt.gca()

    ax.set_ylim(0, 0.015)
    sns.swarmplot(
        x = 'variable', y = 'value', hue = 'group',
        data = data[key].loc[:, ['gk1', 'gk2', 'group']].melt(id_vars = ['group']),
        palette = ['k', 'r'], clip_on = False, linewidth = 0.5, edgecolor = 'gray',
        ax = ax
    )
    ax.get_legend().remove()
    ax.set_xlabel('')
    ax.set_ylabel('Conductance (uS)')

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')
IMG_PATH = './figs/ims/var_dists/'

spec = gs.GridSpec(2, 3)

plt.figure(figsize = (6, 4))

for i, key in enumerate(['base', 'V0 -5', 'V0 -10', 'tau_n 50', 'tau_n 50 V0 -5']):
    plt.subplot(spec[i // 3, i % 3])
    gkswarm(fitted_coeffs, key)
    plt.title(''.join([char for char in key if not char == '_']))

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'gkdist_tweaked_gk2act.png')

plt.show()
