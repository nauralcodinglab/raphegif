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
from scipy import optimize

import sys
sys.path.append('./analysis/gls_regression')

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.SplineGIF import SplineGIF
from grr.Filter_Rect import Filter_Rect_LogSpaced
from grr.Filter_Exps import Filter_Exps

from model_evaluation import *

#%% LOAD DATA

from load_experiments import experiments


#%% PERFORM CONSTRAINED REGRESSION

bad_cells_ = [1, 2, 3, 4, 7, 9, 11, 12, 13, 18]
coeffs = {
    'good': [],
    'bad': []
}

for i, expt in enumerate(experiments):

    print '\rFitting {:.1f}%'.format(100 * i / len(experiments)),

    X, y = build_Xy(expt)
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

for key in ['good', 'bad']:
    coeffs[key] = pd.DataFrame(coeffs[key])
    coeffs[key]['group'] = key

coeffs = coeffs['good'].append(coeffs['bad'])
coeffs = convert_betas(coeffs)

print 'Done!'


#%% JOINTPLOT OF GK ESTIMATES

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')
IMG_PATH = './figs/ims/var_dists/'

g = sns.jointplot(
    x = coeffs.loc[coeffs['group'] == 'good', 'gk1'], y = coeffs.loc[coeffs['group'] == 'good', 'gk2'], kind = 'kde'
)
g.ax_joint.plot(
    coeffs.loc[coeffs['group'] == 'good', 'gk1'],
    coeffs.loc[coeffs['group'] == 'good', 'gk2'],
    'ko', markeredgecolor = 'white'
)
g.fig.set_size_inches(4, 4)
plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'gk_dist_KGIF_-60mV_poscoeffs.png')

plt.show()

#%% SWARMPLOT OF GK ESTIMATES

plt.figure(figsize = (4, 2.5))
plt.subplot(111)
plt.title('5HT constrained $\\bar{{g}}$ estimates')
plt.ylim(0, 0.017)
sns.swarmplot(
    x = 'variable', y = 'value', hue = 'group',
    data = coeffs.loc[:, ['gk1', 'gk2', 'group']].melt(id_vars = ['group']),
    palette = ['k', 'r'], clip_on = False, linewidth = 0.5, edgecolor = 'gray'
)
plt.xlabel('')
plt.ylabel('Conductance (uS)')

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'gk_swarm_KGIF_-60mV_poscoeffs.png')

plt.show()
