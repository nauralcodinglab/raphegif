#%% IMPORT MODULES

from __future__ import division

import pickle
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import optimize

import sys
sys.path.append(os.path.join('analysis', 'regression_tinkering'))
from model_evaluation import *

from src.AugmentedGIF import AugmentedGIF
from src.Filter_Exps import Filter_Exps
from src.Tools import gagProcess


#%% READ IN DATA

DATA_PATH = os.path.join('data', 'processed', '5HT_fastnoise')

with open(os.path.join(DATA_PATH, '5HT_goodcells.ldat'), 'rb') as f:
    experiments = pickle.load(f)
    f.close()


#%% FIT KGIFS

MODEL_PATH = os.path.join('data', 'models', '5HT')

Opt_KGIFs = []
full_coeffs = []

for i, expt in enumerate(experiments):

    print('Fitting AugmentedGIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

    tmp_GIF = AugmentedGIF(0.1)
    tmp_GIF.name = expt.name

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

    coeffs = []

    for h_tau in np.logspace(np.log2(10), np.log2(150), 10, base = 2):

        print '\rFitting h_tau = {:.1f}ms'.format(h_tau),

        tmp_GIF.h_tau = h_tau

        X, y = build_Xy(expt, GIFmod = tmp_GIF)
        mask = X[:, 0] > -80

        betas = optimize.lsq_linear(
            X, y,
            bounds = (
                np.concatenate(([-np.inf, 0, -np.inf, 0, 0], np.full(X.shape[1] - 5, -np.inf))),
                np.concatenate(([0, np.inf, np.inf, np.inf, np.inf], np.full(X.shape[1] - 5, np.inf)))
            )
        )['x'].tolist()

        var_expl_ = var_explained(X, betas, y)

        group_ = h_tau

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

    full_coeffs.append(tmp)

    # Assign coeffs of best model
    best_mod_ind = np.argmax(tmp['var_explained'])

    tmp_GIF.C = tmp.loc[best_mod_ind, 'C']
    tmp_GIF.gl = tmp.loc[best_mod_ind, 'gl']
    tmp_GIF.El = tmp.loc[best_mod_ind, 'El']
    tmp_GIF.gbar_K1 = tmp.loc[best_mod_ind, 'gk1']
    tmp_GIF.h_tau = tmp.loc[best_mod_ind, 'group']
    tmp_GIF.gbar_K2 = tmp.loc[best_mod_ind, 'gk2']
    tmp_GIF.eta.setFilter_Coefficients(-np.array(tmp.loc[best_mod_ind, [x for x in tmp.columns if 'AHP' in x]].tolist()))

    # Fit threshold params
    print 'Fitting threshold dynamics.'
    with gagProcess():
        tmp_GIF.fitVoltageReset(expt, tmp_GIF.Tref, False)
        tmp_GIF.fitStaticThreshold(expt)
        tmp_GIF.fitThresholdDynamics(expt)

    Opt_KGIFs.append(tmp_GIF)

print('Done!')

with open(os.path.join(MODEL_PATH, 'serkgifs.lmod'), 'wb') as f:
    pickle.dump(Opt_KGIFs, f)
    f.close()

