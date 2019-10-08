#%% IMPORT MODULES

from __future__ import division

import os

import copy; from copy import deepcopy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from scipy import stats
from scipy import optimize
import seaborn as sns
import pandas as pd

import sys
sys.path.append('./analysis/regression_tinkering')

from grr.Experiment import Experiment
from grr.GIF import GIF
from grr.resGIF import resGIF
from grr.CalciumGIF import CalciumGIF
from grr.iGIF import iGIF_NP
from grr.iGIF import iGIF_VR
from grr.Filter_Rect import Filter_Rect_LogSpaced
from grr.Filter_Exps import Filter_Exps
from grr.SpikeTrainComparator import intrinsic_reliability
from model_evaluation import *

from grr import pltools
from grr.Tools import gagProcess

#%% LOAD DATA

DATA_PATH = os.path.join('data', 'processed', 'GABA_fastnoise')
with open(os.path.join(DATA_PATH, 'gaba_goodcells.ldat'), 'rb') as f:
    experiments = pickle.load(f)
    f.close()


#%% FIT GIFs

MOD_PATH = os.path.join('data', 'models', 'GABA')

GIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = GIF(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fit(expt, DT_beforeSpike=1.5)

    GIFs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_gifs.mod'), 'wb') as f:
    pickle.dump(GIFs, f)
    f.close()

#%% FIT ResGIFs

resGIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = resGIF(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fitSubthresholdDynamics(expt, DT_beforeSpike=1.5, plot = True, Vmin = -65.)

    resGIFs.append(tmp_GIF)

with open(os.path.join(MOD_PATH, 'gaba_resgifs.mod'), 'wb') as f:
    pickle.dump(resGIFs, f)
    f.close()


#%% FIT CaGIFs

CaGIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = CalciumGIF(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        X, y = build_Xy(expt, GIFmod = tmp_GIF)
        mask = X[:, 0] > -80

        betas = optimize.lsq_linear(
            X[mask, :], y[mask],
            bounds = (
                np.concatenate(([-np.inf, 0, -np.inf, 0, 0], np.full(X.shape[1] - 5, -np.inf))),
                np.concatenate(([0, np.inf, np.inf, np.inf, np.inf], np.full(X.shape[1] - 5, np.inf)))
            )
        )['x'].tolist()

        #var_expl_ = var_explained(X, betas, y)

        coeffs = pd.DataFrame([betas])
        tmp = convert_betas(coeffs)

        tmp_GIF.C = tmp.loc[0, 'C']
        tmp_GIF.gl = tmp.loc[0, 'gl']
        tmp_GIF.El = tmp.loc[0, 'El']
        tmp_GIF.gbar_K1 = tmp.loc[0, 'gk1']
        tmp_GIF.gbar_K2 = tmp.loc[0, 'gk2']
        tmp_GIF.eta.setFilter_Coefficients(-np.array(tmp.loc[0, [x for x in tmp.columns if 'AHP' in x]].tolist()))

        # Fit threshold dynamics.
        tmp_GIF.fitVoltageReset(expt, tmp_GIF.Tref, False)
        tmp_GIF.fitStaticThreshold(expt)
        tmp_GIF.fitThresholdDynamics(expt)

    CaGIFs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_cagifs.mod'), 'wb') as f:
    pickle.dump(CaGIFs, f)
    f.close()


#%% FIT iGIFs

iGIFs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = iGIF_NP(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fit(
            expt, DT_beforeSpike=1.5,
            theta_tau_all = np.logspace(np.log2(1), np.log2(100), 7, base = 2),
            last_bin_constrained = True, do_plot = True
        )

    iGIFs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_igifs.mod'), 'wb') as f:
    pickle.dump(iGIFs, f)
    f.close()

#%% FIT iGIF_VRs
"""Fit iGIF subclass with variable reset rule.
"""

iGIF_VRs = []

for expt in experiments:

    with gagProcess():

        tmp_GIF = iGIF_VR(0.1)
        tmp_GIF.name = expt.name

        # Define parameters
        tmp_GIF.Tref = 4.0

        tmp_GIF.eta = Filter_Rect_LogSpaced()
        tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)

        tmp_GIF.gamma = Filter_Exps()
        tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

        # Define the ROI of the training set to be used for the fit.
        for tr in expt.trainingset_traces:
            tr.setROI([[2000, 58000]])
        for tr in expt.testset_traces:
            tr.setROI([[500, 9500]])

        tmp_GIF.fit(
            expt, DT_beforeSpike=1.5,
            theta_tau_all = np.logspace(np.log2(1), np.log2(100), 7, base = 2),
            do_plot = True
        )

    iGIF_VRs.append(tmp_GIF)

    tmp_GIF.printParameters()

with open(os.path.join(MOD_PATH, 'gaba_igif_vrs.mod'), 'wb') as f:
    pickle.dump(iGIF_VRs, f)
    f.close()

#%% FIT AUGMENTEDGIFs

Opt_KGIFs = []

full_coeffs = []

for i, expt in enumerate(experiments):

    print('Fitting GIF to {} ({:.1f}%)'.format(expt.name, 100 * (i + 1) / len(experiments)))

    tmp_GIF = AugmentedGIF(0.1)

    # Define parameters
    tmp_GIF.Tref = 4.0

    tmp_GIF.eta = Filter_Rect_LogSpaced()
    tmp_GIF.eta.setMetaParameters(length=500.0, binsize_lb=2.0, binsize_ub=1000.0, slope=4.5)


    tmp_GIF.gamma = Filter_Exps()
    tmp_GIF.gamma.setFilter_Timescales([30, 300, 3000])

    # Define the ROI of the training set to be used for the fit
    for tr in expt.trainingset_traces:
        tr.setROI([[1000,59000]])
    for tr in expt.testset_traces:
        tr.setROI([[500, 9500]])

    coeffs = []

    for h_tau in np.logspace(np.log2(10), np.log2(300), 10, base = 2):

        print '\rFitting h_tau = {:.1f}ms'.format(h_tau),

        tmp_GIF.h_tau = h_tau

        X, y = build_Xy(expt, GIFmod = tmp_GIF)
        mask = X[:, 0] > -80

        betas = optimize.lsq_linear(
            X[mask, :], y[mask],
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

with open(os.path.join(MOD_PATH, 'gaba_kgifs.mod'), 'wb') as f:
    pickle.dump(Opt_KGIFs, f)
    f.close()
