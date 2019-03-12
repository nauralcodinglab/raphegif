#%% IMPORT MODULES

from __future__ import division

import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize

import sys
sys.path.append('./analysis/gating/')


from cell_class import Cell, Recording
import src.pltools as pltools


#%% LOAD GATING DATA

FIGDATA_PATH = './figs/figdata/'

with open(FIGDATA_PATH + 'peakact_pdata.pyc', 'rb') as f:
    peakact_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'ss_pdata.pyc', 'rb') as f:
    ss_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'peakinact_pdata.pyc', 'rb') as f:
    peakinact_pdata = pickle.load(f)

pdata_dict = {
    'peakact': peakact_pdata,
    'peakinact': peakinact_pdata,
    'ss': ss_pdata
}

del peakact_pdata, peakinact_pdata, ss_pdata


#%% FIG SIGMOID CURVES TO GATING DATA

max_normalize = lambda cell_channel: cell_channel / cell_channel.max(axis = 0)

#%%

def expand_basis(X, nodes, order = [2, 3], flip_X = False):

    if not flip_X:
        softplus_col = np.log(1 + 1.1**(X - 50))
    else:
        softplus_col = np.log(1 + 1.1**(-(X - 50)))

    X_expanded = [np.ones_like(X), softplus_col]

    for i, node in enumerate(nodes):
        for j in order:
            if not flip_X:
                X_expanded.append(np.clip(X - node, 0, None) ** j)
            else:
                X_expanded.append(np.clip(-(X-node), 0, None) ** j)

    return np.array(X_expanded).T


def OLS(X, y):

    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)
    betas = np.linalg.solve(XTX, XTy)

    return betas

betas_rounded = {
    'peakact': [4.469e-2, 8.973e2, -2.349e-4, 1.468e-5, 3.469e-4, -1.171e-4],
    'peakinact': [-6.495e-2, 9.448e-3, -1.267e-3, 7.340e-6, 2.715e-4, 1.316e-5],
    'ss': [-1.043e-1, 9.824e2, -3.427e-4, 1.274e-5, -3.828e-4, -8.303e-6]
}

nodes = [-60, -40]
plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')
IMG_PATH = './figs/ims/gating/'
plt.figure(figsize = (6.5, 4))
i = 1
for key, pdata in pdata_dict.iteritems():

    X_tmp = pdata[1, :, :].flatten()
    y_tmp = max_normalize(pdata[0, :, :]).flatten()

    if key != 'peakinact':
        X_expanded = expand_basis(X_tmp, nodes)
    else:
        X_expanded = expand_basis(X_tmp, nodes, flip_X = True)

    betas_tmp = OLS(X_expanded, y_tmp)
    print(betas_tmp)
    yhat = np.dot(X_expanded, betas_tmp)

    plt.subplot(1, len(pdata_dict.keys()), i)
    plt.title('{}'.format(key))
    for j, knot in enumerate(nodes):
        if j == 0:
            plt.axvline(
                knot, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10),
                label = 'Spline knots'
            )
        else:
            plt.axvline(knot, color = 'k', ls = '--', lw = 0.5, dashes = (10, 10))
    plt.plot(pdata[1, :, :], max_normalize(pdata[0, :, :]), 'k-', lw = 0.5, alpha = 0.5)
    sup_tmp = np.linspace(-85, -17, 500)
    if key != 'peakinact':
        support = expand_basis(sup_tmp, nodes)
    else:
        support = expand_basis(sup_tmp, nodes, flip_X = True)
    plt.plot(sup_tmp, np.dot(support, betas_tmp), 'b-', label = 'Spline fit')
    #plt.plot(sup_tmp, np.dot(support, betas_rounded[key]), 'g-')
    plt.xlabel('$V$ (mV)')

    if i == 1:
        plt.ylabel(r'$\frac{g}{g_{ref}}$')
        plt.legend()

    i += 1

plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'cubequad_splinefit.png')

plt.show()
