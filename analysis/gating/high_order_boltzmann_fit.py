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
sys.path.append('./figs/scripts')

from cell_class import Cell, Recording
import pltools


#%% LOAD GATING DATA

FIGDATA_PATH = './figs/figdata/'

with open(FIGDATA_PATH + 'peakact_pdata.pyc', 'rb') as f:
    peakact_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'ss_pdata.pyc', 'rb') as f:
    ss_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'peakinact_pdata.pyc', 'rb') as f:
    peakinact_pdata = pickle.load(f)


#%% FIG SIGMOID CURVES TO GATING DATA

max_normalize = lambda cell_channel: cell_channel / cell_channel.max(axis = 0)



def compute_residuals(p, func, Y, X):

    """Compute residuals of a fitted curve.

    Inputs:
        p       -- vector of function parameters
        func    -- a callable function
        Y       -- real values
        X       -- vector of points on which to compute fitted values

    Returns:
        Array of residuals.
    """

    if len(Y) != len(X):
        raise ValueError('Y and X must be of the same length.')

    Y_hat = func(p, X)

    return Y - Y_hat


def optimizer_wrapper(pdata, p0, max_norm = True, func = 'sigmoid_default'):

    """
    Least-squares optimizer

    Uses `compute_residuals` as the loss function to optimize `sigmoid_curve`

    Returns:

    Tupple of parameters and corresponding curve.
    Curve is stored as a [channel, sweep] np.ndarray; channels 0 and 1 should correspond to I and V, respectively.
    Curve spans domain of data used for fitting.
    """

    if func == 'sigmoid_default':
        func = sigmoid_curve

    X = pdata[1, :, :].flatten()

    if max_norm:
        y = max_normalize(pdata[0, :, :]).flatten()
    else:
        y = pdata[0, :, :].flatten()

    p = optimize.least_squares(compute_residuals, p0, kwargs = {
    'func': func,
    'X': X,
    'Y': y
    })['x']

    no_pts = 500

    fitted_points = np.empty((2, no_pts))
    x_min = pdata[1, :, :].mean(axis = 1).min()
    x_max = pdata[1, :, :].mean(axis = 1).max()
    fitted_points[1, :] = np.linspace(x_min, x_max, no_pts)
    fitted_points[0, :] = func(p, fitted_points[1, :])

    return p, fitted_points

def simple_plot(p, x, y, func, no_points = 500):

    fitted_x = np.linspace(-80, -20, no_points)
    fitted_y = func(p, fitted_x)

    plt.figure()

    plt.subplot(111)
    plt.plot(x, y, 'b-', lw = 0.7, alpha = 0.8)
    plt.plot(fitted_x, fitted_y, '--', color = 'gray', lw = 2)

    plt.show()


#%%

pdata_dict = {
    'peakact': peakact_pdata,
    'peakinact': peakinact_pdata,
    'ss': ss_pdata
}
p0_dict = {
    'peakact': [12, 1, -30],
    'peakinact': [12, -1, -80],
    'ss': [12, 1, -25]
}
var_explained = {
    'curve': [],
    'order': [],
    've': []
}

for key, pdata in pdata_dict.iteritems():

    for order in [1, 2, 3, 4]:

        def sigmoid_tmp(p, V):
            if len(p) != 3:
                raise ValueError('p must be vector-like with len 3.')

            A = p[0]
            k = p[1]
            V0 = p[2]

            return A / (1 + np.exp(-k * (V - V0)))**order

        params, fittedpts = optimizer_wrapper(pdata, p0_dict[key], func = sigmoid_tmp)
        simple_plot(params, pdata[1, :, :], pdata[0, :, :] / pdata[0, :, :].max(axis = 0), sigmoid_tmp)

        SSE = np.mean(compute_residuals(
            params, sigmoid_tmp, (pdata[0, :, :] / pdata[0, :, :].max(axis = 0)).flatten(), pdata[1, :, :].flatten()
        )**2)
        var_explained_ = (1 - (SSE / np.var(pdata[0, :, :] / pdata[0, :, :].max(axis = 0)))) * 100.

        print('{} {} order var explained: {:.1f}%'.format(key, order, var_explained_))

        if var_explained_ < 0:
            var_explained_ = 0

        var_explained['curve'].append(key)
        var_explained['order'].append(order)
        var_explained['ve'].append(var_explained_)

var_explained = pd.DataFrame(var_explained)

#%%

IMG_PATH = './figs/ims/gating/'

plt.figure(figsize = (6, 4))

ax = plt.subplot(111)
ax.set_title('Gating curves fitted with nth order Boltzmann')
sns.barplot(x = 'curve', y = 've', hue = 'order', data = var_explained)
ax.set_xticklabels(['Kslow', 'IA activation', 'IA inactivation'])
ax.set_ylabel('Variance explained')
ax.set_xlabel('')
ax.set_ylim(70, 100)

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'boltzmann_order.png', dpi = 300)

plt.show()
