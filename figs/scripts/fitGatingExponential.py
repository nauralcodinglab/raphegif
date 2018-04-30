#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt


#%% GET DATA

FIGDATA_PATH = './figs/figdata/'

with open(FIGDATA_PATH + 'peakact_pdata.pyc', 'rb') as f:
    peakact_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'ss_pdata.pyc', 'rb') as f:
    ss_pdata = pickle.load(f)

with open(FIGDATA_PATH + 'peakinact_pdata.pyc', 'rb') as f:
    peakinact_pdata = pickle.load(f)

#%%

max_normalize = lambda cell_channel: cell_channel / cell_channel.max(axis = 0)

y = np.log(max_normalize(peakact_pdata[0, :, :]).flatten())
X = np.ones((len(y), 2))
X[:, 0] = peakact_pdata[1, :, :].flatten()

y_nanmask = np.isnan(y)
y = y[~y_nanmask]
X = X[~y_nanmask, :]

XTX = np.dot(X.T, X)
XTX_inv = np.linalg.inv(XTX)
XTY = np.dot(X.T, y)
b = np.dot(XTX_inv, XTY)

y_hat = np.dot(X, b)

plt.figure()
plt.plot(X[:, 0], y, 'k.')
plt.plot(X[:, 0], y_hat, 'r-')
plt.show()
