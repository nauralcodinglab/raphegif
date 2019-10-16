#%% IMPORT MODULES

from __future__ import division

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append('./analysis/spk_timing')
from OhmicSpkPredictor import OhmicSpkPredictor
from grr.cell_class import Cell, Recording


#%% LOAD RECORDINGS

DATA_PATH = './data/raw/5HT/spk_time/'

inventory = pd.read_csv(DATA_PATH + 'index.csv')
ctrl_inventory = inventory.loc[np.logical_and(inventory['PE'] == 0, inventory['4AP'] == 0), :]
ctrl_inventory['cumcount'] = ctrl_inventory.groupby('Cell').cumcount()
fnames = ctrl_inventory.pivot('Cell', 'cumcount', values = 'Recording')

cells = []
for i in range(fnames.shape[0]):
    cells.append(Cell().read_ABF([DATA_PATH + fname for fname in fnames.iloc[i, :]]))

#%% FIT SPIKE LATENCY MODELS

baseline_recs = inventory.loc[np.logical_and(inventory['Cell'] == 'DRN332', inventory['4AP'] == 0), 'Recording']
_4AP_recs = inventory.loc[np.logical_and(inventory['Cell'] == 'DRN332', inventory['4AP'] == 1), 'Recording']

baseline_ = Cell().read_ABF([DATA_PATH + fname for fname in baseline_recs])
_4AP = Cell().read_ABF([DATA_PATH + fname for fname in _4AP_recs])

for r in baseline_:
    r.set_dt(0.1)
for r in _4AP:
    r.set_dt(0.1)

pred_baseline = deepcopy(OhmicSpkPredictor())
pred_baseline.add_recordings(baseline_, (0, 100), (5000, 5100), tau = (1000, 1700, 2700))
pred_baseline.scrape_data()

pred_4AP = deepcopy(OhmicSpkPredictor())
pred_4AP.add_recordings(_4AP[-3:], (0, 100), (5000, 5100), tau = (1000, 1700, 2700))
pred_4AP.scrape_data()

pred_baseline.fit_spks(verbose = True, force_tau = np.median(pred_baseline.taus), Vinf_guesses = np.linspace(-50, 50, 100))
pred_4AP.fit_spks(verbose = True, force_tau = np.median(pred_4AP.taus), Vinf_guesses = np.linspace(-50, 50, 100))

#%% MAKE PLOTS

# Plot latency and fits.
plt.figure()
plt.plot(pred_baseline.V0, pred_baseline.spks, 'k.')
plt.plot(pred_4AP.V0, pred_4AP.spks, 'r.')
V0_vec = np.linspace(-100, -55, 200)
plt.plot(V0_vec, pred_baseline.predict_spks(V0 = V0_vec, Vinf = pred_baseline.Vinf_est), 'k--')
plt.plot(V0_vec, pred_4AP.predict_spks(V0 = V0_vec, Vinf = pred_4AP.Vinf_est), 'r--')
plt.show()

# Plot residuals on latency fits.
inds_bl = np.argsort(pred_baseline.V0)
inds_4AP = np.argsort(pred_4AP.V0)
predicted_spks_4AP = pred_4AP.predict_spks(Vinf = pred_4AP.Vinf_est)
predicted_spks_bl = pred_baseline.predict_spks(Vinf = pred_baseline.Vinf_est)
plt.figure()
plt.axhline(0, ls = '--', color = 'k')
plt.plot(pred_4AP.V0[inds_4AP], pred_4AP.spks[inds_4AP] - predicted_spks_4AP[inds_4AP], 'r-')
plt.plot(pred_baseline.V0[inds_bl], pred_baseline.spks[inds_bl] - predicted_spks_bl[inds_bl], 'k-')
plt.show()
