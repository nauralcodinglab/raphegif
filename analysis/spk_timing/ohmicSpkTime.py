#%% IMPORT MODULES

from __future__ import division

from copy import deepcopy
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import sys
sys.path.append('./analysis/spk_timing')
from grr.cell_class import Cell, Recording
from OhmicSpkPredictor import OhmicSpkPredictor


#%% LOAD RECORDINGS

"""
DATA_PATH = './data/raw/5HT/spk_time/'

inventory = pd.read_csv(DATA_PATH + 'index.csv')

ctrl_inventory = inventory.loc[np.logical_and(inventory['PE'] == 0, inventory['4AP'] == 0), :]

ctrl_inventory['cumcount'] = ctrl_inventory.groupby('Cell').cumcount()

fnames = ctrl_inventory.pivot('Cell', 'cumcount', values = 'Recording')

cells = []
for i in range(fnames.shape[0]):

    cells.append(Cell().read_ABF([DATA_PATH + fname for fname in fnames.iloc[i, :]]))


#%% ASSIGN TEST CELL

test_cell = cells[0]
test_rec = test_cell[0]

test_rec.plot()
tmp = test_rec.fit_test_pulse((0, 100), (5000, 5100), V_clamp = False)


#%% TEST SPK FIT

pred = OhmicSpkPredictor()
pred.add_recordings(test_cell, (0, 100), (5000, 5100))
pred.scrape_data()
pred.plot()

#%%

pred.fit_spks(verbose = True)

pred.tau
pred.thresh
pred.Vinf_est
pred.Vinf.mean()

plt.figure()
plt.plot(pred.V0, pred.spks, 'k.')
V0_vec = np.linspace(-100, -40, 200)
plt.plot(V0_vec, pred.predict_spks(V0 = V0_vec, Vinf = pred.Vinf_est), 'k--')
plt.show()
"""


#%% multiprocessing

def worker(input_):

    i = input_[0]
    cell = input_[1]

    for rec in cell:
        rec.fit_test_pulse((0, 100), (5000, 5100), V_clamp = False)

    pred = OhmicSpkPredictor()
    pred.add_recordings(cell, (0, 100), (5000, 5100))
    pred.scrape_data()

    pred.fit_spks(verbose = True)

    f = plt.figure(figsize = (8, 4))
    p = f.add_subplot(111)

    p.plot(pred.V0, pred.spks, 'k.')
    V0_vec = np.linspace(-100, -40, 200)
    p.plot(V0_vec, pred.predict_spks(V0 = V0_vec, Vinf = pred.Vinf_est), 'k--')
    p.set_ylim(-10, p.get_ylim()[1])
    p.set_ylabel('Spike latency (ms)')
    p.set_xlabel('Pre-pulse voltage (mV)')

    f.savefig('./figs/ims/spk_latency/' + 'spk_latency_cell_{}.png'.format(i), dpi = 300)

    return pred


if __name__ == '__main__':

    pl = mp.Pool(8)

    preds = pl.map(worker, [(i, cell) for i, cell in enumerate(cells)])

    pl.close()

    print 'Done!'


#%%

%matplotlib qt5
plt.figure()
plt.axhline(0, color = 'k', linewidth = 0.5)

for i, pr in enumerate(preds):

    if i in [3, 4, 5]:
        continue

    predicted_spks = pr.predict_spks(Vinf = pr.Vinf_est)

    inds = np.argsort(pr.V0)

    plt.plot(pr.V0[inds], (np.array(pr.spks) - predicted_spks)[inds], '-', alpha = 0.3)


plt.show()
