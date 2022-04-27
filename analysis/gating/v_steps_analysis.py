#%% IMPORT MODULES

from __future__ import division

import pickle
import os

import numpy as np
import pandas as pd

from grr.cell_class import Cell, subtract_baseline, subtract_leak
from grr.CurveFit import fit_gating_curve


#%% LOAD DATA

OUTPUT_PATH = os.path.join('data', 'processed', 'gating')
GATING_PATH = os.path.join('data', 'raw', '5HT', 'gating')

# Load gating data
gating = Cell().read_ABF(
    [
        os.path.join(GATING_PATH, '18411002.abf'),
        os.path.join(GATING_PATH, '18411010.abf'),
        os.path.join(GATING_PATH, '18411017.abf'),
        os.path.join(GATING_PATH, '18411019.abf'),
        os.path.join(GATING_PATH, 'c0_inact_18201021.abf'),
        os.path.join(GATING_PATH, 'c1_inact_18201029.abf'),
        os.path.join(GATING_PATH, 'c2_inact_18201034.abf'),
        os.path.join(GATING_PATH, 'c3_inact_18201039.abf'),
        os.path.join(GATING_PATH, 'c4_inact_18213011.abf'),
        os.path.join(GATING_PATH, 'c5_inact_18213017.abf'),
        os.path.join(GATING_PATH, 'c6_inact_18213020.abf'),
        os.path.join(GATING_PATH, '18619018.abf'),
        os.path.join(GATING_PATH, '18614032.abf'),
    ]
)

#%% PROCESS RAW GATING DATA

# Define time intervals from which to grab data.
xrange_baseline = slice(0, 2000)
xrange_test = slice(3500, 4000)
xrange_peakact = slice(26140, 26160)
xrange_ss = slice(55000, 56000)
xrange_peakinact = slice(56130, 56160)

# Format will be [channel, sweep, cell]
# Such that we can use plt.plot(pdata[0, :, :], pdata[1, :, :], '-') to plot I over V by cell.

shape_pdata = (2, gating[0].shape[2], len(gating))

peakact_pdata = np.empty(shape_pdata)
ss_pdata = np.empty(shape_pdata)
peakinact_pdata = np.empty(shape_pdata)

for i, cell in enumerate(gating):

    cell = subtract_baseline(cell, xrange_baseline, 0)
    cell = subtract_leak(cell, xrange_baseline, xrange_test)

    # Average time windows to get leak-subtracted IA and KSlow currents
    peakact_pdata[:, :, i] = cell[:, xrange_peakact, :].mean(axis=1)
    ss_pdata[:, :, i] = cell[:, xrange_ss, :].mean(axis=1)

    # Get prepulse voltage for peakinact
    peakinact_pdata[0, :, i] = cell[0, xrange_peakinact, :].mean(axis=0)
    peakinact_pdata[1, :, i] = cell[1, xrange_peakact, :].mean(axis=0)

peakact_pdata[0, :, :] /= peakact_pdata[1, :, :] - -101
ss_pdata[0, :, :] /= ss_pdata[1, :, :] - -101
peakinact_pdata[0, :, :] /= (
    peakinact_pdata[1, -1, :] - -101
)  # Since driving force is same for all sweeps.

# Average out small differences in cmd between cells due to Rs comp
peakact_pdata[1, :, :] = peakact_pdata[1, :, :].mean(axis=1, keepdims=True)
ss_pdata[1, :, :] = ss_pdata[1, :, :].mean(axis=1, keepdims=True)
peakinact_pdata[1, :, :] = peakinact_pdata[1, :, :].mean(axis=1, keepdims=True)

# Remove contribution of KSlow to apparent inactivation peak.
peakinact_pdata[0, :, :] -= ss_pdata[0, :, :]

# Pickle in case needed.

with open(os.path.join(OUTPUT_PATH, 'peakact_pdata.dat'), 'wb') as f:
    pickle.dump(peakact_pdata, f)

with open(os.path.join(OUTPUT_PATH, 'ss_pdata.dat'), 'wb') as f:
    pickle.dump(ss_pdata, f)

with open(os.path.join(OUTPUT_PATH, 'peakinact_pdata.dat'), 'wb') as f:
    pickle.dump(peakinact_pdata, f)


#%% FIG SIGMOID CURVES TO GATING DATA

peakact_params, peakact_fittedpts = fit_gating_curve(
    peakact_pdata, [12, 1, -30]
)
peakinact_params, peakinact_fittedpts = fit_gating_curve(
    peakinact_pdata, [12, -1, -60]
)
ss_params, ss_fittedpts = fit_gating_curve(ss_pdata, [12, 1, -25])

for name, obj in {
    'peakact_fittedpts': peakact_fittedpts,
    'peakinact_fittedpts': peakinact_fittedpts,
    'ss_fittedpts': ss_fittedpts,
}.iteritems():
    with open(os.path.join(OUTPUT_PATH, name + '.dat'), 'wb') as f:
        pickle.dump(obj, f)
        f.close()

param_pickle_df = pd.DataFrame(
    {'m': peakact_params, 'h': peakinact_params, 'n': ss_params},
    index=('A', 'k', 'V_half'),
)

with open(os.path.join(OUTPUT_PATH, 'gating_params.dat'), 'wb') as f:
    pickle.dump(param_pickle_df, f)
