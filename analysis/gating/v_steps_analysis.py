#%% IMPORT MODULES

from __future__ import division

import os
from textwrap import dedent

import numpy as np
import pandas as pd
import h5py
from ezephys.rectools import ABFLoader

from grr.cell_class import Cell, subtract_baseline, subtract_leak
from grr.CurveFit import fit_gating_curve


#%% LOAD DATA

OUTPUT_PATH = os.path.join(os.getenv('DATA_PATH'), 'processed', '5HT')
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH, mode=755)

# Load gating data
rec_index = pd.read_csv(
    os.path.join(os.getenv('DATA_PATH'), 'raw', '5HT', 'gating', 'index.csv')
)

l = ABFLoader()
ACCESS_RESISTANCE_CUTOFF = 30.0
INCLUSION_QUERY = (
    'access_resistance_megaohm <= @ACCESS_RESISTANCE_CUTOFF and include'
)
gating = l.load(
    [
        os.path.join(os.getenv('DATA_PATH'), 'raw', '5HT', 'gating', fname)
        for fname in rec_index.query(INCLUSION_QUERY)['filename']
    ]
)
del l

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

E_K = 101.  # Reversal potential of potassium in mV.
peakact_pdata[0, :, :] /= peakact_pdata[1, :, :] - E_K
ss_pdata[0, :, :] /= ss_pdata[1, :, :] - E_K
peakinact_pdata[0, :, :] /= (
    peakinact_pdata[1, -1, :] - E_K
)  # Since driving force is same for all sweeps.

# Average out small differences in cmd between cells due to Rs comp
peakact_pdata[1, :, :] = peakact_pdata[1, :, :].mean(axis=1, keepdims=True)
ss_pdata[1, :, :] = ss_pdata[1, :, :].mean(axis=1, keepdims=True)
peakinact_pdata[1, :, :] = peakinact_pdata[1, :, :].mean(axis=1, keepdims=True)

# Remove contribution of KSlow to apparent inactivation peak.
peakinact_pdata[0, :, :] -= ss_pdata[0, :, :]

# Pickle in case needed.

with h5py.File(
    os.path.join(OUTPUT_PATH, 'gating_pdata_room_temp.h5'), 'w'
) as f:
    f.attrs['description'] = dedent(
        """
        Voltage-dependence of whole-cell currents in 5-HT neurons at room
        temperature.

        5-HT neurons express the transient voltage-dependent potassium current
        I_A as well as another non-inactivating voltage-dependent potassium
        current.

        The arrays in this file contain measurements of the amount of whole
        cell that can be used to construct voltage-dependence curves for the
        activation and inactivation gates of I_A and the activation gate of
        the non-inactivating current (`peakact`, `peakinact`, and `ss`
        datasets, respectively). The dimensionality of each array is
        `[channel, measurement, cell]`, where `channel == 0` is the current in
        pA and `channel == 1` is the voltage at which the current was recorded
        in mV. `plt.plot(d[1, ...] d[0, ...])` can be used to plot the gating
        curves of all cells, where `d` is one of the arrays in this file.

        """[1:]
    )
    f.create_dataset('peakact', data=peakact_pdata)
    f.create_dataset('peakinact', data=peakinact_pdata)
    f.create_dataset('ss', data=ss_pdata)
    f.close()

#%% FIG SIGMOID CURVES TO GATING DATA

peakact_params, peakact_fittedpts = fit_gating_curve(
    peakact_pdata, [12, 1, -30]
)
peakinact_params, peakinact_fittedpts = fit_gating_curve(
    peakinact_pdata, [12, -1, -60]
)
ss_params, ss_fittedpts = fit_gating_curve(ss_pdata, [12, 1, -25])

with h5py.File(
    os.path.join(OUTPUT_PATH, 'gating_fittedpts_room_temp.h5'), 'w'
) as f:
    f.attrs['description'] = dedent(
        """
        Voltage-dependence of whole-cell currents in 5-HT neurons at room
        temperature.

        5-HT neurons express the transient voltage-dependent potassium current
        I_A as well as another non-inactivating voltage-dependent potassium
        current.

        The arrays in this file contain sigmoid curves fitted to measurements
        of current over voltage for several cells.

        """[1:]
    )
    f.create_dataset('peakact', data=peakact_fittedpts)
    f.create_dataset('peakinact', data=peakinact_fittedpts)
    f.create_dataset('ss', data=ss_fittedpts)
    f.close()

with open(os.path.join(OUTPUT_PATH, 'gating_params.csv'), 'w') as f:
    included_recs = rec_index.query(INCLUSION_QUERY)
    f.write(
        (
            '# num_cells={}\n'
            '# access_resistance_cutoff_megaohm={}\n'
            '# access_resistance_mean_megaohm={:.2f}\n'
            '# access_resistance_std_megaohm={:.2f}\n'
            '# m="I_A activation gate"\n'
            '# h="I_A inactivation gate"\n'
            '# n="Non-inactivating current activation gate"\n'
            '# A="Sigmoid scaling factor"\n'
            '# k="Sigmoid slope"\n'
            '# V_half="Sigmoid location (half-activation voltage in mV)"\n'
        ).format(
            included_recs.shape[0],
            ACCESS_RESISTANCE_CUTOFF,
            included_recs['access_resistance_megaohm'].mean(),
            included_recs['access_resistance_megaohm'].std(),
        )
    )
    pd.DataFrame(
        {'m': peakact_params, 'h': peakinact_params, 'n': ss_params},
        index=('A', 'k', 'V_half'),
    ).to_csv(f)
    f.close()
